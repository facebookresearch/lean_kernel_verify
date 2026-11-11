# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Launch facts the PTX frontend needs from the host: kernel parameters, CUtensorMap
descriptors and the tensors behind global addresses (design §5.1). Pure stdlib.

    ptx_params(ptx_text, entry=None) -> [{i, name, decl, kind, type, size, align}]
    decode(raw128) -> {dtype, elem_bits, tf32, rank, address, dims, strides_bytes, box,
                       estrides, interleave, swizzle, swizzle_atom, oob_nan, l2,
                       encoder_variant}
    kir_desc(decoded, elem_bytes) -> {shape, strides, box, elem_bytes, dim0_ratio}
    resolve(address, tensors, desc=None) -> [{tensor, storage, elem_off, match}]

Every refusal raises LaunchError with the reason.

The host adapters (frontend/ttgir, frontend/cute) build one LaunchRecord per launch in
dump.json from these: {kernel, entry, ptx: {file, sha256}, target, grid, block, cluster,
smem_dynamic, attrs, params: [ptx_params() entry + name, value?, fields?, tensormap?],
tensors, flags/config/specialization/sources/hinted/stale as today}. They check the
parameter list they pass against ptx_params (size and alignment of every parameter).

CUtensorMap layout (tiled mode; measured against cuTensorMapEncodeTiled on GB300, driver
13000, and CuTe DSL 4.4.2's software encoder). Little-endian u64 words w0..w15; dims
innermost first; k = dim index:

    w0[4:57]   global address >> 4
    w1[0:4]    encoder variant: 0 driver, 2 CuTe 4.4.2 (no KIR meaning); 1 is im2col
    w1[4:7]    rank - 1          w1[7:11]   device dtype code (_DTYPES)
    w1[11:13]  interleave        w1[13:15]  swizzle          w1[15]  OOB fill NaN
    w1[16]     TF32              w1[17:19]  L2 promotion     w1[19:21] swizzle atom
    w1[21]     encoder flag (set for large extents; no meaning)
    stride k (k = 1..rank-1, bytes >> 4): low 32 bits at bit 64 + 32k (w1[32:], w2,
               w3[:32]), high 4 bits at w3[32 + 4(k-1)]
    w4, w5, w6[0:32]   dims[k] - 1, 32 bits each
    w6[32 + 3k]        element stride[k] - 1, 3 bits
    w6[56:], w7[0:32]  box[k] - 1, 8 bits each
    w8, w9     driver-only hints (box bytes, swizzle); CuTe writes 0
    w10..w15   0
Fields of dims >= rank must be 0, like every bit outside a field: a descriptor with an
unknown bit is refused instead of guessed at.
"""

import math
import re


class LaunchError(Exception):
    """A parameter list, descriptor or address the frontend cannot represent exactly."""


# PTX fundamental types allowed in .param declarations -> bytes.
_TYPE_BYTES = {
    **{f"{k}8": 1 for k in "bus"}, **{f"{k}16": 2 for k in "bus"},
    **{f"{k}32": 4 for k in "bus"}, **{f"{k}64": 8 for k in "bus"},
    "b128": 16, "f16": 2, "bf16": 2, "f16x2": 4, "bf16x2": 4, "f32": 4, "f64": 8,
}  # fmt: skip

# `.param [.align A] .type [.ptr [.space] [.align P]] name[[N]]`; P is the pointee's
# alignment, not the parameter's.
_PARAM = re.compile(
    r"\.param\s+(?:\.align\s+(?P<align>\d+)\s+)?\.(?P<type>\w+)\s+"
    r"(?P<ptr>\.ptr(?:\s+\.(?:global|shared|const|local))?(?:\s+\.align\s+\d+)?\s+)?"
    r"(?P<name>[\w$%]+)\s*(?:\[\s*(?P<n>\d+)\s*\])?"
)
_ENTRY = re.compile(r"\.entry\s+([\w$%.]+)\s*(?:\(([^)]*)\))?")


def ptx_params(ptx_text: str, entry: str | None = None) -> list:
    """The parameters of a kernel, in order: {i, name, decl (normalized text), kind: 'ptr'
    (`.ptr`) | 'bytes' (an array, e.g. a CUtensorMap or a CuTe struct) | 'scalar', type
    (PTX type without the dot), size and align (bytes, of the parameter itself)}. `entry`
    picks one `.entry` of a module with several."""
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", ptx_text, flags=re.S)
    entries = {m.group(1): m.group(2) or "" for m in _ENTRY.finditer(text)}
    if entry is None:
        if len(entries) != 1:
            raise LaunchError(
                f"{len(entries)} .entry in the module ({', '.join(entries) or 'none'}): "
                "name one"
            )
        (entry,) = entries
    if entry not in entries:
        raise LaunchError(f"no .entry {entry} (have {', '.join(entries) or 'none'})")
    decls = [d.strip() for d in entries[entry].split(",")]
    return [_param(i, d) for i, d in enumerate(d for d in decls if d)]


def _param(i: int, decl: str) -> dict:
    decl = " ".join(decl.split())
    m = _PARAM.fullmatch(decl)
    if not m or m["type"] not in _TYPE_BYTES or (m["ptr"] and m["n"]):
        raise LaunchError(f"unsupported kernel parameter declaration: {decl}")
    size = _TYPE_BYTES[m["type"]]
    kind = "ptr" if m["ptr"] else "bytes" if m["n"] else "scalar"
    return {
        "i": i,
        "name": m["name"],
        "decl": decl,
        "kind": kind,
        "type": m["type"],
        "size": size * int(m["n"] or 1),
        "align": int(m["align"] or size),
    }


# Device dtype code (w1[7:11]) -> (CUtensorMapDataType name, element bits).
_DTYPES = {
    0: ("UINT8", 8), 1: ("UINT16", 16), 2: ("UINT32", 32), 3: ("INT32", 32),
    4: ("UINT64", 64), 5: ("INT64", 64), 6: ("FLOAT16", 16), 7: ("FLOAT32", 32),
    8: ("FLOAT32_FTZ", 32), 9: ("FLOAT64", 64), 10: ("BFLOAT16", 16),
    11: ("16U4_ALIGN8B", 4), 12: ("16U4_ALIGN16B", 4), 13: ("16U6_ALIGN16B", 6),
}  # fmt: skip
_INTERLEAVE = ("NONE", "16B", "32B")
_SWIZZLE = ("NONE", "32B", "64B", "128B")
# The atoms of the 128B swizzles (host enums 3..6).
_ATOM = ("16B", "32B", "32B_FLIP_8B", "64B")
_L2 = ("NONE", "64B", "128B", "256B")


def decode(raw: bytes) -> dict:
    """The encode parameters of a 128-byte tiled CUtensorMap (layout in the module doc).
    dims and box are in elements of the descriptor's dtype (4- or 6-bit for the packed
    formats), strides_bytes are dims 1..rank-1."""
    if len(raw) != 128:
        raise LaunchError(f"CUtensorMap of {len(raw)} bytes (want 128)")
    bits = int.from_bytes(raw, "little")
    seen = ((1 << 128) - 1) << 512  # w8, w9: driver-only hints, any value

    def field(lo: int, n: int) -> int:
        nonlocal seen
        mask = ((1 << n) - 1) << lo
        seen |= mask
        return (bits & mask) >> lo

    variant = field(64, 4)
    if variant == 1:
        raise LaunchError("im2col CUtensorMap (w1[0:4] = 1)")
    if variant not in (0, 2):
        raise LaunchError(f"unknown CUtensorMap encoder variant w1[0:4] = {variant}")
    rank = field(68, 3) + 1
    code = field(71, 4)
    if rank > 5 or code not in _DTYPES:
        raise LaunchError(f"CUtensorMap with rank {rank}, dtype code {code}")
    dtype, elem_bits = _DTYPES[code]
    interleave, tf32 = field(75, 2), field(80, 1)
    if interleave >= len(_INTERLEAVE) or (
        tf32 and dtype not in ("FLOAT32", "FLOAT32_FTZ")
    ):
        raise LaunchError(
            f"CUtensorMap interleave code {interleave}, TF32 bit on {dtype}"
        )
    out = {
        "dtype": dtype,
        "elem_bits": elem_bits,
        "tf32": bool(tf32),
        "rank": rank,
        "address": field(4, 53) << 4,
        "dims": [field(256 + 32 * k, 32) + 1 for k in range(rank)],
        "strides_bytes": [
            (field(64 + 32 * k, 32) | (field(224 + 4 * (k - 1), 4) << 32)) << 4
            for k in range(1, rank)
        ],
        "box": [field(440 + 8 * k, 8) + 1 for k in range(rank)],
        "estrides": [field(416 + 3 * k, 3) + 1 for k in range(rank)],
        "interleave": _INTERLEAVE[interleave],
        "swizzle": _SWIZZLE[field(77, 2)],
        "swizzle_atom": _ATOM[field(83, 2)],
        "oob_nan": bool(field(79, 1)),
        "l2": _L2[field(81, 2)],
        "encoder_variant": variant,
    }
    field(85, 1)  # encoder flag
    unknown = bits & ~seen
    if unknown:
        words = [(i, (unknown >> 64 * i) & ((1 << 64) - 1)) for i in range(16)]
        raise LaunchError(
            "CUtensorMap with unknown bits: "
            + ", ".join(f"w{i} & {w:#x}" for i, w in words if w)
        )
    return out


def kir_desc(d: dict, elem_bytes: int) -> dict:
    """A decoded descriptor as a KIR `desc` over elements of `elem_bytes` (the base
    tensor's), in KIR order (outermost first): {shape, strides (elements), box, elem_bytes,
    dim0_ratio}. Dim 0 is converted from the descriptor's dtype to KIR elements by
    dim0_ratio = (num, den): x -> x * num / den, exact or refused. That expresses packed
    sub-byte data in bytes (4-bit: (1, 2), 6-bit: (3, 4)) like the CuTe frontend
    (ops_mem.py, `pack`), and a dtype wider than the element in elements (`scale`). The
    transaction bytes of one TMA are prod(box) * elem_bytes: for the packed formats the
    packed global bytes, also for 16U4_ALIGN16B / 16U6_ALIGN16B whose shared-memory copy
    expands. Callers convert a dim-0 TMA coordinate c the same way, asserting
    c % den == 0 at runtime."""
    if d["interleave"] != "NONE" or any(e != 1 for e in d["estrides"]):
        raise LaunchError(
            f"TMA descriptor with interleave {d['interleave']}, element strides "
            f"{d['estrides']} (KIR boxes are dense)"
        )
    g = math.gcd(d["elem_bits"], 8 * elem_bytes)
    num, den = d["elem_bits"] // g, 8 * elem_bytes // g

    def dim0(x: int, what: str) -> int:
        if x * num % den:
            raise LaunchError(
                f"TMA {what} {x} x {d['elem_bits']} bits is not a whole number of "
                f"{elem_bytes}-byte elements"
            )
        return x * num // den

    strides = [1]
    for s in d["strides_bytes"]:
        if s % elem_bytes:
            raise LaunchError(f"TMA stride {s} B over {elem_bytes}-byte elements")
        strides.append(s // elem_bytes)
    shape = [dim0(d["dims"][0], "dim-0 extent"), *d["dims"][1:]]
    box = [dim0(d["box"][0], "dim-0 box"), *d["box"][1:]]
    return {
        "shape": shape[::-1],
        "strides": strides[::-1],
        "box": box[::-1],
        "elem_bytes": elem_bytes,
        "dim0_ratio": (num, den),
    }


_MATCH_ORDER = {"exact": 0, "inside": 1, "shifted": 2}


def resolve(address: int, tensors: dict, desc: dict | None = None) -> list:
    """The named tensors a global byte address may point into, best first:
    [{tensor, storage: [lo, hi), elem_off, match}] with elem_off = (address - data_ptr) /
    elem_bytes (signed) and match
      'exact'   the tensor's data pointer (a view starting there),
      'inside'  inside the tensor's storage (an aliasing view, an offset),
      'shifted' below the storage, but the in-extent span of `desc` (decoded descriptor at
                `address`) overlaps it and walks the tensor's strides: a base shifted
                back by whole rows, like a ragged TMA tensor (base - 2^30 rows);
                elem_off is negative.
    `tensors`: {name: {data_ptr, storage: [lo, hi), elem_bytes, shape, strides
    (elements)}}, the capture snapshot. Tensors with empty storage (zero-length tensors,
    data_ptr 0) never match, and an address that is not on an element boundary of a
    tensor does not match it. All candidates are reported: the caller refuses only when an
    annotation depends on the choice."""
    out: list[dict] = []
    for name in sorted(tensors):
        t = tensors[name]
        lo, hi = t["storage"]
        delta = address - t["data_ptr"]
        if hi <= lo or delta % t["elem_bytes"]:
            continue
        if delta == 0:
            match = "exact"
        elif lo <= address < hi:
            match = "inside"
        elif (
            desc is not None
            and address < lo < _span_end(desc)
            and _strides_agree(desc, t)
        ):
            match = "shifted"
        else:
            continue
        out.append(
            {
                "tensor": name,
                "storage": [lo, hi],
                "elem_off": delta // t["elem_bytes"],
                "match": match,
            }
        )
    return sorted(out, key=lambda c: _MATCH_ORDER[c["match"]])


def _span_end(d: dict) -> int:
    """One past the last byte within the descriptor's extents."""
    row = (d["dims"][0] * d["elem_bits"] + 7) // 8
    return (
        d["address"]
        + row
        + sum((n - 1) * s for n, s in zip(d["dims"][1:], d["strides_bytes"]))
    )


def _strides_agree(d: dict, t: dict) -> bool:
    """The tensor's non-contiguous dims (extent > 1) and the descriptor's dims 1.. walk
    the same set of byte strides (a ragged tensor adds a dim with an existing stride). A
    tensor without such a dim (1-D) is never a shifted candidate."""
    # ponytail: stride sets, not a dim-by-dim match; refine if two named tensors with the
    # same strides overlap one ragged span (both are reported, the caller refuses).
    eb = t["elem_bytes"]
    want = {s * eb for n, s in zip(t["shape"], t["strides"]) if n > 1 and s != 1}
    have = {s for n, s in zip(d["dims"][1:], d["strides_bytes"]) if n > 1 and s != eb}
    return bool(want) and want == have
