# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-Python tests of the PTX frontend's launch facts (frontend/ptx/launch.py).

    python3 -I tests/ptx/test_launch.py     (or the buck target :test_launch)

Kernel parameter lists of hand-written `.entry` headers shaped like Triton's (a host
descriptor is 1 + 2N parameters, two scratch pointers at the end) and CuTe's (tensors as
byte structs); CUtensorMap decoding of real driver and CuTe DSL 4.4.2 encodings
(fixtures/tmaps.py) and of synthesized descriptors over every field, with every bit
outside a field refused; the KIR view of a descriptor (packed sub-byte formats in bytes);
address resolution against named tensors (views, aliases, zero-length tensors, a ragged
TMA base shifted below its storage).
"""

import random
import re
import struct
import sys
import unittest
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    # Run as a script: the project root becomes the package `lkv_root`, so the relative
    # imports below resolve as they do in the buck target.
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.ptx"  # noqa: A001

from ...frontend.ptx.launch import (  # noqa: E402
    decode,
    kir_desc,
    LaunchError,
    ptx_params,
    resolve,
)
from .fixtures.tmaps import CUTE442, DRIVER, expected  # noqa: E402

# A Triton kernel with a rank-2 host descriptor (CUtensorMap, 2 shapes, 2 strides), a
# pointer, a float and an int, and the global / profile scratch pointers.
TRITON = """
.version 9.3
.target sm_100a
.address_size 64

    // .globl kern
.visible .entry kern(
    .param .align 64 .b8 kern_param_0[128],
    .param .b32 kern_param_1,
    .param .b32 kern_param_2,
    .param .b64 kern_param_3,
    .param .b64 kern_param_4,
    .param .u64 .ptr .global .align 1 kern_param_5,
    .param .f32 kern_param_6,
    .param .b32 kern_param_7,
    .param .u64 .ptr .global .align 1 kern_param_8,
    .param .u64 .ptr .global .align 1 kern_param_9
)
.reqntid 128
{
    ret;
}
"""

# CuTe tensors and derived values as byte structs; a 3-byte struct without .align.
CUTE = """
.extern .shared .align 1024 .b8 __dynamic_shmem__0[];

.visible .entry kernel_cutlass_k_0(
    .param .align 8 .b8 kernel_cutlass_k_0_param_0[16],
    .param .align 4 .b8 kernel_cutlass_k_0_param_1[12],
    .param .align 64 .b8 kernel_cutlass_k_0_param_2[128],
    .param .align 8 .b8 kernel_cutlass_k_0_param_3[40],
    .param .b8 kernel_cutlass_k_0_param_4[3],
    .param .u32 kernel_cutlass_k_0_param_5
)
.reqntid 512, 1, 1
{
    ret;
}
"""


def _p(params) -> list:
    return [(p["kind"], p["type"], p["size"], p["align"]) for p in params]


class ParamsTest(unittest.TestCase):
    def test_triton_descriptor_and_scratch(self):
        params = ptx_params(TRITON)
        self.assertEqual([p["i"] for p in params], list(range(10)))
        self.assertEqual(
            [p["name"] for p in params], [f"kern_param_{i}" for i in range(10)]
        )
        self.assertEqual(
            _p(params),
            [("bytes", "b8", 128, 64), ("scalar", "b32", 4, 4), ("scalar", "b32", 4, 4),
             ("scalar", "b64", 8, 8), ("scalar", "b64", 8, 8), ("ptr", "u64", 8, 8),
             ("scalar", "f32", 4, 4), ("scalar", "b32", 4, 4), ("ptr", "u64", 8, 8),
             ("ptr", "u64", 8, 8)],
        )  # fmt: skip
        self.assertEqual(
            params[5]["decl"], ".param .u64 .ptr .global .align 1 kern_param_5"
        )

    def test_cute_structs(self):
        self.assertEqual(
            _p(ptx_params(CUTE)),
            [("bytes", "b8", 16, 8), ("bytes", "b8", 12, 4), ("bytes", "b8", 128, 64),
             ("bytes", "b8", 40, 8), ("bytes", "b8", 3, 1), ("scalar", "u32", 4, 4)],
        )  # fmt: skip

    def test_scalar_types_arrays_and_spacing(self):
        ptx = (
            ".entry k ( .param .u8 a , .param .s16 b,.param .f64 c, .param .b128 d,"
            " .param .align 16 .b32 e [ 4 ], .param .u64 .ptr .align 16 f,"
            " .param .u64 .ptr .shared g /* generic and shared pointers */)"
        )
        self.assertEqual(
            _p(ptx_params(ptx)),
            [("scalar", "u8", 1, 1), ("scalar", "s16", 2, 2), ("scalar", "f64", 8, 8),
             ("scalar", "b128", 16, 16), ("bytes", "b32", 16, 16), ("ptr", "u64", 8, 8),
             ("ptr", "u64", 8, 8)],
        )  # fmt: skip

    def test_entries(self):
        both = TRITON + CUTE
        with self.assertRaisesRegex(LaunchError, "2 .entry in the module"):
            ptx_params(both)
        self.assertEqual(len(ptx_params(both, "kernel_cutlass_k_0")), 6)
        self.assertEqual(len(ptx_params(both, entry="kern")), 10)
        with self.assertRaisesRegex(LaunchError, "no .entry other"):
            ptx_params(both, "other")
        with self.assertRaisesRegex(LaunchError, "0 .entry"):
            ptx_params(".visible .func f()\n{\n}\n")
        self.assertEqual(ptx_params(".entry k()\n{ ret; }"), [])
        self.assertEqual(ptx_params(".entry k\n.reqntid 32\n{ ret; }"), [])
        # a commented-out .entry is not one
        self.assertEqual(len(ptx_params("// .entry old(.param .b32 x)\n" + TRITON)), 10)

    def test_unsupported_declarations(self):
        for decl in (
            ".param .pred p",
            ".param .u64 .ptr .global .align 1 p[2]",
            ".param .b8 p[]",
            ".param .align 8 .ptr .u64 p",
        ):
            with self.subTest(decl):
                with self.assertRaisesRegex(
                    LaunchError, "unsupported kernel parameter"
                ):
                    ptx_params(f".entry k({decl})")


# Field slots of the CUtensorMap layout (launch.py's module doc), transcribed word by word
# for the packer below: (word, low bit) per dim k.
STRIDE_LO = [(1, 32), (2, 0), (2, 32), (3, 0)]  # dims 1..4
DIMS = [(4, 0), (4, 32), (5, 0), (5, 32), (6, 0)]
BOX = [(6, 56), (7, 0), (7, 8), (7, 16), (7, 24)]
CODES = [
    "UINT8", "UINT16", "UINT32", "INT32", "UINT64", "INT64", "FLOAT16", "FLOAT32",
    "FLOAT32_FTZ", "FLOAT64", "BFLOAT16", "16U4_ALIGN8B", "16U4_ALIGN16B",
    "16U6_ALIGN16B",
]  # fmt: skip
CODE_BITS = [8, 16, 32, 32, 64, 64, 16, 32, 32, 64, 16, 4, 4, 6]
INTERLEAVE = ["NONE", "16B", "32B"]
SWIZZLE = ["NONE", "32B", "64B", "128B"]
ATOM = ["16B", "32B", "32B_FLIP_8B", "64B"]
L2 = ["NONE", "64B", "128B", "256B"]


def pack(d: dict, flag: int = 0, w8: int = 0, w9: int = 0) -> bytes:
    """The descriptor whose decode() is `d` (a decode() result)."""
    w = [0] * 16
    w[0] = d["address"]
    w[1] = (
        d["encoder_variant"] | (d["rank"] - 1) << 4 | CODES.index(d["dtype"]) << 7
        | INTERLEAVE.index(d["interleave"]) << 11 | SWIZZLE.index(d["swizzle"]) << 13
        | d["oob_nan"] << 15 | d["tf32"] << 16 | L2.index(d["l2"]) << 17
        | ATOM.index(d["swizzle_atom"]) << 19 | flag << 21
    )  # fmt: skip
    for k, s in enumerate(d["strides_bytes"]):
        word, lo = STRIDE_LO[k]
        w[word] |= ((s >> 4) & 0xFFFFFFFF) << lo
        w[3] |= (s >> 36) << (32 + 4 * k)
    for k in range(d["rank"]):
        word, lo = DIMS[k]
        w[word] |= (d["dims"][k] - 1) << lo
        w[6] |= (d["estrides"][k] - 1) << (32 + 3 * k)
        word, lo = BOX[k]
        w[word] |= (d["box"][k] - 1) << lo
    w[8], w[9] = w8, w9
    return struct.pack("<16Q", *w)


def known_bits(rank: int) -> int:
    """The bits of the fields of a rank-`rank` descriptor (and the hint words w8, w9)."""
    ones = {
        "address": ((1 << 53) - 1) << 4, "rank": rank, "dtype": "UINT8", "tf32": True,
        "encoder_variant": 0xF, "interleave": "NONE", "swizzle": "NONE", "l2": "NONE",
        "swizzle_atom": "16B", "oob_nan": True, "dims": [1 << 32] * rank,
        "box": [256] * rank, "estrides": [8] * rank,
        "strides_bytes": [(1 << 40) - 16] * (rank - 1),
    }  # fmt: skip
    w = list(
        struct.unpack("<16Q", pack(ones, flag=1, w8=(1 << 64) - 1, w9=(1 << 64) - 1))
    )
    w[1] |= (0xF << 7) | (0x3 << 11) | (0x3 << 13) | (0x3 << 17) | (0x3 << 19)
    w[1] |= ((1 << 3) - 1) << 4  # rank: all 3 bits
    return int.from_bytes(struct.pack("<16Q", *w), "little")


def random_fields(rng: random.Random) -> dict:
    code = rng.randrange(len(CODES))
    rank = rng.randint(1, 5)
    return {
        "dtype": CODES[code],
        "elem_bits": CODE_BITS[code],
        "tf32": CODES[code] in ("FLOAT32", "FLOAT32_FTZ") and rng.random() < 0.5,
        "rank": rank,
        "address": rng.randrange(1 << 53) << 4,
        "dims": [rng.randint(1, 1 << 32) for _ in range(rank)],
        "strides_bytes": [rng.randrange(1, 1 << 36) << 4 for _ in range(rank - 1)],
        "box": [rng.randint(1, 256) for _ in range(rank)],
        "estrides": [rng.randint(1, 8) for _ in range(rank)],
        "interleave": rng.choice(INTERLEAVE),
        "swizzle": rng.choice(SWIZZLE),
        "swizzle_atom": rng.choice(ATOM),
        "oob_nan": rng.random() < 0.5,
        "l2": rng.choice(L2),
        "encoder_variant": rng.choice((0, 2)),
    }


class DecodeTest(unittest.TestCase):
    def test_driver_encodings(self):
        for name, v in DRIVER.items():
            with self.subTest(name):
                self.assertEqual(decode(v["raw"]), expected(v["args"]))

    def test_cute_encodings(self):
        """CuTe 4.4.2's encodings decode with encoder_variant 2 and map back onto the
        tensors they were made from (test_gpu_tmap compares them with the driver's)."""
        for name, v in CUTE442.items():
            with self.subTest(name):
                d = decode(v["raw"])
                t = v["tensor"]
                self.assertEqual(
                    (d["encoder_variant"], d["address"]), (2, t["data_ptr"])
                )
                k = kir_desc(d, t["elem_bytes"])
                self.assertEqual(
                    (k["shape"], k["strides"], k["box"]),
                    (t["shape"], t["strides"], v["box"]),
                )
        self.assertEqual(
            kir_desc(decode(CUTE442["fp4"]["raw"]), 1)["dim0_ratio"], (1, 2)
        )
        self.assertEqual(
            kir_desc(decode(CUTE442["u16_format_over_u8"]["raw"]), 1)["dim0_ratio"],
            (2, 1),
        )

    def test_synthesized_fields(self):
        rng = random.Random(7)
        for _ in range(3000):
            d = random_fields(rng)
            raw = pack(
                d,
                flag=rng.randrange(2),
                w8=rng.randrange(1 << 64),
                w9=rng.randrange(1 << 64),
            )
            self.assertEqual(decode(raw), d)

    def test_every_bit_outside_a_field_is_refused(self):
        """Flip each of the 1024 bits of a valid descriptor: flips outside the fields are
        refused, flips inside are decoded, except where the new value is refused (encoder
        variant, rank, dtype code, interleave)."""
        checked = (
            0xF | 0x7 << 4 | 0xF << 7 | 0x3 << 11
        ) << 64  # w1 fields with refused values
        for rank in (1, 3, 5):
            d = random_fields(random.Random(rank))
            d.update(rank=rank, dims=[5] * rank, box=[3] * rank, estrides=[2] * rank,
                     strides_bytes=[4096] * (rank - 1), dtype="FLOAT32", elem_bits=32,
                     encoder_variant=0, interleave="NONE")  # fmt: skip
            base = int.from_bytes(pack(d), "little")
            accepted = 0
            for i in range(1024):
                try:
                    decode((base ^ (1 << i)).to_bytes(128, "little"))
                    accepted |= 1 << i
                except LaunchError:
                    pass
            known = known_bits(rank)
            with self.subTest(rank=rank):
                self.assertEqual(accepted & ~known, 0)
                self.assertEqual(known & ~accepted & ~checked, 0)

    def test_refusals(self):
        d = decode(DRIVER["u4_align16b"]["raw"])
        raw = pack(d)
        w = list(struct.unpack("<16Q", raw))

        def with_w1(value: int) -> bytes:
            return struct.pack("<16Q", w[0], value, *w[2:])

        cases = {
            "im2col CUtensorMap": with_w1(w[1] | 1),
            "unknown CUtensorMap encoder variant w1\\[0:4\\] = 3": with_w1(w[1] | 3),
            "rank 6": with_w1(w[1] | 5 << 4),
            "dtype code 14": with_w1(w[1] & ~(0xF << 7) | 14 << 7),
            "dtype code 15": with_w1(w[1] & ~(0xF << 7) | 15 << 7),
            "interleave code 3": with_w1(w[1] | 3 << 11),
            "TF32 bit on 16U4_ALIGN16B": with_w1(w[1] | 1 << 16),
            "unknown bits: w1 & 0x400000": with_w1(w[1] | 1 << 22),
            "unknown bits: w0 & 0x1, w15 & 0x8000000000000000": struct.pack(
                "<16Q", w[0] | 1, *w[1:15], 1 << 63
            ),
            "CUtensorMap of 127 bytes": raw[:127],
        }
        for msg, bad in cases.items():
            with self.subTest(msg):
                with self.assertRaisesRegex(LaunchError, msg):
                    decode(bad)


def _desc(**kw) -> dict:
    d = {"dtype": "BFLOAT16", "elem_bits": 16, "dims": [64, 256], "strides_bytes": [192],
         "box": [32, 64], "estrides": [1, 1], "interleave": "NONE"}  # fmt: skip
    return {**d, **kw}


class KirDescTest(unittest.TestCase):
    def test_dense(self):
        self.assertEqual(
            kir_desc(_desc(), 2),
            {"shape": [256, 64], "strides": [96, 1], "box": [64, 32], "elem_bytes": 2,
             "dim0_ratio": (1, 1)},
        )  # fmt: skip

    def test_dim0_units(self):
        """Dim 0 in KIR elements: packed 4- and 6-bit data in bytes, a format wider than the
        element scaled up, a narrower one scaled down (whole elements only)."""
        u4 = kir_desc(
            _desc(dtype="16U4_ALIGN16B", elem_bits=4, dims=[256, 8], box=[128, 4]), 1
        )
        self.assertEqual(
            (u4["shape"], u4["box"], u4["dim0_ratio"]), ([8, 128], [4, 64], (1, 2))
        )
        u6 = kir_desc(
            _desc(dtype="16U6_ALIGN16B", elem_bits=6, dims=[128, 8], box=[128, 4]), 1
        )
        self.assertEqual(
            (u6["shape"], u6["box"], u6["dim0_ratio"]), ([8, 96], [4, 96], (3, 4))
        )
        wide = kir_desc(_desc(dtype="UINT16", dims=[64, 8], box=[32, 4]), 1)
        self.assertEqual(
            (wide["shape"], wide["box"], wide["dim0_ratio"]),
            ([8, 128], [4, 64], (2, 1)),
        )
        narrow = kir_desc(
            _desc(dtype="UINT8", elem_bits=8, dims=[256, 8], box=[16, 4]), 4
        )
        self.assertEqual(
            (narrow["shape"], narrow["strides"], narrow["box"], narrow["dim0_ratio"]),
            ([8, 64], [48, 1], [4, 4], (1, 4)),
        )
        for d, eb, msg in (
            (_desc(dtype="16U4_ALIGN8B", elem_bits=4, dims=[255, 8]), 1, "dim-0 extent 255 x 4 bits"),
            (_desc(dtype="16U6_ALIGN16B", elem_bits=6, box=[130, 4]), 1, "dim-0 box 130 x 6 bits"),
            (_desc(dtype="UINT8", elem_bits=8, dims=[254, 8]), 4, "not a whole number of 4-byte"),
        ):  # fmt: skip
            with self.subTest(msg):
                with self.assertRaisesRegex(LaunchError, msg):
                    kir_desc(d, eb)

    def test_refusals(self):
        for d, eb, msg in (
            (_desc(estrides=[1, 2]), 2, re.escape("element strides [1, 2]")),
            (_desc(interleave="16B"), 2, "interleave 16B"),
            (_desc(strides_bytes=[48]), 32, "stride 48 B over 32-byte elements"),
        ):
            with self.subTest(msg):
                with self.assertRaisesRegex(LaunchError, msg):
                    kir_desc(d, eb)


def _tensor(ptr: int, shape: list, strides: list, eb: int, storage: list | None = None):
    span = 1 + sum((n - 1) * s for n, s in zip(shape, strides)) if all(shape) else 0
    return {"data_ptr": ptr, "storage": storage or [ptr, ptr + span * eb],
            "elem_bytes": eb, "shape": shape, "strides": strides}  # fmt: skip


def _names(cands) -> list:
    return [(c["tensor"], c["match"], c["elem_off"]) for c in cands]


B = 0x7F0000100000  # a storage start


class ResolveTest(unittest.TestCase):
    def test_views_and_aliases(self):
        """The probe's view base[:, 16:80] of a (256, 96) bf16 tensor: the view is exact,
        its base contains it at element 16; a byte alias of the base too, at byte 32."""
        store = [B, B + 256 * 96 * 2]
        ts = {
            "base": _tensor(B, [256, 96], [96, 1], 2),
            "view": _tensor(B + 32, [256, 64], [96, 1], 2, store),
            "bytes": _tensor(B, [256 * 96 * 2], [1], 1, store),
            "other": _tensor(B + (1 << 20), [16], [1], 4),
        }
        self.assertEqual(
            _names(resolve(B + 32, ts)),
            [("view", "exact", 0), ("base", "inside", 16), ("bytes", "inside", 32)],
        )
        self.assertEqual(resolve(B + 32, ts)[0]["storage"], store)
        # not on an element boundary of the bf16 tensors
        self.assertEqual(_names(resolve(B + 33, ts)), [("bytes", "inside", 33)])
        self.assertEqual(_names(resolve(B, ts)), [("base", "exact", 0), ("bytes", "exact", 0),
                                                  ("view", "inside", -16)])  # fmt: skip
        self.assertEqual(resolve(B - 16, ts), [])

    def test_zero_length_tensors(self):
        ts = {
            "empty": _tensor(
                0, [0], [1], 4
            ),  # CPU torch.empty(0): data_ptr 0, no storage
            "n0": _tensor(0, [0, 64], [64, 1], 2),
            "x": _tensor(B, [8], [1], 4),
            "tail": _tensor(
                B + 32, [0], [1], 4, [B, B + 32]
            ),  # x[8:]: data pointer at the end
        }
        self.assertEqual(resolve(0, ts), [])
        self.assertEqual(_names(resolve(B + 32, ts)), [("tail", "exact", 0)])
        self.assertEqual(
            _names(resolve(B + 4, ts)), [("tail", "inside", -7), ("x", "inside", 1)]
        )

    def test_shifted_base(self):
        """A ragged TMA tensor shifted back by 2^30 rows (quack create_ragged_tensor_for_tma,
        ptr_shift): (2^30, H, D) plus a (2^31 - 1)-row dim with the row stride, starting far
        below Q's storage; Q is the candidate at a negative offset, and so is V with Q's
        strides. 1-D tensors and tensors with other strides inside the span are not."""
        h, dh, rows = 4, 64, 1000
        q = _tensor(B, [rows, h, dh], [h * dh, dh, 1], 2)
        ts = {
            "Q": q,
            "V": _tensor(B + (1 << 24), [rows, h, dh], [h * dh, dh, 1], 2),
            "K": _tensor(B + (1 << 25), [rows, dh], [dh, 1], 2),
            "cu_seqlens": _tensor(B + (1 << 26), [9], [1], 4),
        }
        row = h * dh * 2
        addr = B - (1 << 30) * row
        desc = {"address": addr, "elem_bits": 16, "dims": [dh, h, 1 << 30, (1 << 31) - 1],
                "strides_bytes": [dh * 2, row, row]}  # fmt: skip
        self.assertEqual(
            _names(resolve(addr, ts, desc)),
            [
                ("Q", "shifted", -(1 << 30) * h * dh),
                ("V", "shifted", -((1 << 30) * h * dh + (1 << 23))),
            ],
        )
        self.assertEqual(resolve(addr, ts), [])  # without the descriptor: no span
        # a span that ends before the storage
        short = {**desc, "dims": [dh, h, 1 << 20, 1]}
        self.assertEqual(resolve(addr, ts, short), [])
        # Q's own descriptor: exact; the shifted one also reaches V (a candidate after it)
        plain = {
            **desc,
            "address": B,
            "dims": [dh, h, rows],
            "strides_bytes": [dh * 2, row],
        }
        self.assertEqual(_names(resolve(B, ts, plain)), [("Q", "exact", 0)])
        self.assertEqual(
            _names(resolve(B, ts, {**desc, "address": B})),
            [("Q", "exact", 0), ("V", "shifted", -(1 << 23))],
        )


if __name__ == "__main__":
    unittest.main()
