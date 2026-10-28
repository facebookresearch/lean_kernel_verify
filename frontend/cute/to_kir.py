# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""to_kir: analyzed CuTe stage + recorded launches -> KIR JSON (references/kir.md).

For every kernel variant (distinct initial MLIR) the host compiled and launched:

  1. parse the stages (generic MLIR, mlir_text.py);
  2. bind the host function's arguments to KIR params (dynamic shape/stride leaves of
     tensor arguments, i32/i64 scalars) and translate the host function: grid, block,
     shared memory, TMA descriptors and every kernel operand as an expression over params;
  3. slice the kernel (slicer.py) and translate it role by role (interp.py), refining
     the warp partition whenever a branch separates warps of a role;
  4. resolve mbarrier addresses into barrier arrays, set `threads_per_issue`, mark
     address arithmetic (`addr`), build tensors, descriptors and the case inputs from the
     recorded launches;
  5. validate with frontend/kir.py and, only if valid, write
     <out>/kir/<Kernel>__<digest>.json;
  6. P1 pass (P1-1 TMEM reuse ordered, ops_tmem.py): translate the variant again with
     the P1 slice and op table into <out>/kir/<Kernel>__<digest>.p1.json (the P0 events
     plus tmem, tmem_wait and named_bar events and the top-level `tmem`). A refusal there
     (or a failure: status "error") is recorded as the variant's "p1" entry in
     report.json; the P0 KIR is not affected (`--no-p1` skips the pass).

A construct without an exact translation stops that variant with the op name and its
source location; every outcome is in <out>/kir/report.json, a stopped variant with its
"kind" (`Unsupported.kind`: "gap" for a construct refused by design, "error" for what the
case, the DSL install or LKV must fix). A metadata tensor written by
the analyzed launch itself (or sharing storage with a tensor it writes), or holding data
an earlier (skipped) launch of the run may write, directly or through host computation,
is a hard error too: the KIR would read the recorded contents, not what the kernel
reads. So is a host read, before a launch of the variant, of data an earlier launch may
write (the host may branch or size the launch differently), and a dump made with a CuTe
DSL other than the one the op table was checked against (ops.DSL_VERSION).

Standalone (no CuTe DSL needed): python3 -I to_kir.py CASE.json [--out DIR]
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):  # python3 -I path/to/to_kir.py
    sys.dont_write_bytecode = True  # keep the source tree free of __pycache__
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "cute"  # noqa: A001

from . import (  # noqa: E402
    expr as E,
    layout as L,
    mlir_text,
    ops,
    ops_mem,
    ops_tmem,
    slicer,
)
from .interp import (  # noqa: E402
    CLUSTER_BAR,
    CLUSTER_PARITY,
    Data,
    int_width,
    Kernel,
    Lay,
    Opaque,
    Ptr,
    SplitWarps,
    Translator,
    Unsupported,
    View,
)

assert ops_mem  # registers the memory, TMA, mbarrier and host part of the op table

KIR_PY = Path(__file__).resolve().parents[1] / "kir.py"

TORCH_BITS = {
    "float64": 64, "int64": 64, "float32": 32, "int32": 32, "bfloat16": 16, "float16": 16,
    "int16": 16, "uint8": 8, "int8": 8, "bool": 8, "float8_e4m3fn": 8, "float8_e5m2": 8,
    "float8_e8m0fnu": 8, "float4_e2m1fn_x2": 8,
}  # fmt: skip

# Kernel-stage verifier failures whose op-table meaning does not use the violated
# constraint (stage.py; ops_mem.py bulk_copy_g2s does not use the alignment).
ACCEPTED_FAILURES = {
    "cute_nvgpu.arch.copy.SM90.bulk_copy_g2s": "must be CuTe pointer to global memory with byte alignment",
}


class Variant:
    """One compiled kernel variant and the inputs that launched it."""

    def __init__(self, case: dict, dump: dict, rec: dict, p1: bool = False):
        self.case, self.rec = case, rec
        self.p1 = p1  # the P1 pass: ops_tmem's slice and op table (TMEM events)
        self.spec = ops_tmem.spec_p1 if p1 else ops.spec
        self.dsl_version = dump["dsl"]["version"]
        self.cls = rec["callable"].rsplit(".", 1)[-1]
        self.notes = case.get("kernels", {}).get(self.cls, {})
        self.src_root = dump["source"]["src"].rstrip("/") + "/"
        self.stage = mlir_text.parse(Path(rec["stage"]).read_text())
        self.host_stage = mlir_text.parse(Path(rec["host_stage"]).read_text())
        self.params: dict = {}  # name -> width
        self.host_args: list = []  # (name, kind, info) per host-function argument
        self.k = Kernel(self.stage, self.src_root, max_iters=int(case.get("max_iters", 4096)),
                        work=self.notes.get("work", {}))  # fmt: skip

    # ----------------------------------------------------------------- host side
    def host(self, first_args: list) -> dict:
        func = next(
            o for o in self.host_stage.top.regions[0][0].ops if o.name == "func.func"
        )
        hk = Kernel(
            self.host_stage, self.src_root, tensors=self.k.tensors, descs=self.k.descs
        )
        tr = Translator(hk, self.spec, host=True, p1=self.p1)
        tr.cur_loc = hk.user_loc(func) or "host"
        args = self._bind_host_args(func, first_args)
        hk.slice = slicer.Slice(func, ops.effect_operands(tr, ops_mem.HOST_EFFECTS))
        tr.run_block(func.regions[0][0], args)
        if len(tr.launches) != 1:
            raise Unsupported(
                "cuda.launch_ex",
                "host",
                f"{len(tr.launches)} kernel launches in one host function",
            )
        return tr.launches[0]

    def _bind_host_args(self, func, first_args: list) -> list:
        """Host-function arguments in order: the non-constexpr compile arguments (tuples
        flattened, None dropped). The launch passes the same values except the stream
        (the TVM-FFI environment stream is implicit)."""
        block = func.regions[0][0]
        names = [
            leaf
            for p in self.rec["params"]
            if not p["constexpr"]
            for leaf in p["leaves"]
        ]
        types = [self.host_stage.resolve_type(t) for _n, t in block.args]
        launch = iter(_flatten(first_args))
        nlaunch = len(_flatten(first_args))
        nlive = sum(not t.startswith("!cuda.stream") for t in types)
        if len(names) != len(types) or nlaunch != nlive:
            raise Unsupported("func.func", "host", f"{len(types)} host arguments, {len(names)} compile arguments, "
                                                   f"{nlaunch} launch arguments")  # fmt: skip
        values: list[Opaque | View | Data | tuple] = []
        for name, ty in zip(names, types):
            if ty.startswith("!cuda.stream"):
                values.append(Opaque("stream"))
                continue
            arg = next(launch)
            w = int_width(ty)
            if ty.startswith("!cute.memref<"):
                values.append(self._tensor_arg(name, ty, arg))
            elif w in (32, 64):
                self.params[name] = w
                self.host_args.append((name, "scalar", None))
                values.append(E.var(name, w))
            else:
                self.host_args.append((name, "data", None))
                values.append(Data(f"host argument {name} ({ty})"))
        return values

    def _tensor_arg(self, name: str, ty: str, arg: dict):
        bits, space, ltxt = ops.memref_info(ty)
        if space != "gmem" or arg["kind"] != "tensor":
            raise Unsupported(
                "func.func",
                "host",
                f"host argument {name}: {space} memref for a {arg['kind']}",
            )
        if ltxt is None:
            raise Unsupported(
                "func.func",
                "host",
                f"host argument {name}: memref type {ty} has no layout",
            )
        shape, stride = L.parse_layout(ltxt)
        if any(isinstance(x, list) for x in (shape if isinstance(shape, list) else [])):
            raise Unsupported(
                "func.func", "host", f"hierarchical host layout for {name}"
            )
        tbits = TORCH_BITS.get(arg["dtype"])
        if tbits is None or tbits % bits:
            raise Unsupported(
                "func.func",
                "host",
                f"{name}: {arg['dtype']} storage for {bits}-bit elements",
            )
        leaf = (
            lambda pname, x: x if not isinstance(x, L.Dyn) else self._param(pname, x.w)
        )  # noqa: E731
        sh = [leaf(f"{name}_shape{i}", x) for i, x in enumerate(L.leaves(shape))]
        st = [leaf(f"{name}_stride{i}", x) for i, x in enumerate(L.leaves(stride))]
        self.k.tensors[name] = {"bits": bits, "elem_bytes": tbits // 8, "pack": tbits // bits, "shape": sh,
                                "strides": st, "uses": set(), "metadata": False}  # fmt: skip
        self.host_args.append(
            (name, "tensor", {"shape": L.leaves(shape), "stride": L.leaves(stride)})
        )
        lay = Lay(sh, st) if isinstance(shape, list) else Lay(sh[0], st[0])
        return View(Ptr("gmem", name, E.const(0, 64), bits), lay)

    def _param(self, name: str, w: int):
        self.params[name] = w
        return E.var(name, w)

    # --------------------------------------------------------------- kernel side
    def roles(self, launch: dict):
        kop = next(
            (o for o in mlir_text.walk(self.stage.top)
             if o.name == "cuda.kernel" and f'"{launch["kernel"]}"' in o.props.get("sym_name", "")), None)  # fmt: skip
        if kop is None:
            raise Unsupported(
                "cuda.launch_ex", "host", f"kernel {launch['kernel']} not in the stage"
            )
        cfg = dict(launch["cfg"].info)
        kargs = kop.regions[0][0].args
        if len(kargs) != len(launch["args"]):
            raise Unsupported(
                "cuda.launch_ex",
                "host",
                f"{len(launch['args'])} launch operands for {len(kargs)} kernel arguments",
            )
        if not all(E.is_const(b) for b in cfg["block"]):
            raise Unsupported(
                "cuda.launch_cfg.create", "host", "the block size is not a constant"
            )
        block = tuple(b[2] for b in cfg["block"])
        if block[1] != 1 or block[2] != 1 or block[0] % 32:
            raise Unsupported(
                "cuda.launch_cfg.create",
                "host",
                f"block {block} (only (32k, 1, 1) is supported)",
            )
        cluster = tuple(
            x[2] if E.is_const(x) else -1
            for x in cfg.get("cluster", (E.const(1, 32),) * 3)
        )
        if any(c < 1 for c in cluster):
            raise Unsupported(
                "cuda.launch_cfg.cluster_dim",
                "host",
                f"cluster dims {cluster} are not positive constants",
            )
        self.k.block, self.k.cluster, self.k.smem = block, cluster, cfg["smem"]
        effects = ops_tmem.effects_p1() if self.p1 else None
        self.k.slice = slicer.Slice(
            kop, ops.effect_operands(Translator(self.k, self.spec, p1=self.p1), effects)
        )
        todo, roles = [(0, block[0] // 32)], []
        while todo:
            w0, nw = todo.pop(0)
            tr = Translator(self.k, self.spec, w0, nw, p1=self.p1)
            tr.cur_loc = self.k.user_loc(kop) or "unknown"
            try:
                tr.run_block(kop.regions[0][0], launch["args"])
            except SplitWarps as s:
                todo = s.groups + todo
                continue
            if self.k.work:  # work fields are role-level variables, -1 until set
                tr.stmts[:0] = [
                    {
                        "op": "let",
                        "var": f"work_{f}",
                        "expr": E.to_json(E.const(-1, 64)),
                    }
                    for f in self.k.work
                ]
            if (
                tr.uses_cluster_wait
            ):  # the role's first cluster barrier wait is for phase 0
                tr.stmts[:0] = [
                    {
                        "op": "let",
                        "var": CLUSTER_PARITY,
                        "expr": E.to_json(E.const(0, 32)),
                    }
                ]
            roles.append(tr)
        return roles

    def barriers(self, roles: list) -> list:
        """Barrier arrays from the constant init addresses; resolves every event's
        (bar, idx) and sets each role's threads_per_issue."""
        inits, arrays = _barrier_runs(roles)
        arrays = _split_arrays(
            arrays, [u for tr in roles for u in tr.barriers if not u.fixed]
        )
        for tr in roles:
            for u in tr.barriers:
                if not u.fixed:  # a fixed slot's bar and idx are the translator's
                    _resolve_barrier(u, arrays)
            arrive = [u for u in tr.barriers if u.event["ev"] == "arrive"]
            tr.tpi = math.gcd(*[u.issuers or 1 for u in arrive]) if arrive else 1
            for u in arrive:
                if (u.issuers or 1) != tr.tpi:
                    u.event["n"] = (u.issuers or 1) // tr.tpi
        for start, _num, count in arrays:
            if _json_vars(count) - set(self.params):
                raise Unsupported(
                    "nvvm.mbarrier.init.shared",
                    inits[start][1],
                    "mbarrier count is not a host value",
                )
        out = [
            {"name": f"mbar_{s}", "count": c, "num": E.to_json(E.const(n, 32))}
            for s, n, c in arrays
        ]
        if any(u.fixed == CLUSTER_BAR for tr in roles for u in tr.barriers):
            # one arrival per thread of the cluster (interp.Translator.cluster_barrier_event)
            threads = self.k.block[0] * math.prod(self.k.cluster)
            out.append(
                {
                    "name": CLUSTER_BAR,
                    "count": E.to_json(E.const(threads, 32)),
                    "num": E.to_json(E.const(1, 32)),
                }
            )
        return out

    # ---------------------------------------------------------------- assembly
    def mark_segment_offsets(self) -> None:
        """Segment offsets are metadata even if the kernel never reads them."""
        notes = self.notes.get("tensors", {})
        unknown = set(notes) - set(self.k.tensors)
        if unknown:
            raise Unsupported(
                "case",
                "kernels",
                f"{self.cls}: annotated tensors {sorted(unknown)} are not kernel tensors",
                kind="error",
            )
        for note in notes.values():
            seg = note.get("segments")
            if seg and seg["offsets"] in self.k.tensors:
                self.k.tensors[seg["offsets"]]["metadata"] = True

    def check_metadata_writes(self, launches: list) -> None:
        """No metadata tensor is written by this kernel, or shares storage bytes with a
        tensor it writes in one of its launches (the KIR reads the recorded contents)."""
        written = {
            n for n, t in self.k.tensors.items() if t["uses"] & {"gstore", "gatomic"}
        }
        meta = {n for n, t in self.k.tensors.items() if t["metadata"]}
        if meta & written:
            raise Unsupported(
                "case",
                "kernels",
                f"{self.cls} writes its metadata tensors {sorted(meta & written)}",
            )
        for in_name, args in launches:
            spans = {
                name: _byte_span(a)
                for (name, kind, _i), a in zip(self.host_args, _flatten(args))
                if kind == "tensor"
            }
            for m in sorted(meta):
                for w in sorted(written):
                    if _overlap(spans[m], spans[w]):
                        raise Unsupported("case", "kernels", f"{self.cls}, input {in_name}: metadata tensor {m} shares "
                                                             f"storage with the written tensor {w}")  # fmt: skip

    def tensors(self) -> list:
        out = []
        notes = self.notes.get("tensors", {})
        for name, t in self.k.tensors.items():
            if not (t["uses"] or t["metadata"]):
                continue
            if t["metadata"]:  # flat buffer: `load` indices are element offsets
                shape, strides = [self._extent(t)], [E.const(1, 64)]
            else:
                shape, strides = self._storage(t)
            written = bool(t["uses"] & {"gstore", "gatomic"})
            note = notes.get(name, {})
            out.append({
                "name": name, "kind": "global", "elem_bytes": t["elem_bytes"],
                "shape": [E.to_json(x) for x in shape], "strides": [E.to_json(x) for x in strides],
                "segments": note.get("segments"), "role": note.get("role", "output" if written else "input"),
                "metadata": t["metadata"],
            })  # fmt: skip
        for name, d in self.k.descs.items():
            if not d["uses"]:
                continue
            if "bulk" in d:
                t = self.k.tensors[d["bulk"]]
                shape, strides, eb, segments = (
                    [self._extent(t)],
                    [E.const(1, 64)],
                    t["elem_bytes"],
                    None,
                )
            else:
                shape, strides, eb = d["shape"], d["strides"], d["elem_bytes"]
                seg = notes.get(d["base"], {}).get("segments")
                dims = [
                    i
                    for i, td in enumerate(d["tensor_dims"])
                    if seg and td == seg["dim"]
                ]
                segments = {**seg, "dim": dims[0]} if dims else None
            role = notes.get(d.get("base", d.get("bulk")), {}).get("role", "input")
            out.append({
                "name": name, "kind": "desc", "elem_bytes": eb, "shape": [E.to_json(x) for x in shape],
                "strides": [E.to_json(x) for x in strides], "box": d["box"], "segments": segments,
                "role": role, "metadata": False,
            })  # fmt: skip
        return out

    def _storage(self, t):
        """Shape/strides in storage elements (`pack` DSL elements per storage element)."""
        k = t["pack"]
        if k == 1:
            return list(t["shape"]), list(t["strides"])
        sh, st = [], []
        for x, s in zip(t["shape"], t["strides"]):
            unit = E.is_const(s, 1)
            sh.append(E.bin("sdiv", x, E.const(k, E.width(x))) if unit else x)
            st.append(s if unit else E.bin("sdiv", s, E.const(k, E.width(s))))
        return sh, st

    def _extent(self, t):
        """Number of storage elements spanned by the tensor (1 + sum (n-1) * stride)."""
        ext = E.const(1, 64)
        for x, s in zip(*self._storage(t)):
            ext = E.bin(
                "add",
                ext,
                E.bin(
                    "mul",
                    E.bin("sub", E.resize(x, 64), E.const(1, 64)),
                    E.resize(s, 64),
                ),
            )
        return ext

    def inputs(self, launches: list) -> list:
        out = []
        for in_name, args in launches:
            params, meta = {}, {}
            for (name, kind, info), a in zip(self.host_args, _flatten(args)):
                if kind == "scalar":
                    params[name] = int(a["value"])
                elif kind == "tensor":
                    self._tensor_params(name, info, a, params)
                    if self.k.tensors[name]["metadata"]:
                        if "data" not in a:
                            raise Unsupported(
                                "cuda.launch_ex",
                                "host",
                                f"metadata tensor {name} has no recorded contents",
                            )
                        meta[name] = a["data"]
            inp = {
                "name": in_name,
                "params": params,
                "metadata": meta,
                "flags": self.rec["flags"],
            }
            if self.notes.get("intended_writes"):
                inp["intended_writes"] = self.notes["intended_writes"]
            out.append(inp)
        return out

    def _tensor_params(self, name, info, a, params) -> None:
        k = self.k.tensors[name]["pack"]
        shape, stride = list(a["shape"]), list(a["stride"])
        if len(shape) != len(info["shape"]):
            raise Unsupported(
                "cuda.launch_ex",
                "host",
                f"{name}: rank {len(shape)} vs host rank {len(info['shape'])}",
            )
        if k != 1:  # in the kernel's (unpacked) element units
            shape = [s * k if st == 1 else s for s, st in zip(shape, stride)]
            stride = [st if st == 1 else st * k for st in stride]
        for kind, leaves, vals in (
            ("shape", info["shape"], shape),
            ("stride", info["stride"], stride),
        ):
            for i, (leaf, v) in enumerate(zip(leaves, vals)):
                if isinstance(leaf, L.Dyn):
                    params[f"{name}_{kind}{i}"] = v
                elif leaf[2] != v:
                    raise Unsupported(
                        "cuda.launch_ex",
                        "host",
                        f"{name}: static {kind} {leaf[2]} but the launch passes {v}",
                    )

    def kir(self, launch: dict, roles: list, barriers: list, inputs: list) -> dict:
        elision = self.notes.get("elision_flags", [])
        unknown = sorted(set(elision) - set(self.rec["flags"]))
        if unknown:
            raise Unsupported("case", "kernels", f"{self.cls}: elision_flags {unknown} are not flags of the variant "
                                                 f"({sorted(self.rec['flags'])})", kind="error")  # fmt: skip
        obj = {
            "kir_version": 1,
            "kernel": self.cls,
            "source": self.rec["callable"].rsplit(".", 1)[0].replace(".", "/") + ".py",
            "ir": "cute-mlir",
            "target": "sm_100",  # dump.py compiles with CUTE_DSL_ARCH=sm_100a
            "compiler": {"name": "nvidia-cutlass-dsl", "version": self.dsl_version},
            "grid": [E.to_json(g) for g in launch["cfg"].get("grid")],
            "params": [
                {"name": n, "type": f"i{w}"} for n, w in sorted(self.params.items())
            ],
            **(
                {"cluster": list(self.k.cluster)} if self.k.cluster != (1, 1, 1) else {}
            ),
            "tensors": self.tensors(),
            "barriers": barriers,
            "roles": [
                {
                    "name": f"warps{tr.w0}_{tr.w0 + tr.nw - 1}",
                    "warps": tr.nw,
                    "threads_per_issue": tr.tpi,
                    "body": tr.stmts,
                }
                for tr in roles
            ],
            "case": {
                "name": self.case["name"],
                "inputs": inputs,
                **({"elision_flags": elision} if elision else {}),
            },
        }
        mark_addr(obj)
        return obj


def _byte_span(a: dict):
    """(storage block, first byte, end byte) a tensor argument can address; None if empty."""
    if any(n == 0 for n in a["shape"]):
        return None
    eb = TORCH_BITS[a["dtype"]] // 8
    lo = sum(min(0, (n - 1) * st) for n, st in zip(a["shape"], a["stride"]))
    hi = sum(max(0, (n - 1) * st) for n, st in zip(a["shape"], a["stride"]))
    return a["storage"][0], a["byte_offset"] + lo * eb, a["byte_offset"] + (hi + 1) * eb


def _overlap(x, y) -> bool:
    return (
        x is not None and y is not None and x[0] == y[0] and x[1] < y[2] and y[1] < x[2]
    )


def _flatten(args: list) -> list:
    """Launch arguments in host-function order: tuples flattened, None dropped, and an
    explicit `cuda.CUstream` dropped (the host function's stream argument takes no launch
    value: a kernel uses the TVM-FFI environment stream or passes its stream explicitly)."""
    out = []
    for a in args:
        if a["kind"] == "tuple":
            out += _flatten(a["items"])
        elif a["kind"] == "opaque" and a.get("type") == "CUstream":
            continue
        elif a["kind"] != "none":
            out.append(a)
    return out


def _json_vars(e) -> set:
    """Variable names in a KIR Expr."""
    if e.get("e") == "var":
        return {e["name"]}
    out = set()
    for k in ("a", "b", "c"):
        if isinstance(e.get(k), dict):
            out |= _json_vars(e[k])
    for x in e.get("index", []):
        out |= _json_vars(x)
    return out


def _linear(e):
    """e = c + sum k_i * t_i (exact in wrapping i64 arithmetic) -> (c, [(k_i, t_i)])."""
    if E.is_const(e):
        return e[2], []
    if e[0] == "bin" and e[2] in ("add", "sub"):
        ca, ta = _linear(e[3])
        cb, tb = _linear(e[4])
        if e[2] == "sub":
            cb, tb = -cb, [(-k, t) for k, t in tb]
        return ca + cb, ta + tb
    if e[0] == "bin" and e[2] in ("mul", "shl") and E.is_const(e[4]):
        f = e[4][2] if e[2] == "mul" else 1 << e[4][2]
        c, ts = _linear(e[3])
        return c * f, [(k * f, t) for k, t in ts]
    return 0, [(1, e)]


def _barrier_runs(roles: list):
    """({init address: (count, loc, site)}, [[start byte, num, count]]): the initialized
    barriers and their runs, consecutive 8-byte barriers initialized at one site (source
    frames and enclosing op, interp.BarrierUse) with one count."""
    inits = {}
    for tr in roles:
        for u in tr.barriers:
            for a in () if u.fixed else u.inits:
                count = u.event["count"]
                if inits.setdefault(a, (count, u.loc, u.site))[0] != count:
                    raise Unsupported(
                        "nvvm.mbarrier.init.shared",
                        u.loc,
                        f"smem+{a} initialized with two counts",
                    )
    arrays = []
    for a in sorted(inits):
        count, _loc, site = inits[a]
        last = arrays[-1] if arrays else None
        if (
            last
            and last[0] + 8 * last[1] == a
            and last[2] == count
            and inits[last[0]][2] == site
        ):
            last[1] += 1
        else:
            arrays.append([a, 1, count])
    return inits, arrays


def _resolve_barrier(u, arrays: list) -> None:
    """Set the event's bar and idx: address `C + sum k_i t_i` in the array holding the
    constant part of the address with let-bound index variables expanded (an index `v`
    bound to `stage + 29` names the array of slot 29), idx `(C - start)/8 + sum
    (k_i/8) t_i`."""
    c, terms = _linear(u.addr)
    cb = _constant_part(u.base) if u.base is not None else c
    arr = next((x for x in arrays if x[0] <= cb < x[0] + 8 * x[1]), None)
    if arr is None:
        raise Unsupported(
            u.event["ev"], u.loc, f"smem+{cb} is not an initialized mbarrier"
        )
    if (c - arr[0]) % 8 or any(k % 8 for k, _t in terms):
        raise Unsupported(
            u.event["ev"],
            u.loc,
            "mbarrier address is not a multiple of 8 bytes from its array",
        )
    idx = E.const((c - arr[0]) // 8, 64)
    for k, t in terms:
        idx = E.bin("add", idx, E.bin("mul", t, E.const(k // 8, 64)))
    u.event["bar"] = f"mbar_{arr[0]}"
    u.event["idx"] = E.to_json(E.trunc(idx, 32))


def _split_arrays(arrays: list, uses: list) -> list:
    """Split each run of barriers (one call site, one count) into the barrier arrays its
    uses index: a use with a dynamic index keeps together the slots its address may take
    (interval of the address with let-bound index variables expanded); a run with a use
    whose range is unknown, or leaves the run, stays whole. Slots only constant addresses
    name become arrays of one."""
    cuts = [set(range(1, num)) for _start, num, _count in arrays]
    for u in uses:
        if not _linear(u.addr)[1]:
            continue  # a constant address
        base = u.base if u.base is not None else u.addr
        cb = _constant_part(base)
        i = next(
            (j for j, (st, n, _c) in enumerate(arrays) if st <= cb < st + 8 * n), None
        )
        if i is None:
            continue  # reported when the address is resolved
        start, num, _count = arrays[i]
        rng = E.interval(base)
        if rng is None or rng[0] < start or rng[1] >= start + 8 * num:
            cuts[i] = set()
        else:
            cuts[i] -= set(range((rng[0] - start) // 8 + 1, (rng[1] - start) // 8 + 1))
    out = []
    for (start, num, count), cs in zip(arrays, cuts):
        bounds = [0, *sorted(cs), num]
        out += [[start + 8 * a, b - a, count] for a, b in zip(bounds, bounds[1:])]
    return out


def _constant_part(e) -> int:
    """The constant c of e = c + sum k_i * t_i, looking through integer casts (which is
    exact without wrapping): it picks the barrier array of an address only, the index
    itself is computed from _linear of the unexpanded address."""
    if e[0] in ("ext", "trunc"):
        return _constant_part(e[3] if e[0] == "ext" else e[2])
    if E.is_const(e):
        return e[2]
    if e[0] == "bin" and e[2] in ("add", "sub"):
        a, b = _constant_part(e[3]), _constant_part(e[4])
        return a + b if e[2] == "add" else a - b
    if e[0] == "bin" and e[2] in ("mul", "shl") and E.is_const(e[4]):
        f = e[4][2] if e[2] == "mul" else 1 << e[4][2]
        return _constant_part(e[3]) * f
    return 0


# ------------------------------------------------------------------ addr marking

_ADDR_OPS = ("add", "sub", "mul", "shl")  # wraps are P0-1 violations on address paths
_MEM_EVENTS = ("gload", "gstore", "gatomic")


def _mark(node) -> None:
    kind = node.get("e") or node.get("t")
    if kind == "trunc" or kind == "bin" and node["o"] in _ADDR_OPS:
        node["addr"] = True


def _statements(stmts):
    for s in stmts:
        yield s
        for k in ("then", "else", "body"):
            yield from _statements(s.get(k, []))


def _children(node) -> list:
    """Sub-nodes of a KIR Expr or TExpr (operands and load indices)."""
    return [
        node[k] for k in ("a", "b", "c") if isinstance(node.get(k), dict)
    ] + node.get("index", [])


def _nodes(node):
    yield node
    for c in _children(node):
        yield from _nodes(c)


def mark_addr(kir: dict) -> None:
    """Set "addr" on every add/sub/mul/shl and trunc whose value reaches a global-memory
    offset, a TMA coordinate or a metadata load index, through any chain of `let`s (Lean
    reports their wraps and lossy truncations as P0-1 violations)."""
    stmts = [s for r in kir["roles"] for s in _statements(r["body"])]
    lets: dict = {}
    seeds: list = []
    for s in stmts:
        if s["op"] == "let":
            lets.setdefault(s["var"], []).append(s["expr"])
        if s["op"] == "event" and s["ev"] in _MEM_EVENTS:
            seeds += [
                n["a"]
                for n in _nodes(s["offsets"])
                if "t" in n and "e" in n.get("a", {})
            ]
            for n in _nodes(s["offsets"]):  # the TExpr arithmetic itself
                if "t" in n:
                    _mark(n)
        if s["op"] == "event" and s["ev"] == "tma_load":
            seeds += s["coords"]
        roots = [v for k, v in s.items() if isinstance(v, dict) and "e" in v] + s.get(
            "coords", []
        )
        seeds += [
            i
            for r in roots
            for n in _nodes(r)
            if n.get("e") == "load"
            for i in n["index"]
        ]
    done: set = set()
    while seeds:
        e = seeds.pop()
        if id(e) in done:
            continue
        done.add(id(e))
        _mark(e)
        if e.get("e") == "var":
            seeds += lets.get(e["name"], [])
        seeds += _children(e)


# ------------------------------------------------------------------ driver


def validate(obj) -> list:
    spec = importlib.util.spec_from_file_location("lkv_kir_validator", KIR_PY)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the KIR validator {KIR_PY}")
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True  # KIR_PY is in the source tree
    spec.loader.exec_module(mod)
    return mod.validate(obj)


def write_kir(path: Path, obj: dict, entry: dict) -> bool:
    """Write obj to path only if frontend/kir.py accepts it; the outcome goes to the
    variant's report entry."""
    errors = validate(obj)
    if errors:
        entry["error"] = "invalid KIR (frontend/kir.py): " + "; ".join(errors[:20])
        entry["kind"] = "error"
        return False
    path.write_text(json.dumps(obj, separators=(",", ":")))
    entry.update({"kir": str(path), "bytes": path.stat().st_size,
                  "roles": [[r["name"], r["warps"], r["threads_per_issue"]] for r in obj["roles"]]})  # fmt: skip
    return True


def translate_p1(case: dict, dump: dict, rec: dict, launches: list) -> dict | None:
    """The P1 KIR of a variant (to_kir step 6): a fresh translation with the P1 slice."""
    v = Variant(case, dump, rec, p1=True)
    launch = v.host(launches[0][1])
    roles = v.roles(launch)
    barriers = v.barriers(roles)
    v.mark_segment_offsets()
    v.check_metadata_writes(launches)
    obj = v.kir(launch, roles, barriers, v.inputs(launches))
    if not any(tr.tmem_allocs or tr.tmem_bases for tr in roles):
        return None  # no TMEM in this kernel: nothing for P1-1 to check
    obj["tmem"] = ops_tmem.tmem_top(roles)
    return obj


def translate(case: dict, dump: dict, rec: dict, launches: list):
    dsl = dump["dsl"]
    native = dsl.get("installed", {}).get("nvidia-cutlass-dsl-libs-base")
    if dsl["version"] != ops.DSL_VERSION or native != ops.DSL_VERSION:
        raise Unsupported("dump", "dsl", f"the dump was made with CuTe DSL {dsl['version']} (native libraries "
                                         f"nvidia-cutlass-dsl-libs-base {native}); the op table was checked against "
                                         f"{ops.DSL_VERSION} (references/ir_ops_cute.md)", kind="error")  # fmt: skip
    for f in rec.get("verifier_failures", []):
        if ACCEPTED_FAILURES.get(f["op"], "\0") not in f["error"]:
            raise Unsupported(
                f["op"], f["loc"], f"does not verify: {f['error'][:300]}", kind="error"
            )
    v = Variant(case, dump, rec)
    launch = v.host(launches[0][1])
    roles = v.roles(launch)
    barriers = v.barriers(roles)
    v.mark_segment_offsets()
    v.check_metadata_writes(launches)
    return v, v.kir(launch, roles, barriers, v.inputs(launches))


def _writes(variant, launch: dict) -> list:
    """Storage blocks a launch may write: its written tensors, or, when the kernel was not
    translated, every tensor argument and every storage its integer arguments point into
    (address hints, ttgir.host_reads)."""
    flat = _flatten(launch["args"])
    if variant is None:
        return [a["storage"][0] for a in flat if a["kind"] == "tensor"] + launch[
            "hinted"
        ]
    written = {
        n for n, t in variant.k.tensors.items() if t["uses"] & {"gstore", "gatomic"}
    }
    return [
        a["storage"][0]
        for (name, kind, _i), a in zip(variant.host_args, flat)
        if kind == "tensor" and name in written
    ]


def _load_dump(out: Path) -> dict:
    """<out>/dump.json; one an older dump wrote (per-input launch indices, no address
    hints) is refused."""
    dump = json.loads((out / "dump.json").read_text())
    if any("hinted" not in la for inp in dump["inputs"] for la in inp["launches"]):
        raise SystemExit(
            f"{out / 'dump.json'} was written by an older dump (per-input launch indices, no "
            "address hints): dump the case again"
        )
    return dump


def run(case_path: Path, out: Path, p1: bool = True) -> list:
    case = json.loads(Path(case_path).read_text())
    dump = _load_dump(out)
    kir_dir = out / "kir"
    kir_dir.mkdir(parents=True, exist_ok=True)
    for old in kir_dir.glob("*.json"):
        old.unlink()
    by_digest: dict = {}
    for c in dump["compiles"]:
        by_digest.setdefault(c["digest"], []).append(c)
    report, results = [], {}
    for digest, recs in by_digest.items():
        rec = recs[0]
        cls = rec["callable"].rsplit(".", 1)[-1]
        indices = {c["index"] for c in recs}
        launches = []
        for inp in dump["inputs"]:
            mine = [la for la in inp["launches"] if la["compile"] in indices]
            for j, la in enumerate(mine):
                launches.append(
                    (
                        inp["name"] if len(mine) == 1 else f"{inp['name']}#{j}",
                        la["args"],
                    )
                )
        entry = {
            "kernel": cls,
            "digest": digest,
            "inputs": [n for n, _a in launches],
            "dsl": dump["dsl"]["version"],
        }
        report.append(entry)
        # "kind" of a variant that gets no KIR: "gap", a construct the frontend refuses by
        # design (not an error), or "error", a failure the case, the DSL or LKV must fix.
        if "stage" not in rec:
            entry["error"] = f"no analyzed stage: {rec.get('stage_error', '?')}"
            entry["kind"] = "error"
            continue
        if not launches:
            entry["error"] = "compiled but never launched"
            entry["kind"] = "gap"
            continue
        try:
            variant, obj = translate(case, dump, rec, launches)
        except Unsupported as e:
            entry["error"] = str(e)
            entry["kind"] = e.kind
            continue
        results[digest] = (variant, obj, entry)
        if p1:
            try:
                obj1 = translate_p1(case, dump, rec, launches)
                entry["p1"] = (
                    {
                        "status": "not_applicable",
                        "error": "the kernel allocates and accesses no tensor memory",
                    }
                    if obj1 is None
                    else {"status": "translated", "obj": obj1}
                )
            except Unsupported as e:
                entry["p1"] = {
                    "status": "error" if e.kind == "error" else "refused",
                    "error": str(e),
                }
            except Exception as e:  # noqa: BLE001 - the P1 pass must never stop the P0 KIR
                entry["p1"] = {
                    "status": "error",
                    "error": f"P1 pass internal error: {type(e).__name__}: {e}",
                }
    for inp in dump["inputs"]:
        if "host_error" in inp:
            report.append({"kernel": "(host)", "digest": "-", "inputs": [inp["name"]], "dsl": dump["dsl"]["version"],
                           "error": f"the host rejected input {inp['name']}: {inp['host_error']}"})  # fmt: skip
    _check_provenance(dump, by_digest, results)
    written = []
    for digest, (_variant, obj, entry) in results.items():
        path = kir_dir / f"{entry['kernel']}__{digest}.json"
        p1e = entry.pop("p1", None)
        if "error" not in entry and write_kir(path, obj, entry):
            written.append(str(path))
        if p1e is not None and "error" not in entry:
            entry["p1"] = p1e
            if p1e["status"] == "translated":
                p1path = kir_dir / f"{entry['kernel']}__{digest}.p1.json"
                obj1 = p1e.pop("obj")
                errors = validate(obj1)
                if errors:
                    entry["p1"] = {
                        "status": "error",
                        "error": "invalid P1 KIR (frontend/kir.py): "
                        + "; ".join(errors[:20]),
                    }
                else:
                    p1path.write_text(json.dumps(obj1, separators=(",", ":")))
                    entry["p1"] = {
                        "status": "checked",
                        "kir": str(p1path),
                        "bytes": p1path.stat().st_size,
                        "roles": [
                            [r["name"], r["warps"], r["threads_per_issue"]]
                            for r in obj1["roles"]
                        ],
                    }
                    written.append(str(p1path))
    (kir_dir / "report.json").write_text(json.dumps(report, indent=1))
    return written + [str(kir_dir / "report.json")]


def _check_provenance(dump: dict, by_digest: dict, results: dict) -> None:
    """Every launch is skipped, so data an earlier launch of the run may write keeps its
    CPU contents (the inputs run in one process, so a launch of one input may write what a
    later input reads, e.g. through a module-level cache). A translated variant must not
    read such data as metadata, nor may the host read it before one of the variant's
    launches; data reaches a launch argument or a host read directly or through host
    computation (its "sources", ttgir.host_reads). Launch indices are run-wide; a host
    read in an earlier input counts too (it may have filled a cache)."""
    digest_of = {c["index"]: c["digest"] for c in dump["compiles"]}
    writes = []  # per launch of the run: storage blocks it may write -> its kernel
    for inp in dump["inputs"]:
        for la in inp["launches"]:
            pres = results.get(digest_of[la["compile"]])
            kernel = by_digest[digest_of[la["compile"]]][0]["callable"]
            writes.append(
                dict.fromkeys(
                    _writes(pres[0] if pres and "error" not in pres[2] else None, la),
                    kernel,
                )
            )
    reads, i = (
        [],
        0,
    )  # the host reads of the run so far; the run-wide index of the launch
    for inp in dump["inputs"]:
        reads += inp["host_reads"]
        for la in inp["launches"]:
            res = results.get(digest_of[la["compile"]])
            if res is not None and "error" not in res[2]:
                variant, entry = res[0], res[2]
                for (name, kind, _i), a in zip(variant.host_args, _flatten(la["args"])):
                    if (
                        kind == "tensor"
                        and variant.k.tensors[name]["metadata"]
                        and (w := _writer(a["sources"], writes))
                    ):
                        entry["error"] = (
                            f"metadata tensor {name} of input {inp['name']} may hold data that {w} writes "
                            "(directly or through host computation), but the CPU host skipped that launch"
                        )
                        entry["kind"] = "gap"  # skipped launches: a known limit
                for r in reads:
                    if r["after"] <= i and (w := _writer(r["sources"], writes)):
                        entry["error"] = (
                            f"input {inp['name']}: before launch {i} the host reads data ({r['op']} at "
                            f"{r['loc']}) that {w} may write; the CPU host skipped that launch, so it may "
                            "branch or size the launch differently"
                        )
                        entry["kind"] = "gap"
            i += 1


def _writer(sources: list, writes: list):
    """The first earlier launch that may write data with these sources: (storage, t) is
    stale if one of the first t launches writes the storage."""
    return next(
        (
            f"launch {k} ({writes[k][blk]})"
            for blk, t in sources
            for k in range(t)
            if blk in writes[k]
        ),
        None,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--no-p1", action="store_true", help="skip the P1 pass (no .p1.json KIR)"
    )
    a = ap.parse_args()
    case = json.loads(a.case.read_text())
    out = a.out or Path("/tmp/lkv_out") / a.case.resolve().parent.name / case["name"]
    for p in run(a.case, out, p1=not a.no_p1):
        print(p)


if __name__ == "__main__":
    main()
