# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the PTX frontend inside the family assembly (LKV_FRONTEND=ptx), no Lean.

    python3 -I tests/ptx/test_integration.py      (or the buck target :test_integration)

The e2e probes (tests/e2e/cases/probe_{tlx,cute}.json) from their recorded dumps
(fixtures/probe_*.dump.json and .ptx: the frontends' output trimmed to what the assembly
reads, source paths as @SRC@): frontend/ttgir/to_kir.py translate and
frontend/cute/to_kir.py run produce KIR that frontend/kir.py accepts, with `ir` "ptx", the
legacy names (roles, barriers, tensors, params, inputs), every launch's params and grid,
and the families' locations. Also the PTX write set the CuTe stale-data checks use for a
refused variant (translate.writes; a call or a loaded pointer leaves it unknown), the
P1 pass of the CuTe assembly (to_kir step 6, translate_p1_ptx: checked, refused, skipped
by --no-p1) and the switch's values. CuTe assembly (frontend/cute/to_kir.py): reroll
(decision D8: LLVM's unrolled blocks back into a `for`, the same trace; runs it must
keep), the jit_params naming of pointers and descriptor bases, and the KIR desc tensors
of Facts.descs (a segmented base's segments on the walking dim; a bulk descriptor's
extent). TTGIR assembly (frontend/ttgir/to_kir.py build_ptx): host descriptor blocks
(equal in every launch), the KIR desc tensors of Facts.descs (annotations, role) and the
CLC tables.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.ptx"  # noqa: A001

from ...frontend import kir  # noqa: E402
from ...frontend.cute import to_kir as cute_to_kir  # noqa: E402
from ...frontend.ptx import parse, translate as T  # noqa: E402
from ...frontend.ttgir import to_kir as ttgir_to_kir  # noqa: E402
from .fixtures.kernels import evaluate, kernel, run  # noqa: E402

# The source tree (buck runs link trees: resolve() leads back to it, as tests/cute does).
SKILL = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
KERNELS = SKILL / "tests/e2e/kernels"
PTX_MODE = {"LKV_FRONTEND": "ptx"}


def events(stmts):
    for s in stmts:
        if s["op"] == "event":
            yield s
        for k in ("then", "else", "body"):
            yield from events(s.get(k, []))


def fixture(name: str, where: Path) -> Path:
    """fixtures/<name> written to `where` with the probe sources' directory for @SRC@."""
    path = where / name
    path.write_text((FIXTURES / name).read_text().replace("@SRC@", str(KERNELS)))
    return path


class TtgirProbe(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        fixture("probe_tlx.ptx", self.tmp)
        self.dump = fixture("probe_tlx.dump.json", self.tmp)
        self.case = json.loads((SKILL / "tests/e2e/cases/probe_tlx.json").read_text())

    def test_kir(self):
        with mock.patch.dict(os.environ, PTX_MODE):
            results = ttgir_to_kir.translate(self.dump, self.case)
        inputs = {}
        for key, k in results.items():
            self.assertIsInstance(k, dict, f"{key}: {k}")
            self.assertEqual(kir.validate(k), [])
            self.assertEqual(json.loads((self.tmp / f"{key}.kir.json").read_text()), k)
            self.assertEqual(
                (k["ir"], k["kernel"], k["target"]), ("ptx", "probe", "sm_100")
            )
            self.assertEqual([(r["name"], r["warps"], r["threads_per_issue"]) for r in k["roles"]],
                             [("default", 4, 1), ("partition0", 1, 1)])  # fmt: skip
            self.assertEqual(
                [(b["name"], b["count"]["v"], b["num"]["v"]) for b in k["barriers"]],
                [("bars", 1, 1)],  # the alloc_barriers assignment target
            )
            self.assertEqual(
                [(t["name"], t["role"]) for t in k["tensors"]], [("x_ptr", "output")]
            )
            self.assertEqual(
                {p["name"] for p in k["params"]},
                {"n", "x_ptr.shape.0", "x_ptr.stride.0"},
            )
            evs = [e for r in k["roles"] for e in events(r["body"])]
            self.assertEqual(
                [e["ev"] for e in evs],
                ["bar_init", "gload", "gstore", "arrive", "wait"],
            )
            self.assertTrue(
                all(e["loc"].startswith("fbcode/lkv_e2e_probe_tlx.py:") for e in evs)
            )
            for inp in k["case"]["inputs"]:
                inputs[inp["name"]] = (inp["grid"][0], inp["params"])
        self.assertEqual(
            {n: (g, p["n"], p["x_ptr.shape.0"]) for n, (g, p) in inputs.items()},
            {"aligned": (2, 256, 256), "empty": (0, 0, 0), "partial": (3, 300, 300),
             "partial_floor": (2, 300, 300), "below_block": (1, 100, 100),
             "below_block_floor": (0, 100, 100)},
        )  # fmt: skip


class CuteProbe(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        ptx = fixture("probe_cute.ptx", self.tmp)
        dump = json.loads(fixture("probe_cute.dump.json", self.tmp).read_text())
        for c in dump["compiles"]:
            c["ptx"]["file"] = str(ptx)
        (self.tmp / "dump.json").write_text(json.dumps(dump))

    def run_kir(self):
        with mock.patch.dict(os.environ, PTX_MODE):
            cute_to_kir.run(SKILL / "tests/e2e/cases/probe_cute.json", self.tmp)
        return json.loads((self.tmp / "kir/report.json").read_text())

    def test_kir(self):
        variant, host = self.run_kir()
        self.assertEqual(
            (variant["kernel"], variant["inputs"]), ("host", ["n128", "n100", "n0"])
        )
        self.assertEqual(variant["roles"], [["warps0_3", 4, 1]])
        self.assertEqual(variant["p1"]["status"], "not_applicable")
        self.assertEqual((host["kernel"], host["inputs"]), ("(host)", ["n200"]))
        k = json.loads(Path(variant["kir"]).read_text())
        self.assertEqual(kir.validate(k), [])
        self.assertEqual((k["ir"], k["kernel"], k["target"]), ("ptx", "host", "sm_100"))
        self.assertEqual(
            [(t["name"], t["role"], t["metadata"]) for t in k["tensors"]],
            [("m", "output", False)],
        )
        self.assertEqual(
            [(i["name"], i["grid"], i["params"]["m_shape0"], i["params"]["m_stride0"],
              i["intended_writes"]) for i in k["case"]["inputs"]],
            [("n128", [1, 1, 1], 128, 1, {"m": "all"}), ("n100", [1, 1, 1], 100, 1, {"m": "all"}),
             ("n0", [1, 1, 1], 0, 1, {"m": "all"})],
        )  # fmt: skip
        evs = list(events(k["roles"][0]["body"]))
        self.assertEqual([e["ev"] for e in evs], ["gload", "gstore"])
        # DSL 4.4.2: lines only, the library line 847 (a float add) leaves no trace
        self.assertEqual({e["loc"] for e in evs}, {"lkv_e2e_probe_cute.py:23"})

    def test_write_set(self):
        dump = json.loads((self.tmp / "dump.json").read_text())
        records = [dump["inputs"][0]["launches"][0]["launch_records"][0]]
        text = Path(dump["compiles"][0]["ptx"]["file"]).read_text()
        fn = parse.parse(text).function(records[0]["entry"])
        self.assertEqual(
            T.writes(text, fn.name, T.param_table(records, "cute", fn, ["m"])), ["m"]
        )


class CuteP1(unittest.TestCase):
    """to_kir step 6 under LKV_FRONTEND=ptx (translate_p1_ptx): the probe with a TMEM
    allocation by warp 0 and a tcgen05.ld of every warp's lane quarter (or one in a shape
    the P1 pass refuses), the entries legacy writes for each outcome."""

    tmp: Path
    setUp, run_kir = CuteProbe.setUp, CuteProbe.run_kir

    TMEM = """
    shr.u32 %r2, %r1, 5;
    setp.eq.s32 %p1, %r2, 0;
    mov.u32 %r3, tmem_holder;
    mov.b32 %r4, 512;
    @%p1 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r3], %r4;
    shl.b32 %r5, %r2, 21;
    tcgen05.ld.sync.aligned.SHAPE.x1.b32 {%r6}, [%r5];
    tcgen05.wait::ld.sync.aligned;"""

    def with_tmem(self, shape: str) -> None:
        ptx = self.tmp / "probe_cute.ptx"
        text = ptx.read_text().replace("%r<2>;", "%r<7>;\n\t.reg .pred \t%p<2>;")
        text = text.replace(
            ".address_size 64",
            ".address_size 64\n\n.shared .align 16 .b8 tmem_holder[16];",
        )
        text = text.replace("%r1, %tid.x;", "%r1, %tid.x;" + self.TMEM.replace("SHAPE", shape))  # fmt: skip
        ptx.write_text(text)

    def test_checked(self):
        self.with_tmem("32x32b")
        variant = self.run_kir()[0]
        p1 = variant["p1"]
        self.assertEqual((p1["status"], p1["roles"]), ("checked", [["warps0_3", 4, 1]]))
        k1 = json.loads(Path(p1["kir"]).read_text())
        self.assertEqual(kir.validate(k1), [])
        self.assertEqual(k1["tmem"], {"cols": 512, "base": "zero"})
        self.assertEqual([e["ev"] for e in events(k1["roles"][0]["body"])],
                         ["tmem", "tmem_wait", "gload", "gstore"])  # fmt: skip
        k = json.loads(Path(variant["kir"]).read_text())  # P0: unchanged by the P1 pass
        self.assertNotIn("tmem", k)
        self.assertEqual(
            [e["ev"] for e in events(k["roles"][0]["body"])], ["gload", "gstore"]
        )

    def test_refused(self):
        self.with_tmem("16x64b")  # not translated by the P1 pass (R5)
        variant = self.run_kir()[0]
        self.assertEqual(variant["p1"]["status"], "refused")
        self.assertIn("R5", variant["p1"]["error"])
        self.assertTrue(Path(variant["kir"]).exists())
        self.assertEqual(list((self.tmp / "kir").glob("*.p1.json")), [])

    def test_no_p1(self):
        self.with_tmem("32x32b")
        with mock.patch.dict(os.environ, PTX_MODE):
            cute_to_kir.run(
                SKILL / "tests/e2e/cases/probe_cute.json", self.tmp, p1=False
            )
        variant = json.loads((self.tmp / "kir/report.json").read_text())[0]
        self.assertNotIn("p1", variant)
        self.assertTrue(Path(variant["kir"]).exists())


def c(v, w=32):
    return {"e": "const", "w": w, "v": v}


def var(n, w=32):
    return {"e": "var", "w": w, "name": n}


def add(a, b, w=32):
    return {"e": "bin", "w": w, "o": "add", "a": a, "b": b}


def let(n, e):
    return {"op": "let", "var": n, "expr": e, "loc": "k.py:1"}


def store(off):
    """A gstore of x at element offset `off` (an i32 Expr)."""
    e = {"e": "ext", "w": 64, "signed": True, "a": off}
    return {"op": "event", "ev": "gstore", "tensor": "x", "mask": None, "loc": "k.py:2",
            "offsets": {"t": "splat", "shape": [1], "w": 64, "a": e}}  # fmt: skip


def thread_loop(t, body):
    return {"op": "for", "var": t, "lb": c(0), "ub": c(4), "step": c(1), "body": body,
            "loc": "k.py:2"}  # fmt: skip


def trace(body, params=None):
    facts = SimpleNamespace(roles=[{"body": body}])
    return evaluate(facts, params or {"n": 3})


class Reroll(unittest.TestCase):
    """cute/to_kir.py reroll (decision D8): LLVM's unrolled blocks back into a `for`."""

    def unrolled(self, offsets):
        body = [let("v0", var("n"))]
        for k, off in enumerate(offsets):
            body += [let(f"a{k}", add(var("v0"), c(off))),
                     thread_loop(f"t{k}", [store(add(var(f"t{k}"), var(f"a{k}")))])]  # fmt: skip
        return body

    def test_rolls_the_same_trace(self):
        body = self.unrolled([0, 16, 32, 48])
        got = cute_to_kir.reroll(body)
        self.assertEqual([s["op"] for s in got], ["let", "for"])
        self.assertEqual((got[1]["ub"]["v"], len(got[1]["body"])), (4, 2))
        self.assertEqual(trace(got), trace(body))
        self.assertEqual(len([e for e in trace(body) if e[0] == "gstore"]), 16)

    def test_kept(self):
        # Constants out of progression roll only the run that is one (0, 16; then 64).
        body = self.unrolled([0, 16, 64])
        got = cute_to_kir.reroll(body)
        self.assertEqual([s["op"] for s in got], ["let", "for", "let", "for"])
        self.assertEqual(trace(got), trace(body))
        # A name the run defines read after it (here a1 by the last store).
        body = self.unrolled([0, 16]) + [store(var("a1"))]
        self.assertEqual(cute_to_kir.reroll(body), body)
        # Blocks reassigning different outer variables (each reads its own first).
        body = [let("c1", c(0)), let("c2", c(0))]
        for k in (1, 2):
            body += [let(f"w{k}", add(var(f"c{k}"), c(1))), let(f"c{k}", var(f"w{k}"))]
        body += [store(add(var("c1"), var("c2")))]
        self.assertEqual(cute_to_kir.reroll(body), body)

    def test_reassigned_variable(self):
        # Every block reassigns the same visible variable: rolled, it ends as the last
        # block left it.
        body = [let("s", c(0))]
        for k in range(3):
            body += [let("s", add(var("s"), c(4 * k + 1))), store(var("s"))]
        body += [store(var("s"))]
        got = cute_to_kir.reroll(body)
        self.assertEqual([s["op"] for s in got], ["let", "for", "event"])
        self.assertEqual(trace(got), trace(body))


class JitNames(unittest.TestCase):
    """cute/to_kir.py _jit_named: the LaunchRecords' jit_params (host_patches.py P11)."""

    def table(self):
        return [
            {"ptx": "p0", "kind": "aggregate", "name": "param0", "size": 16,
             "fields": {0: {"size": 8, "kind": "ptr", "name": "mA2"},
                        8: {"size": 4, "kind": "scalar", "name": "param0_8"}}},
            {"ptx": "p1", "kind": "tensormap", "name": "param1", "size": 128, "fields": {},
             "base": "mB2", "aliases": ["mB", "mB2"]},
            {"ptx": "p2", "kind": "aggregate", "name": "param2", "size": 16,
             "fields": {0: {"size": 8, "kind": "ptr", "name": None}}},
            {"ptx": "p3", "kind": "ptr", "name": "param3", "size": 8, "fields": {}},
        ]  # fmt: skip

    def records(self, jit):
        two = [
            {"tensor": n, "match": "exact", "storage": [1, 9]} for n in ("mA", "mA2")
        ]
        view = [
            {"tensor": n, "match": "exact", "storage": [1, 9]} for n in ("mB", "mB2")
        ]
        return [{"params": [
            {"fields": [{"off": 0, "tensor": two}, {"off": 8}]},
            {"tensormap": {"base": view}, "fields": []},
            {"fields": [{"off": 0, "tensor": view}]},
            {"fields": []},
        ], "jit_params": jit}]  # fmt: skip

    def test_names(self):
        jit = [{"arg": "mA", "leaf": "mA"}, {"from": ["mB"]}, {"from": ["mB2", "mQ"]},
               {"arg": "e", "leaf": "e"}]  # fmt: skip
        tensors = dict.fromkeys(["mA", "mA2", "mB", "mB2", "e"])
        got = cute_to_kir._jit_named(self.table(), self.records(jit), tensors)
        # passed through: by its leaf, whatever its address (an alias; e: zero-length,
        # no address candidate); computed: the one candidate it is computed from
        self.assertEqual(got[0]["fields"][0]["name"], "mA")
        self.assertEqual((got[1]["base"], got[1]["aliases"]), ("mB", []))
        self.assertEqual(got[2]["fields"][0]["name"], "mB2")
        self.assertEqual(got[3]["name"], "e")
        self.assertEqual(got[0]["fields"][8]["name"], "param0_8")
        # No jit_params (an older dump, or the trace failed): the address naming stays.
        for recs in (
            self.records(None),
            self.records(jit) + self.records(jit[:2] + jit[3:]),
        ):
            self.assertEqual(
                cute_to_kir._jit_named(self.table(), recs, tensors), self.table()
            )

    def test_stream_not_last(self):
        """A TVM-FFI launch passes nothing for its environment stream, wherever the jit
        function has it (nvfp4_2cta_bwd: `..., stream, mCuSeqlensDKV, mGlobalScales`):
        the arguments after it bind to the parameters after it, as the jit_params agree,
        and a LaunchRecord candidate naming them shifted is an error."""

        def tensor(ptr, dtype="int32"):
            return {"kind": "tensor", "dtype": dtype, "shape": [3], "stride": [1],
                    "storage": [ptr, 12], "byte_offset": 0}  # fmt: skip

        names = ["n", "stream", "mCuSeqlensDKV", "mGlobalScales"]
        rec = {
            "params": [{"name": n, "constexpr": False, "leaves": [n]} for n in names]
        }
        args = [
            {"kind": "scalar", "type": "int", "value": 7},
            tensor(4096),
            tensor(8192, "float32"),
        ]
        cands = [
            [{"tensor": "mCuSeqlensDKV", "storage": [4096, 4108], "match": "exact"}],
            [{"tensor": "mGlobalScales", "storage": [8192, 8204], "match": "exact"}],
        ]
        record = {"params": [{"kind": "scalar", "size": 4, "value": 7}] + [
            {"kind": "aggregate", "fields": [{"off": 0, "kind": "ptr", "value": v, "tensor": c}]}
            for v, c in zip((4096, 8192), cands)],
            "jit_params": [{"arg": n, "leaf": n} for n in ("n", "mCuSeqlensDKV", "mGlobalScales")]}  # fmt: skip
        launched = cute_to_kir._launch_params(rec, [("a", args)], [record])
        self.assertEqual(
            [p["name"] for p in launched], ["n", "mCuSeqlensDKV", "mGlobalScales"]
        )
        named = dict(cute_to_kir._named_args(launched, args))
        self.assertEqual(named["mCuSeqlensDKV"]["storage"][0], 4096)
        self.assertEqual(named["mGlobalScales"]["dtype"], "float32")
        cute_to_kir._check_candidates(list(named.items()), record)
        # host_patches naming the arguments shifted (the scales as mCuSeqlensDKV)
        record["params"][2]["fields"][0]["tensor"] = [
            {**cands[0][0], "storage": [8192, 8204]}
        ]
        with self.assertRaisesRegex(cute_to_kir.Unsupported, "another tensor"):
            cute_to_kir._check_candidates(list(named.items()), record)
        # no choice of the skipped parameter agrees with the records (n is 7, not 8)
        record["params"][0]["value"] = 8
        with self.assertRaisesRegex(cute_to_kir.Unsupported, "no choice"):
            cute_to_kir._launch_params(rec, [("a", args)], [record])


class Descriptors(unittest.TestCase):
    """PtxVariant's KIR desc tensors from Facts.descs (tma.py), as Variant.tensors."""

    def test_desc_tensors(self):
        case = {
            "name": "c",
            "kernels": {
                "K": {
                    "tensors": {
                        "mQ": {
                            "segments": {"dim": 0, "offsets": "cu", "per": "batch"},
                            "role": "input",
                        }
                    }
                }
            },
        }
        rec = {
            "callable": "m.K",
            "flags": [],
            "params": [{"name": "mQ", "constexpr": False, "leaves": ["mQ"]}],
        }
        arg = {
            "kind": "tensor",
            "dtype": "bfloat16",
            "shape": [256, 2, 64],
            "stride": [128, 64, 1],
        }
        v = cute_to_kir.PtxVariant(
            case, {"dsl": {"version": "4.4.2"}}, rec, rec["params"], [arg]
        )
        dims = [
            var(f"tma0_mQ.{f}.{j}", 64) for f in ("shape", "stride") for j in range(3)
        ]
        v.descs = {
            "tma0_mQ": {"kind": "host", "base": "mQ", "elem_bytes": 2, "box": [1, 128, 64],
                        "shape": dims[:3], "strides": dims[3:], "uses": ["tma_load"]},
            "bulk_mQ_64": {"kind": "bulk", "base": "mQ", "elem_bytes": 2, "box": [64],
                           "shape": None, "strides": None, "uses": ["tma_load"]},
        }  # fmt: skip
        # the descriptor walks (heads, rows, cols): rows are the segmented tensor dim 0
        v.values = [dict(zip([d["name"] for d in dims], [2, 256, 64, 64, 128, 1]))]
        v.dims = [{"mQ": (arg["shape"], arg["stride"])}]
        host, bulk = [t for t in v.tensors() if t["kind"] == "desc"]
        # innermost first, as the legacy CuTe frontend: (cols, rows, heads)
        self.assertEqual((host["name"], host["segments"]["dim"], host["role"], host["box"]),
                         ("tma0_mQ", 1, "input", [64, 128, 1]))  # fmt: skip
        self.assertEqual([x["name"] for x in host["shape"]],
                         [f"tma0_mQ.shape.{j}" for j in (2, 1, 0)])  # fmt: skip
        self.assertEqual((bulk["segments"], bulk["strides"], len(bulk["shape"])),
                         (None, [c(1, 64)], 1))  # fmt: skip
        v.values[0]["tma0_mQ.stride.1"] = 64  # no unique match: no segments
        self.assertIsNone(v.tensors()[-2]["segments"])
        v.values[0]["tma0_mQ.stride.1"] = (
            128  # (heads, rows, cols) -> (rows, cols) desc
        )
        body = [{"op": "event", "ev": "tma_load", "desc": d, "coords": [c(0), c(1), c(2)][: len(b)],
                 "bar": "mbar_0", "idx": c(0), "loc": "k.py:1"}
                for d, b in (("tma0_mQ", [1, 128, 64]), ("bulk_mQ_64", [64]))]  # fmt: skip
        got = cute_to_kir._innermost_first(body, {"tma0_mQ"})
        self.assertEqual([[x["v"] for x in e["coords"]] for e in got], [[2, 1, 0], [0]])


# A store through a parameter pointer, a generic store into shared memory and a store
# through a pointer loaded from memory (any tensor).
WRITES = kernel(
    """
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd1, %rd3;
    st.global.b32 [%rd4], %r1;
    mov.u64 %rd5, smem;
    st.b32 [%rd5+8], %r1;
    LOADED
    ret;
""",
    ptrs=("x", "y"),
    scalars=(),
)
PARAMS = [{"ptx": f"k_param_{i}", "kind": "ptr", "name": n, "size": 8, "fields": {}}
          for i, n in enumerate("xy")]  # fmt: skip


class WriteSet(unittest.TestCase):
    def test_named_pointers_only(self):
        self.assertEqual(T.writes(WRITES.replace("LOADED", ""), "k", PARAMS), ["x"])

    def test_loaded_pointer_is_unknown(self):
        loaded = "ld.global.u64 %rd7, [%rd2];\n    st.global.b32 [%rd7], %r1;"
        self.assertIsNone(T.writes(WRITES.replace("LOADED", loaded), "k", PARAMS))

    def test_call_is_unknown(self):
        # A called device function may store anywhere (the op table refuses the call),
        # here into y, which the entry itself never writes.
        func = """
.func store_it(.param .b64 store_it_param_0)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.b64 %rd1, [store_it_param_0];
    st.global.b32 [%rd1], %r1;
    ret;
}
"""
        call = """{
    .param .b64 param0;
    st.param.b64 [param0], %rd2;
    call.uni store_it, (param0);
    }"""
        text = WRITES.replace("LOADED", call).replace(
            ".visible .entry", func + ".visible .entry"
        )
        self.assertIsNone(T.writes(text, "k", PARAMS))


# Work fields under the TTGIR gate: `batch` is the shr's result; `head` runs at the same
# place but no instruction holds its value (anchors.py "novalue", e.g. an unused off_h).
GATED = """
    mov.u32 %r1, %tid.x;
    ld.param.u64 %rd1, [k_param_0];
    mov.u32 %r2, %ctaid.y;
    shr.u32 %r3, %r2, 1;
    mul.wide.u32 %rd2, %r3, 4;
    add.s64 %rd3, %rd1, %rd2;
    setp.eq.s32 %p1, %r1, 0;
    @%p1 st.global.b32 [%rd3], %r1;
    ret;
"""


class NoValueAnchor(unittest.TestCase):
    def facts(self, per):
        fn = parse.parse(kernel(GATED, scalars=())).function("k")
        k = next(j for j, i in enumerate(fn.instructions()) if i.text.startswith("shr"))
        at = ("k", k, fn.instructions()[k].line)
        sites = [{"field": "batch", "kind": "assign", "instr": at, "detail": {"reg": "%r3"}},
                 {"field": "head", "kind": "novalue", "instr": at, "detail": {"var": "off_h"}}]  # fmt: skip
        work = {"fields": ["batch", "head"], "mode": "gate", "per": per, "sites": sites}
        return run(GATED, scalars=(), work=work)

    def test_defined_without_value(self):
        f = self.facts([])
        (w,) = [e for r in f.roles for e in events(r["body"]) if e["ev"] == "work"]
        self.assertEqual(w["fields"]["head"], {"e": "const", "w": 64, "v": -1})
        self.assertEqual(w["fields"]["batch"]["e"], "var")

    def test_refused_for_segments(self):
        (refusal,) = self.facts(["head"]).refusals
        self.assertEqual(refusal["kind"], "error")
        self.assertIn("no instruction holds its value", refusal["msg"])


class TtgirAssembly(unittest.TestCase):
    """frontend/ttgir/to_kir.py build_ptx's parts for TMA descriptors and CLC."""

    def test_desc_blocks(self):
        def launch(block):
            return {"args": {"d": {"kind": "desc", "block_shape": block},
                             "x": {"kind": "tensor"}}}  # fmt: skip

        two = [({"name": "a"}, launch([128, 128])), ({"name": "b"}, launch([128, 128]))]
        self.assertEqual(ttgir_to_kir._desc_blocks(two), {"d": [128, 128]})
        two[1] = ({"name": "b"}, launch([128, 64]))
        with self.assertRaisesRegex(
            ttgir_to_kir.KirError, "input b: host descriptor d"
        ):
            ttgir_to_kir._desc_blocks(two)

    def test_desc_tensors(self):
        shape = [{"e": "var", "w": 32, "name": "d.shape.0"}]
        stride = [{"e": "var", "w": 64, "name": "d.stride.0"}]
        descs = {"d": {"kind": "host", "base": "d", "elem_bytes": 2, "box": [64],
                       "shape": shape, "strides": stride, "uses": ["tma_store"]}}  # fmt: skip
        seg = {"dim": 0, "offsets": "off", "per": "batch"}
        (t,) = ttgir_to_kir._desc_tensors(descs, {"d": {"segments": seg}}, {"d"})
        self.assertEqual(
            (t["kind"], t["box"], t["shape"], t["strides"], t["role"], t["segments"]),
            ("desc", [64], shape, stride, "output", seg),
        )
        descs["d"]["kind"] = "bulk"
        with self.assertRaisesRegex(ttgir_to_kir.KirError, "not in the Triton family"):
            ttgir_to_kir._desc_tensors(descs, {}, set())

    def test_clc_tables(self):
        tensors, params = [], []
        ttgir_to_kir._add_clc(tensors, params)
        self.assertEqual([(t["name"], t["kind"], t["metadata"]) for t in tensors],
                         [("__clc_response", "desc", False), ("__clc_schedule", "global", True),
                          ("__clc_launched", "global", True)])  # fmt: skip
        self.assertEqual(len(params), 6)  # shape and stride of 2 + 1 dims


class Switch(unittest.TestCase):
    def test_values(self):
        for to_kir in (ttgir_to_kir, cute_to_kir):
            for value, want in (("", "legacy"), ("legacy", "legacy"), ("ptx", "ptx")):
                with mock.patch.dict(os.environ, {"LKV_FRONTEND": value}):
                    self.assertEqual(to_kir.frontend(), want)
            with mock.patch.dict(os.environ, {"LKV_FRONTEND": "mlir"}):
                self.assertRaises(SystemExit, to_kir.frontend)


if __name__ == "__main__":
    unittest.main()
