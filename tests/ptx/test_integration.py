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
refused variant (translate.writes; a call or a loaded pointer leaves it unknown), and
the switch's values.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
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
from .fixtures.kernels import kernel, run  # noqa: E402

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
