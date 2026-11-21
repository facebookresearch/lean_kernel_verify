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
(LLVM's unrolled blocks back into a `for`, the same trace; runs it must
keep), the jit_params naming of pointers and descriptor bases, and the KIR desc tensors
of Facts.descs (a segmented base's segments on the walking dim; a bulk descriptor's
extent). TTGIR assembly (frontend/ttgir/to_kir.py build_ptx): host descriptor blocks
(equal in every launch), the KIR desc tensors of Facts.descs (annotations, role) and the
CLC tables; translate_nb in forked processes (the sequential results) and its
all-preferred `<spec>@cluster<dims>` P0-6 KIR, translate_p1's P1 KIR per geometry, and
to_kir's report.json (the variants, their inputs and results; what a to_kir that stops
leaves; the legacy translation's).
"""

import contextlib
import io
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
from ...frontend.ptx import nbar, parse, translate as T  # noqa: E402
from ...frontend.ttgir import to_kir as ttgir_to_kir  # noqa: E402
from ...frontend.ttgir.slicing import SliceError  # noqa: E402
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


class PreferredGeometryRules(unittest.TestCase):
    """preferred_geometry refuses (gap) launches that break a rule CUDA states for the
    preferred cluster attribute: no regular cluster (cluster dims 1 x 1 x 1 and no
    REQUIRED_CLUSTER), a preferred x or y that is no multiple of the regular one or a z
    that differs from it, a grid the preferred dims do not divide."""

    @staticmethod
    def dump(*launches) -> dict:
        return {"specs": {"s": {}},
                "inputs": [{"name": f"i{k}", "launches": [{"spec": "s", "launch": la}]}
                           for k, la in enumerate(launches)]}  # fmt: skip

    @staticmethod
    def launch(cluster, pref, grid, required=None) -> dict:
        attrs = {"preferred_cluster": pref}
        if required is not None:
            attrs["func_attributes"] = {"required_cluster": required}
        return {"cluster": cluster, "grid": grid, "attrs": attrs}

    def test_rules(self):
        la, geo = self.launch, ttgir_to_kir.preferred_geometry
        self.assertEqual(
            geo(self.dump(la([2, 1, 1], [4, 2, 1], [8, 4, 1])), "s"), (4, 2, 1)
        )
        required = la([1, 1, 1], [3, 1, 1], [6, 1, 1], required=[1, 1, 1])
        self.assertEqual(geo(self.dump(required), "s"), (3, 1, 1))
        cases = [
            ("without a regular cluster", [la([1, 1, 1], [2, 1, 1], [4, 1, 1])]),
            ("without a regular cluster", [la([1, 1, 1], [3, 1, 1], [6, 1, 1]), required]),
            ("no multiple of the regular (2, 1, 1)", [la([2, 1, 1], [3, 1, 1], [6, 1, 1])]),
            ("no multiple of the regular (2, 1, 1)", [la([2, 1, 1], [2, 1, 2], [2, 1, 2])]),
            ("do not divide the grid of input(s) i1, i2",
             [la([1, 1, 1], [2, 1, 1], [g, 1, 1], required=[1, 1, 1]) for g in (4, 3, 1)]),
        ]  # fmt: skip
        for msg, launches in cases:
            with self.subTest(msg):
                with self.assertRaises(ttgir_to_kir.KirError) as e:
                    geo(self.dump(*launches), "s")
                self.assertIn(msg, str(e.exception))
                self.assertEqual(ttgir_to_kir.failure_kind(e.exception), "gap")


class Geometries(unittest.TestCase):
    """geometries: the variants the P0 translation of the PTX, P0-6 and P1-2 check alike,
    decided before anything is translated: the regular cluster dims, the all-preferred
    geometry or the refused `<spec>@cluster`, each refused with a launch's required
    cluster that differs from its cluster dims (a refused `@cluster` keeps its reason)."""

    dump = staticmethod(PreferredGeometryRules.dump)
    launch = staticmethod(PreferredGeometryRules.launch)

    def variants(self, *launches) -> list:
        got = ttgir_to_kir.geometries(self.dump(*launches), "s")
        return [(n, g, r if r is None else (ttgir_to_kir.failure_kind(r), str(r)))
                for n, g, r in got]  # fmt: skip

    def test_variants(self):
        la = self.launch
        self.assertEqual(
            self.variants(la([2, 1, 1], None, [8, 1, 1])), [("s", None, None)]
        )
        self.assertEqual(self.variants(la([2, 1, 1], [4, 1, 1], [8, 1, 1])),
                         [("s", None, None), ("s@cluster4x1x1", (4, 1, 1), None)])  # fmt: skip
        regular, refused = self.variants(
            la([2, 1, 1], [4, 1, 1], [8, 1, 1]), la([2, 1, 1], [8, 1, 1], [8, 1, 1])
        )
        self.assertEqual(regular, ("s", None, None))
        self.assertEqual(refused[:2], ("s@cluster", None))
        self.assertEqual(refused[2][0], "gap")
        self.assertIn("differ in their (preferred) cluster dims", refused[2][1])

    def test_required_cluster_refuses_every_geometry(self):
        la = self.launch
        required = (
            "function's required cluster (cuFuncSetAttribute REQUIRED_CLUSTER) is"
        )
        got = self.variants(la([1, 1, 1], [2, 1, 1], [4, 1, 1], required=[2, 1, 1]))
        self.assertEqual([(n, g) for n, g, _ in got],
                         [("s", None), ("s@cluster2x1x1", (2, 1, 1))])  # fmt: skip
        for _n, _g, (kind, msg) in got:
            self.assertEqual(kind, "gap")
            self.assertIn(required, msg)
        regular, refused = self.variants(
            la([1, 1, 1], [2, 1, 1], [4, 1, 1], required=[2, 1, 1]),
            la([1, 1, 1], [4, 1, 1], [4, 1, 1], required=[2, 1, 1]),
        )
        self.assertIn(required, regular[2][1])
        self.assertEqual(refused[0], "s@cluster")
        self.assertIn("differ in their (preferred) cluster dims", refused[2][1])


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

    def test_preferred_without_regular_cluster(self):
        """A preferred cluster beside no regular one (cluster dims 1 x 1 x 1, no
        REQUIRED_CLUSTER): CUDA uses it only with one, so no all-preferred KIR is written,
        each specialization has the refused `<spec>@cluster` (a gap) and keeps its
        regular KIR."""
        dump = json.loads(self.dump.read_text())
        for inp in dump["inputs"]:
            for la in inp["launches"]:
                la["launch"]["attrs"]["preferred_cluster"] = [2, 1, 1]
        self.dump.write_text(json.dumps(dump))
        with mock.patch.dict(os.environ, PTX_MODE):
            got = ttgir_to_kir.translate(self.dump, self.case)
        self.assertEqual(list(got), ["1691f771b8a0", "1691f771b8a0@cluster",
                                     "6614c8dcb51d", "6614c8dcb51d@cluster"])  # fmt: skip
        for key in ("1691f771b8a0", "6614c8dcb51d"):
            self.assertIsInstance(got[key], dict)
            e = got[key + "@cluster"]
            self.assertIsInstance(e, ttgir_to_kir.KirError)
            self.assertEqual(ttgir_to_kir.failure_kind(e), "gap")
            self.assertIn("(2, 1, 1) without a regular cluster", str(e))
        self.assertEqual(list(self.tmp.glob("*@*.kir.json")), [])

    def test_preferred_geometry_two_launches(self):
        """An input with several launches of one specialization and a preferred cluster:
        the all-preferred KIR names them as the regular KIR does, `<input>#<i>` by their
        order in the input (the other specialization, a grid of whose launches the
        preferred dims do not divide, has the refused `<spec>@cluster` instead),
        and every KIR translates (a copy with the preferred dims used to be looked up by
        equality, a ValueError that stopped the whole translation); the P0-6 KIRs too."""
        dump = json.loads(self.dump.read_text())
        for inp in dump["inputs"]:
            for la in inp["launches"]:
                la["launch"]["attrs"]["preferred_cluster"] = [2, 1, 1]
                la["launch"]["attrs"]["func_attributes"] = {
                    "required_cluster": [1, 1, 1]
                }
        (inp,) = [i for i in dump["inputs"] if i["name"] == "aligned"]
        for grid, index in (([4, 1, 1], 6), ([2, 1, 1], 7)):
            la = json.loads(json.dumps(inp["launches"][0]))
            la["index"], la["grid"], la["launch"]["grid"] = index, grid, grid
            inp["launches"].append(la)
        self.dump.write_text(json.dumps(dump))
        with mock.patch.dict(os.environ, PTX_MODE):
            got = ttgir_to_kir.translate(self.dump, self.case)
        self.assertEqual(list(got), ["1691f771b8a0", "1691f771b8a0@cluster2x1x1",
                                     "6614c8dcb51d", "6614c8dcb51d@cluster"])  # fmt: skip
        e = got["6614c8dcb51d@cluster"]
        self.assertIn("do not divide the grid of input(s) below_block, partial", str(e))
        for name, k in got.items():
            if k is not e:
                self.assertIsInstance(k, dict, f"{name}: {k}")
                self.assertEqual(kir.validate(k), [], name)
        self.assertEqual(
            [i["name"] for i in got["1691f771b8a0"]["case"]["inputs"]],
            ["aligned#0", "aligned#1", "aligned#2", "empty"],
        )  # fmt: skip
        self.assertEqual(
            [i["name"] for i in got["1691f771b8a0@cluster2x1x1"]["case"]["inputs"]],
            ["aligned#0", "aligned#1", "aligned#2", "empty"],
        )
        nb = ttgir_to_kir.translate_nb(self.dump, self.case)  # the P0-6 KIRs alike
        self.assertEqual(list(nb), list(got))
        self.assertEqual(str(nb.pop("6614c8dcb51d@cluster")), str(e))
        for name, k in nb.items():
            self.assertIsInstance(k, dict, f"{name}: {k}")
            self.assertEqual([i["name"] for i in k["case"]["inputs"]],
                             [i["name"] for i in got[name]["case"]["inputs"]])  # fmt: skip

    def test_nb_parallel(self):
        """translate_nb's forked workers (--jobs) give the sequential results: the same
        P0-6 KIRs, files and order."""
        seq = ttgir_to_kir.translate_nb(self.dump, self.case)
        files = {k: (self.tmp / f"{k}.nb.json").read_bytes() for k in seq}
        for f in self.tmp.glob("*.nb.json"):
            f.unlink()
        par = ttgir_to_kir.translate_nb(self.dump, self.case, jobs=2)
        self.assertEqual(list(par), list(seq))
        self.assertEqual(par, seq)
        self.assertTrue(all(isinstance(k, dict) for k in par.values()), par)
        self.assertEqual(
            {k: (self.tmp / f"{k}.nb.json").read_bytes() for k in par}, files
        )

    def test_nb_preferred_geometry(self):
        """A launch with a preferred cluster gets a second P0-6 KIR per specialization,
        `<spec>@cluster<dims>`, in the all-preferred geometry (its cluster dims), next to
        the regular one, which does not change; a specialization with a launch whose grid
        the preferred dims do not divide has the refused `<spec>@cluster` instead."""
        regular = ttgir_to_kir.translate_nb(self.dump, self.case)
        dump = json.loads(self.dump.read_text())
        for inp in dump["inputs"]:
            for la in inp["launches"]:
                la["launch"]["attrs"]["preferred_cluster"] = [2, 1, 1]
                la["launch"]["attrs"]["func_attributes"] = {
                    "required_cluster": [1, 1, 1]
                }
        self.dump.write_text(json.dumps(dump))
        got = ttgir_to_kir.translate_nb(self.dump, self.case, jobs=2)
        names = ["1691f771b8a0", "1691f771b8a0@cluster2x1x1",
                 "6614c8dcb51d", "6614c8dcb51d@cluster"]  # fmt: skip
        self.assertEqual(list(got), names)
        for key in regular:
            self.assertEqual(got[key], regular[key])
        e = got["6614c8dcb51d@cluster"]
        self.assertEqual(ttgir_to_kir.failure_kind(e), "gap")
        self.assertIn("do not divide the grid of input(s) below_block, partial", str(e))
        self.assertEqual(list(self.tmp.glob("6614c8dcb51d@*.nb.json")), [])
        name = "1691f771b8a0@cluster2x1x1"
        k = got[name]
        self.assertEqual(kir.validate(k), [])
        self.assertEqual(json.loads((self.tmp / f"{name}.nb.json").read_text()), k)
        self.assertEqual(
            [c["v"] for c in k["cluster"]]
            if isinstance(k["cluster"][0], dict)
            else k["cluster"],
            [2, 1, 1],
        )
        self.assertEqual([i["name"] for i in k["case"]["inputs"]], ["aligned", "empty"])
        self.assertTrue(k["case"]["name"].endswith(name))

    def test_preferred_dims_disagree(self):
        """Launches of a specialization that disagree on the preferred cluster dims: its
        regular KIR is kept and written, and the all-preferred geometry is the refused
        variant `<spec>@cluster` (a gap, no dims known); the other specialization's
        geometries are unaffected. The P0-6 KIRs alike."""
        dump = json.loads(self.dump.read_text())
        for inp in dump["inputs"]:
            for la in inp["launches"]:
                pref = [4, 1, 1] if inp["name"] == "partial" else [2, 1, 1]
                la["launch"]["attrs"]["preferred_cluster"] = pref
                la["launch"]["attrs"]["func_attributes"] = {
                    "required_cluster": [1, 1, 1]
                }
        self.dump.write_text(json.dumps(dump))
        with mock.patch.dict(os.environ, PTX_MODE):
            got = ttgir_to_kir.translate(self.dump, self.case)
        self.assertEqual(list(got), ["1691f771b8a0", "1691f771b8a0@cluster2x1x1",
                                     "6614c8dcb51d", "6614c8dcb51d@cluster"])  # fmt: skip
        k = got["6614c8dcb51d"]
        self.assertIsInstance(k, dict)
        self.assertEqual(
            json.loads((self.tmp / "6614c8dcb51d.kir.json").read_text()), k
        )
        e = got["6614c8dcb51d@cluster"]
        self.assertIsInstance(e, ttgir_to_kir.KirError)
        self.assertEqual(ttgir_to_kir.failure_kind(e), "gap")
        self.assertIn("differ in their (preferred) cluster dims", str(e))
        self.assertEqual(list(self.tmp.glob("6614c8dcb51d@*.kir.json")), [])
        nb = ttgir_to_kir.translate_nb(self.dump, self.case)  # the P0-6 KIRs alike
        self.assertEqual(list(nb), list(got))
        self.assertIsInstance(nb["6614c8dcb51d"], dict)
        self.assertEqual(str(nb["6614c8dcb51d@cluster"]), str(e))

    def test_nb_required_cluster(self):
        """A launch whose function's required cluster differs from its cluster dims
        refuses the P0-6 KIR too (gap), with the P0 KIR's reason: neither checks a
        geometry that may not be the one that runs."""
        dump = json.loads(self.dump.read_text())
        for inp in dump["inputs"]:
            for la in inp["launches"]:
                la["launch"]["attrs"]["func_attributes"] = {
                    "required_cluster": [2, 1, 1]
                }
        self.dump.write_text(json.dumps(dump))
        with mock.patch.dict(os.environ, PTX_MODE):
            p0 = ttgir_to_kir.translate(self.dump, self.case)
        got = ttgir_to_kir.translate_nb(self.dump, self.case)
        self.assertEqual(list(got), list(p0))
        for key, e in got.items():
            self.assertIsInstance(e, ttgir_to_kir.KirError, key)
            self.assertEqual(ttgir_to_kir.failure_kind(e), "gap")
            self.assertIn("function's required cluster", str(e))
            self.assertEqual(str(e), str(p0[key]))
        self.assertEqual(list(self.tmp.glob("*.nb.json")), [])

    def test_required_cluster_in_every_pass(self):
        """A launch whose function's required cluster differs from its cluster dims
        refuses each geometry of its specialization, the same variants with the same
        reasons in the P0, P0-6 and P1-2 passes (a refused `<spec>@cluster` keeps its
        own); no KIR is written."""
        dump = json.loads(self.dump.read_text())
        for inp in dump["inputs"]:
            for la in inp["launches"]:
                la["launch"]["attrs"]["preferred_cluster"] = [2, 1, 1]
                la["launch"]["attrs"]["func_attributes"] = {
                    "required_cluster": [2, 1, 1]
                }
        self.dump.write_text(json.dumps(dump))
        with mock.patch.dict(os.environ, PTX_MODE):
            p0 = ttgir_to_kir.translate(self.dump, self.case)
        nb = ttgir_to_kir.translate_nb(self.dump, self.case)
        p1 = ttgir_to_kir.translate_p1(self.dump, self.case)
        self.assertEqual(list(p0), ["1691f771b8a0", "1691f771b8a0@cluster2x1x1",
                                    "6614c8dcb51d", "6614c8dcb51d@cluster"])  # fmt: skip
        for got in (p0, nb, p1):
            self.assertEqual({k: (ttgir_to_kir.failure_kind(e), str(e)) for k, e in got.items()},
                             {k: ("gap", str(e)) for k, e in p0.items()})  # fmt: skip
        for key, e in p0.items():
            want = (
                "do not divide the grid"
                if key == "6614c8dcb51d@cluster"
                else "function's required cluster"
            )
            self.assertIn(want, str(e))
        kirs = [p for s in ("kir", "nb", "p1") for p in self.tmp.glob(f"*.{s}.json")]
        self.assertEqual(kirs, [])

    def test_tasks_in_forked_processes(self):
        """to_kir's tasks (P0, P0-6 and P1 of every specialization) in forked processes
        (--jobs) give the sequential results: the same exit status, log lines, files and
        report.json; the lines in task order, the P0 ones, then the P0-6 and the P1
        ones."""
        case = SKILL / "tests/e2e/cases/probe_tlx.json"

        def to_kir(jobs: int):
            for f in self.tmp.glob("*.*.json"):
                if f.name.endswith((".kir.json", ".nb.json", ".p1.json")):
                    f.unlink()
            err = io.StringIO()
            argv = [
                "to_kir",
                "--dump",
                str(self.dump),
                "--case",
                str(case),
                "--jobs",
                str(jobs),
            ]
            with (
                mock.patch.dict(os.environ, {**PTX_MODE, "LKV_P1": "1"}),
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stderr(err),
                self.assertRaises(SystemExit) as stop,
            ):
                ttgir_to_kir.main()
            files = {f.name: f.read_bytes() for f in sorted(self.tmp.glob("*.json"))
                     if f.name.endswith((".kir.json", ".nb.json", ".p1.json"))
                     or f.name == "report.json"}  # fmt: skip
            return stop.exception.code, err.getvalue().splitlines(), files

        seq = to_kir(1)
        self.assertEqual(to_kir(3), seq)
        code, lines, files = seq
        self.assertEqual(code, 0)
        self.assertEqual(lines[0], "[to_kir] LKV_FRONTEND=ptx: translating the PTX")
        self.assertEqual(
            [x.split(": ", 1)[1].split(" ")[0] for x in lines[1:]],
            [
                "1691f771b8a0.kir.json",
                "6614c8dcb51d.kir.json",
                "P0-6",
                "P0-6",
                "P1",
                "P1",
            ],
        )
        self.assertTrue(
            lines[-1].endswith(
                ": P1 not_applicable: the kernel has no bulk async store"
            )
        )
        self.assertEqual(len(files), 5)  # the P0 and P0-6 KIRs, report.json
        not_applicable = {"status": "not_applicable", "specs": ["P1-2"],
                          "error": "the kernel has no bulk async store"}  # fmt: skip
        self.assertEqual(
            json.loads(files["report.json"]),
            [{"variant": k, "kernel": "probe", "inputs": inputs,
              "kir": str(self.tmp / f"{k}.kir.json"),
              "nb": {"status": "checked", "kir": str(self.tmp / f"{k}.nb.json")},
              "p1": not_applicable}
             for k, inputs in (("1691f771b8a0", ["aligned", "empty"]),
                               ("6614c8dcb51d", ["partial", "partial_floor", "below_block",
                                                 "below_block_floor"]))],
        )  # fmt: skip

    def test_p1_without_a_bulk_store_before_any_translation(self):
        """A kernel without a bulk store has no P1 KIR, decided on its parsed PTX before
        it is translated, so also where the translation would fail (that used to be
        P1-2 refused or error instead of not applicable)."""

        def translate(*args, **kw):
            raise AssertionError("translated")

        with mock.patch.object(T, "translate", translate):
            got = ttgir_to_kir.translate_p1(self.dump, self.case)
        self.assertEqual(got, {"1691f771b8a0": None, "6614c8dcb51d": None})

    def test_ptx_parsed_once_per_task(self):
        """Each P0 and P0-6 KIR parses its PTX once: the core translates the parsed module
        (it used to parse it again)."""
        real, texts = parse.parse, []

        def counted(text, *args, **kw):
            texts.append(text)
            return real(text, *args, **kw)

        with (
            mock.patch.dict(os.environ, PTX_MODE),
            mock.patch.object(parse, "parse", counted),
        ):
            p0 = ttgir_to_kir.translate(self.dump, self.case)
            nb = ttgir_to_kir.translate_nb(self.dump, self.case)
        self.assertTrue(all(isinstance(k, dict) for k in [*p0.values(), *nb.values()]))
        self.assertEqual(len(texts), 4)

    def test_ptx_parsed_once_per_task_with_gsync(self):
        """With the case's `gsync` too: gsync.py's census and every round of the
        translation read the task's parsed module (each parsed the PTX again); the probe
        has no sync tensor, so its KIRs are those without `gsync`."""
        real, texts = parse.parse, []

        def counted(text, *args, **kw):
            texts.append(text)
            return real(text, *args, **kw)

        with mock.patch.dict(os.environ, PTX_MODE):
            want = ttgir_to_kir.translate(self.dump, self.case)
            with mock.patch.object(parse, "parse", counted):
                got = ttgir_to_kir.translate(self.dump, {**self.case, "gsync": True})
        self.assertEqual(got, want)
        self.assertTrue(all(isinstance(k, dict) for k in got.values()))
        self.assertEqual(len(texts), 2)

    def test_nb_metadata_checked_before_the_pass(self):
        """A 1-D tensor with recorded contents that a launch cannot hold as metadata
        (_metadata_problem: here it shares storage with a written tensor, itself) is left
        out of the barrier pass's metadata before the pass, which runs once per P0-6 KIR
        (it used to run again for each tensor _input rejected), with the write set it
        would compute itself (nb_written)."""
        dump = json.loads(self.dump.read_text())
        for inp in dump["inputs"]:
            for la in inp["launches"]:
                la["args"]["x_ptr"]["values"] = [0] * la["args"]["x_ptr"]["shape"][0]
        self.dump.write_text(json.dumps(dump))
        real, pols = nbar.translate, []

        def spy(text, entry, records, pol, module=None):
            pols.append(pol)
            self.assertIsNotNone(module)
            return real(text, entry, records, pol, module)

        with mock.patch.object(nbar, "translate", spy):
            got = ttgir_to_kir.translate_nb(self.dump, self.case)
        self.assertTrue(all(isinstance(k, dict) for k in got.values()), got)
        self.assertEqual([(p["nb_meta"], p["nb_written"]) for p in pols],
                         [(frozenset(), ["x_ptr"])] * 2)  # fmt: skip

    def test_nb_metadata_contents_given_by_the_case(self):
        """A tensor whose contents the case gives (`metadata_contents`: an earlier launch
        writes them, so the recorded ones are stale) is not checked for the inputs it
        gives them for, as _input does not check it: it stays in the barrier pass's
        metadata where _metadata_problem would leave it out (the test above), and is
        checked for the other inputs."""
        dump = json.loads(self.dump.read_text())
        for inp in dump["inputs"]:
            for la in inp["launches"]:
                la["args"]["x_ptr"]["values"] = [0] * la["args"]["x_ptr"]["shape"][0]
        self.dump.write_text(json.dumps(dump))
        real, pols = nbar.translate, []

        def spy(text, entry, records, pol, module=None):
            pols.append(pol)
            return real(text, entry, records, pol, module)

        # `partial` gives them, the other inputs of its specialization do not
        one = {**self.case, "inputs": [{**i, "metadata_contents": {"x_ptr": []}}
                                       if i["name"] == "partial" else i
                                       for i in self.case["inputs"]]}  # fmt: skip
        with mock.patch.object(nbar, "translate", spy):
            every = ttgir_to_kir.translate_nb(
                self.dump, {**self.case, "metadata_contents": {"x_ptr": []}}
            )
            some = ttgir_to_kir.translate_nb(self.dump, one)
        self.assertTrue(
            all(isinstance(k, dict) for k in [*every.values(), *some.values()])
        )
        self.assertEqual([p["nb_meta"] for p in pols],
                         [frozenset({"x_ptr"})] * 2 + [frozenset()] * 2)  # fmt: skip


class MetadataProblems(unittest.TestCase):
    """_metadata_problem: why a launch's tensor cannot be a metadata tensor, with the
    message _input raises after the input's name (build_nb asks it before the pass)."""

    REC = {"kind": "tensor", "shape": [4], "strides": [1], "storage_offset": 0,
           "storage": [0, 16], "values": [0, 1, 2, 3]}  # fmt: skip
    OUT = {"kind": "tensor", "shape": [4], "strides": [1], "storage_offset": 0,
           "storage": [64, 80]}  # fmt: skip

    def problem(self, written=("o",), stale=None, **rec):
        launch = {
            "args": {"m": {**self.REC, **rec}, "o": self.OUT},
            "stale": stale or {},
        }
        return ttgir_to_kir._metadata_problem("m", launch, set(written))

    def test_problems(self):
        self.assertIsNone(self.problem())
        self.assertEqual(self.problem(storage_offset=1), "metadata tensor m is a view")
        self.assertIn("metadata tensor m has strides [2]", self.problem(strides=[2]))
        writer = [{"launch": 0, "kernel": "w", "param": "out"}]
        self.assertIn("metadata tensor m may hold data that launch 0 of w (argument out)",
                      self.problem(stale={"m": writer}))  # fmt: skip
        self.assertEqual(self.problem(storage=[60, 76]),
                         "metadata tensor m shares storage with o, which the analyzed kernel writes")  # fmt: skip
        self.assertIsNone(self.problem(written=(), storage=[60, 76]))
        rec = dict(self.REC)
        del rec["values"]
        launch = {"args": {"m": rec, "o": self.OUT}, "stale": {}}
        self.assertEqual(ttgir_to_kir._metadata_problem("m", launch, set()),
                         "metadata tensor m contents were not recorded")  # fmt: skip


class TtgirP1Geometry(unittest.TestCase):
    """translate_p1: with a preferred cluster geometry also the all-preferred P1 KIR,
    `<spec>@cluster<dims>.p1.json` (as translate's P0 KIRs)."""

    # what to_kir's summary of a KIR reads, for a stand-in build_ptx's KIR
    BARE = {"roles": [], "tensors": [], "barriers": [], "case": {"inputs": []}}

    def test_p1_kir_per_geometry(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        rec = {"cluster": [2, 1, 1], "grid": [8, 1, 1],
               "attrs": {"preferred_cluster": [4, 1, 1]}}  # fmt: skip
        dump = {"specs": {"s": {}},
                "inputs": [{"name": "i", "launches": [{"spec": "s", "launch": rec}]}]}  # fmt: skip
        (tmp / "dump.json").write_text(json.dumps(dump))
        calls = []

        def build(where, d, key, case, geometry=None, p1=False):
            calls.append((key, geometry, p1))
            return {"p1": ["P1-2"], "geometry": geometry, **self.BARE}

        with (
            mock.patch.object(ttgir_to_kir, "build_ptx", build),
            mock.patch.object(ttgir_to_kir.kir_contract, "validate", lambda k: []),
        ):
            results = ttgir_to_kir.translate_p1(tmp / "dump.json", {})
        self.assertEqual(calls, [("s", None, True), ("s", (4, 1, 1), True)])
        self.assertEqual(sorted(results), ["s", "s@cluster4x1x1"])
        written = json.loads((tmp / "s@cluster4x1x1.p1.json").read_text())
        self.assertEqual(written["geometry"], [4, 1, 1])

    def test_p1_preferred_dims_disagree(self):
        # launches that disagree on the preferred dims: the regular P1 KIR is checked and
        # the all-preferred one is a refused entry, `<spec>@cluster` (gap), as P0's is
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        recs = [{"cluster": [2, 1, 1], "grid": [8, 1, 1], "attrs": {"preferred_cluster": p}}
                for p in ([4, 1, 1], [8, 1, 1])]  # fmt: skip
        dump = {"specs": {"s": {}},
                "inputs": [{"name": f"i{k}", "launches": [{"spec": "s", "launch": r}]}
                           for k, r in enumerate(recs)]}  # fmt: skip
        (tmp / "dump.json").write_text(json.dumps(dump))
        calls = []

        def build(where, d, key, case, geometry=None, p1=False):
            calls.append((key, geometry, p1))
            return {"p1": ["P1-2"], "geometry": geometry, **self.BARE}

        with (
            mock.patch.object(ttgir_to_kir, "build_ptx", build),
            mock.patch.object(ttgir_to_kir.kir_contract, "validate", lambda k: []),
        ):
            results = ttgir_to_kir.translate_p1(tmp / "dump.json", {})
        self.assertEqual(calls, [("s", None, True)])
        self.assertEqual(list(results), ["s", "s@cluster"])
        self.assertEqual(results["s"], {"p1": ["P1-2"], "geometry": None, **self.BARE})
        e = results["s@cluster"]
        self.assertIsInstance(e, ttgir_to_kir.KirError)
        self.assertEqual(ttgir_to_kir.failure_kind(e), "gap")
        self.assertIn("differ", str(e))
        self.assertEqual(sorted(p.name for p in tmp.glob("*.p1.json")), ["s.p1.json"])


class TtgirPassGeometries(unittest.TestCase):
    """The P0 translation checks the variants geometries decides, as P0-6 and P1-2 do: the
    all-preferred KIR also where the regular one does not translate (translate gives back
    the exception that stopped it, of its class)."""

    def test_all_preferred_kir_without_the_regular_one(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        rec = {"cluster": [2, 1, 1], "grid": [8, 1, 1],
               "attrs": {"preferred_cluster": [4, 1, 1]}}  # fmt: skip
        dump = {"specs": {"s": {}},
                "inputs": [{"name": "i", "launches": [{"spec": "s", "launch": rec}]}]}  # fmt: skip
        (tmp / "dump.json").write_text(json.dumps(dump))

        def build(where, d, key, case, geometry=None, p1=False):
            if geometry is None:
                raise SliceError("a construct of the regular geometry")
            return {"geometry": geometry, **TtgirP1Geometry.BARE}

        with (
            mock.patch.dict(os.environ, PTX_MODE),
            mock.patch.object(ttgir_to_kir, "build_ptx", build),
            mock.patch.object(ttgir_to_kir.kir_contract, "validate", lambda k: []),
        ):
            got = ttgir_to_kir.translate(tmp / "dump.json", {})
        self.assertEqual(list(got), ["s", "s@cluster4x1x1"])
        self.assertIsInstance(got["s"], SliceError)
        self.assertEqual(str(got["s"]), "a construct of the regular geometry")
        self.assertEqual(ttgir_to_kir.failure_kind(got["s"]), "gap")
        self.assertEqual(got["s@cluster4x1x1"]["geometry"], [4, 1, 1])


class TtgirReport(unittest.TestCase):
    """to_kir's report.json (Report), what lkv.py reads: one entry per variant in spec
    order (each specialization's regular variant first) with the inputs of its
    specialization and every pass's result; written before the first task with every
    result still to come an error, so a to_kir that stops leaves the results it logged;
    in the legacy translation, P1-2 not checked on its P0 variants (it has no P1 pass)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        rec = {"cluster": [2, 1, 1], "grid": [8, 1, 1],
               "attrs": {"preferred_cluster": [4, 1, 1]}}  # fmt: skip
        self.dump = self.tmp / "dump.json"
        self.dump.write_text(json.dumps({
            "kernel": "k", "specs": {"t": {"ttgir": "t.ttgir"}, "s": {"ttgir": "s.ttgir"}},
            "inputs": [{"name": "a", "launches": [{"spec": "s", "launch": rec},
                                                  {"spec": "t", "launch": {}}]},
                       {"name": "b", "launches": [{"spec": "s", "launch": rec}]},
                       {"name": "r", "host_error": "rejected", "launches": []}]}))  # fmt: skip
        for key in ("s", "t"):
            (self.tmp / f"{key}.ttgir").write_text("")
        (self.tmp / "case.json").write_text("{}")

    def to_kir(self, env, stop=None):
        """to_kir's main on the dump with stand-in translations (the P0-6 KIR of
        s@cluster4x1x1 refused, no bulk store in t); `stop`: the task at which to_kir
        stops. report.json."""

        def kir(where, dump, key, case, geometry=None, p1=False):
            if p1 and key == "t":
                return None
            return dict(TtgirP1Geometry.BARE, **({"p1": ["P1-2"]} if p1 else {}))

        def nb(where, dump, key, case, geometry=None):
            if geometry:
                raise ttgir_to_kir.KirError("a refusal")
            return TtgirP1Geometry.BARE

        real, calls = ttgir_to_kir._task, []

        def task(*args):
            calls.append(args)
            if len(calls) == stop:
                raise KeyboardInterrupt
            return real(*args)

        argv = [
            "to_kir",
            "--dump",
            str(self.dump),
            "--case",
            str(self.tmp / "case.json"),
        ]
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(ttgir_to_kir, "build_ptx", kir),
            mock.patch.object(
                ttgir_to_kir, "build", lambda t, d, key, case: kir(0, d, key, case)
            ),
            mock.patch.object(ttgir_to_kir, "build_nb", nb),
            mock.patch.object(ttgir_to_kir, "_task", task),
            mock.patch.object(ttgir_to_kir.kir_contract, "validate", lambda k: []),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(KeyboardInterrupt if stop else SystemExit),
        ):
            ttgir_to_kir.main()
        return json.loads((self.tmp / "report.json").read_text())

    def kir(self, name, suffix=".kir.json"):
        return str(self.tmp / f"{name}{suffix}")

    def test_ptx(self):
        got = self.to_kir({"LKV_FRONTEND": "ptx", "LKV_P1": "1"})
        p1 = {"status": "checked", "kir": self.kir("s", ".p1.json"), "specs": ["P1-2"]}
        self.assertEqual(got, [
            {"variant": "t", "kernel": "k", "inputs": ["a"], "kir": self.kir("t"),
             "nb": {"status": "checked", "kir": self.kir("t", ".nb.json")},
             "p1": {"status": "not_applicable", "error": "the kernel has no bulk async store",
                    "specs": ["P1-2"]}},
            {"variant": "s", "kernel": "k", "inputs": ["a", "b"], "kir": self.kir("s"),
             "nb": {"status": "checked", "kir": self.kir("s", ".nb.json")}, "p1": p1},
            {"variant": "s@cluster4x1x1", "kernel": "k", "inputs": ["a", "b"],
             "kir": self.kir("s@cluster4x1x1"),
             "nb": {"status": "refused", "error": "a refusal"},
             "p1": {**p1, "kir": self.kir("s@cluster4x1x1", ".p1.json")}},
        ])  # fmt: skip
        # LKV_P1=0: no P1 results
        got = self.to_kir({"LKV_FRONTEND": "ptx", "LKV_P1": "0"})
        self.assertEqual([("p1" in e) for e in got], [False, False, False])

    def test_stopped(self):
        # stopped at the 4th task, the first P0-6 one: the 3 P0 results stay, the rest
        # are errors
        got = self.to_kir({"LKV_FRONTEND": "ptx", "LKV_P1": "1"}, stop=4)
        self.assertEqual([e.get("kir") for e in got],
                         [self.kir("t"), self.kir("s"), self.kir("s@cluster4x1x1")])  # fmt: skip
        for e in got:
            for pas, what in (("nb", "P0-6 pass"), ("p1", "P1 pass")):
                self.assertEqual(e[pas]["status"], "error")
                self.assertEqual(e[pas]["error"],
                                 f"to_kir stopped before the {what} of this variant ended")  # fmt: skip

    def test_legacy(self):
        # one geometry per P0 variant, P1-2 not checked on them; the P0-6 pass checks both
        got = self.to_kir({"LKV_FRONTEND": "legacy", "LKV_P1": "1"})
        not_checked = {"status": "not_checked", "specs": ["P1-2"],
                       "error": "P1-2 not checked: the legacy TTGIR translation has no P1 pass"}  # fmt: skip
        self.assertEqual(
            [(e["variant"], e.get("kir"), e["nb"]["status"], e.get("p1")) for e in got],
            [("t", self.kir("t"), "checked", not_checked),
             ("s", self.kir("s"), "checked", not_checked),
             ("s@cluster4x1x1", None, "refused", None)],
        )  # fmt: skip
        self.assertNotIn("error", got[2])


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

    def test_preferred_cluster_refuses_p0_6(self):
        # a launch with a preferred cluster other than its cluster: the P0-6 KIR is
        # refused as the P0 one is (gap, the same reason), not checked in one geometry
        dump = json.loads((self.tmp / "dump.json").read_text())
        for inp in dump["inputs"]:
            for la in inp["launches"]:
                for r in la["launch_records"]:
                    r["attrs"]["preferred_cluster"] = [2, 1, 1]
        (self.tmp / "dump.json").write_text(json.dumps(dump))
        variant, _host = self.run_kir()
        self.assertNotIn("kir", variant)
        self.assertEqual(variant["kind"], "gap")
        self.assertIn(
            "a preferred cluster (2, 1, 1) beside the cluster dims", variant["error"]
        )
        self.assertEqual(
            variant["nb"], {"status": "refused", "error": variant["error"]}
        )
        self.assertEqual(list((self.tmp / "kir").glob("*.nb.json")), [])

    def test_ptx_parsed_once_per_pass(self):
        # the P0 translation and the P0-6 pass each parse the variant's PTX once
        # (_ptx_variant) and translate that module; the barrier pass runs once (its
        # metadata tensors are chosen before it); the P1 pass parses nothing here (no
        # tcgen05)
        real_parse, real_nb, texts, modules = parse.parse, nbar.translate, [], []

        def counted(text, *args, **kw):
            texts.append(text)
            return real_parse(text, *args, **kw)

        def spy(text, entry, records, pol, module=None):
            modules.append(module)
            return real_nb(text, entry, records, pol, module)

        with mock.patch.object(parse, "parse", counted), \
                mock.patch.object(nbar, "translate", spy):  # fmt: skip
            variant, _host = self.run_kir()
        self.assertIn("kir", variant)
        self.assertEqual(variant["nb"]["status"], "checked")
        self.assertEqual(len(texts), 2)
        self.assertEqual(len(modules), 1)
        self.assertIsNotNone(modules[0])

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

    def test_jobs(self):
        # The passes in forked processes (_Passes.start) write the same files as one after
        # another; a pass that raises there stops run as it would here.
        self.with_tmem("32x32b")  # a P1 KIR too
        got = {}
        for jobs in (1, 3):
            with mock.patch.dict(os.environ, PTX_MODE):
                cute_to_kir.run(SKILL / "tests/e2e/cases/probe_cute.json", self.tmp,
                                jobs=jobs)  # fmt: skip
            got[jobs] = {
                p.name: p.read_bytes() for p in (self.tmp / "kir").glob("*.json")
            }
        self.assertEqual(got[1], got[3])
        self.assertEqual(
            len([n for n in got[3] if n.endswith((".nb.json", ".p1.json"))]), 2
        )
        with mock.patch.dict(os.environ, PTX_MODE), \
                mock.patch.object(cute_to_kir, "_p0", side_effect=ValueError("boom")):  # fmt: skip
            with self.assertRaisesRegex(ValueError, "boom"):
                cute_to_kir.run(SKILL / "tests/e2e/cases/probe_cute.json", self.tmp)
            with self.assertRaisesRegex(
                RuntimeError, "(?s)the p0 pass of .*ValueError: boom"
            ):
                cute_to_kir.run(SKILL / "tests/e2e/cases/probe_cute.json", self.tmp,
                                jobs=3)  # fmt: skip

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
    """cute/to_kir.py reroll: LLVM's unrolled blocks back into a `for`."""

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

    def test_view_inside_a_tensor(self):
        # A computed pointer field left unnamed (no tensor starts at its address) that
        # points inside the one tensor it is computed from, in every record: named after
        # it with an element offset param, each record's own (_record_values). Inside
        # two of its tensors, or null in a record (no candidate), it stays unnamed.
        def records(*offs):
            return [{"params": [{"fields": [{"off": 0, "value": 0 if o is None else 4096 + o, "tensor": [
                {"tensor": "mB", "match": "inside" if o else "exact", "storage": [4096, 8192], "elem_off": o},
                {"tensor": "mB32", "match": "inside" if o else "exact", "storage": [4096, 8192], "elem_off": o // 4},
            ] if o is not None else None}]}], "jit_params": [{"from": ["mB"]}]} for o in offs]  # fmt: skip

        table = lambda: [{"ptx": "p0", "kind": "aggregate", "name": "param0", "size": 8,  # noqa: E731
                          "fields": {0: {"size": 8, "kind": "ptr", "name": "param0_0"}}}]  # fmt: skip
        tensors = dict.fromkeys(["mB", "mB32"])
        recs = records(384, 0, 1024)
        got = cute_to_kir._jit_named(table(), recs, tensors)
        self.assertEqual(got[0]["fields"][0], {"size": 8, "kind": "ptr", "name": "mB",
                                               "off_param": "param0_0.off"})  # fmt: skip
        self.assertEqual(
            [cute_to_kir._record_values(got, r, {"param0_0.off": 64}) for r in recs],
            [{"param0_0.off": 384}, {"param0_0.off": 0}, {"param0_0.off": 1024}],
        )
        for recs in (records(384, None), [{**r, "jit_params": [{"from": ["mB", "mB32"]}]}
                                          for r in records(384)]):  # fmt: skip
            self.assertEqual(cute_to_kir._jit_named(table(), recs, tensors), table())

    def test_stream_not_last(self):
        """A TVM-FFI launch passes nothing for its environment stream, wherever the jit
        function has it (a 2-CTA backward kernel: `..., stream, mCuSeqlensDKV,
        mGlobalScales`): the arguments after it bind to the parameters after it, as the
        jit_params agree, and a LaunchRecord candidate naming them shifted is an error."""

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

    def test_environment_stream_in_the_middle(self):
        # A launch passes no value for the TVM-FFI environment stream (host_patches marks
        # it env_stream). Unmarked, skipping it or the descriptor after it fits the
        # records alike (several choices); skipping mA shifts mA's value onto `opt`
        # (compiled as None, no leaves), which is no choice (it used to stop the search).
        params = [{"name": n, "constexpr": False, "leaves": [] if n == "opt" else [n]}
                  for n in ("mA", "opt", "stream", "mDesc", "mS")]  # fmt: skip
        t = {"kind": "tensor", "dtype": "int32", "shape": [3], "stride": [1],
             "storage": [4096, 12], "byte_offset": 0}  # fmt: skip
        args = [
            {**t, "storage": [1024, 12]},
            {"kind": "none"},
            t,
            {**t, "storage": [8192, 12]},
        ]
        record = {"params": [{"kind": "ptr", "value": 1024, "fields": []},
                             {"kind": "tensormap", "fields": []},
                             {"kind": "ptr", "value": 8192, "fields": []}],
                  "jit_params": [{"arg": "mA", "leaf": "mA"}, {"from": ["mDesc"]},
                                 {"arg": "mS", "leaf": "mS"}]}  # fmt: skip
        with self.assertRaisesRegex(cute_to_kir.Unsupported, "several choices"):
            cute_to_kir._launch_params({"params": params}, [("a", args)], [record])
        params[2]["env_stream"] = True
        launched = cute_to_kir._launch_params(
            {"params": params}, [("a", args)], [record]
        )
        self.assertEqual([p["name"] for p in launched], ["mA", "opt", "mDesc", "mS"])


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
        # Also where `head` names the segments of a tensor the role does not access.
        for per in ({}, {"head": ["y"]}, {"batch": ["x"]}):
            with self.subTest(per=per):
                f = self.facts(per)
                self.assertEqual(f.refusals, [])
                (w,) = [
                    e for r in f.roles for e in events(r["body"]) if e["ev"] == "work"
                ]
                self.assertEqual(w["fields"]["head"], {"e": "const", "w": 64, "v": -1})
                self.assertEqual(w["fields"]["batch"]["e"], "var")

    def test_refused_where_an_access_needs_its_segment(self):
        (refusal,) = self.facts({"head": ["x"]}).refusals
        self.assertEqual(refusal["kind"], "error")
        self.assertIn("no instruction holds its value", refusal["msg"])
        self.assertIn("the role accesses x", refusal["msg"])


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
        # unset or empty: ptx, the default of both families
        for to_kir, default in ((ttgir_to_kir, "ptx"), (cute_to_kir, "ptx")):
            for value, want in (("", default), ("legacy", "legacy"), ("ptx", "ptx")):
                with mock.patch.dict(os.environ, {"LKV_FRONTEND": value}):
                    self.assertEqual(to_kir.frontend(), want)
            with mock.patch.dict(os.environ):
                os.environ.pop("LKV_FRONTEND", None)
                self.assertEqual(to_kir.frontend(), default)
            with mock.patch.dict(os.environ, {"LKV_FRONTEND": "mlir"}):
                self.assertRaises(SystemExit, to_kir.frontend)


if __name__ == "__main__":
    unittest.main()
