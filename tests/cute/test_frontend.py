# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end tests of the CuTe frontend on the tiny kernels of kernels.py.

Each test writes a case file, runs the real host path (dump.run: compile on the CPU,
record the launches, lower to the analyzed stage) and translates it (to_kir.run, which
validates every KIR with frontend/kir.py). Needs the CuTe DSL: buck target :test_frontend
(no GPU), built with the DSL the op table was checked against (ops.DSL_VERSION). The TMA,
inline-PTX and strided-metadata tests also check the Lean verdicts of their KIR
(frontend/emit_lean.py, scripts/run_lean_check.sh) when the Lean toolchain is installed.
"""

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from ...frontend.cute import dump, ops, to_kir
from ...frontend.ttgir import host_env

HERE = Path(__file__).resolve().parent
SKILL = HERE.parents[1]


def run_case(
    name: str,
    entry: str,
    builder: str,
    inputs: list,
    kernels: dict | None = None,
    env: dict | None = None,
):
    """dump + to_kir for one case; returns (report, {kernel class: KIR}, dump.json)."""
    tmp = Path(tempfile.mkdtemp(prefix=f"lkv_cute_test_{name}_"))
    case = {
        "name": name,
        "source": {"package": "kernels.py", "src": str(HERE)},
        "env": env or {},
        "entry": f"kernels:{entry}",
        "builder": f"{HERE / 'kernels.py'}:{builder}",
        "num_sms": 2,
        "kernels": kernels or {},
        "inputs": inputs,
    }
    path = tmp / f"{name}.json"
    path.write_text(json.dumps(case))
    recorded = dump.run(path, tmp / "out")
    to_kir.run(path, tmp / "out")
    report = json.loads((tmp / "out" / "kir" / "report.json").read_text())
    kirs = {
        e["kernel"]: json.loads(Path(e["kir"]).read_text())
        for e in report
        if "kir" in e
    }
    return report, kirs, recorded


def lean_check(test, kir_path: str) -> tuple:
    """({(input, "P0-n"): "PASS" | "FAIL"}, the checker's output) from the Lean checker;
    skips the test when no Lean toolchain is installed."""
    lean_file = Path(kir_path).with_suffix(".lean")
    subprocess.run(
        [
            "python3",
            "-I",
            str(SKILL / "frontend/emit_lean.py"),
            kir_path,
            "-o",
            str(lean_file),
        ],
        check=True,
    )
    run = subprocess.run(
        [str(SKILL / "scripts/run_lean_check.sh"), str(lean_file)],
        capture_output=True,
        text=True,
    )
    if run.returncode == 127:
        test.skipTest(f"no Lean toolchain: {run.stderr.strip()[:200]}")
    verdicts = {
        (m[0], m[1]): m[2]
        for m in re.findall(r"^VERDICT (\S+) (P0-\d) (PASS|FAIL)$", run.stdout, re.M)
    }
    return verdicts, run.stdout


def lean_verdicts(test, kir_path: str) -> dict:
    return lean_check(test, kir_path)[0]


def events(stmts, out=None):
    """Events in program order, descending into if/for/while bodies."""
    out = [] if out is None else out
    for s in stmts:
        if s["op"] == "event":
            out.append(s)
        for k in ("then", "else", "body"):
            if k in s:
                events(s[k], out)
    return out


def loops(stmts, out=None):
    out = [] if out is None else out
    for s in stmts:
        if s["op"] in ("for", "while"):
            out.append(s)
        for k in ("then", "else", "body"):
            if k in s:
                loops(s[k], out)
    return out


class CuteFrontendTest(unittest.TestCase):
    def assert_valid(self, report):
        for e in report:  # a variant has a KIR file only if frontend/kir.py accepted it
            self.assertNotIn("error", e, e)
            self.assertIn("kir", e, e)

    def test_tma_producer_consumer(self):
        report, kirs, recorded = run_case("tma", "run_tma_pipeline", "args_tma_pipeline", [{"name": "rows100", "rows": 100}],
                                          {"host": {"intended_writes": {"mOut": "all"}}})  # fmt: skip
        self.assert_valid(report)
        # The dump records the DSL it loaded; it is the one the op table was checked against.
        self.assertEqual(recorded["dsl"]["version"], ops.DSL_VERSION)
        self.assertTrue(recorded["dsl"]["native"])
        (k,) = kirs.values()
        self.assertEqual(
            (k["target"], k["compiler"]),
            ("sm_100", {"name": "nvidia-cutlass-dsl", "version": ops.DSL_VERSION}),
        )
        roles = {r["name"]: r for r in k["roles"]}
        self.assertEqual(sorted(roles), ["warps0_0", "warps1_1"])
        prod = events(roles["warps0_0"]["body"])
        self.assertEqual(
            [e["ev"] for e in prod],
            ["bar_init", "expect_tx", "arrive", "tma_load", "gload"],
        )
        # The TMA load's footprint on the tensor behind the descriptor: the box, masked to
        # the descriptor's extents.
        self.assertEqual(
            (prod[4]["tensor"], prod[4]["offsets"]["shape"], prod[4]["mask"]["t"]),
            ("mA", [64, 64], "bin"),
        )
        self.assertEqual(roles["warps0_0"]["threads_per_issue"], 1)
        self.assertNotIn("n", prod[2])  # one elected lane arrives once
        self.assertEqual(prod[1]["bytes"], {"e": "const", "w": 32, "v": 64 * 64 * 2})
        desc = next(t for t in k["tensors"] if t["name"] == prod[3]["desc"])
        self.assertEqual(
            (desc["kind"], desc["box"], desc["elem_bytes"]), ("desc", [64, 64], 2)
        )
        self.assertEqual(k["barriers"], [{"name": prod[0]["bar"], "count": {"e": "const", "w": 32, "v": 1},
                                          "num": {"e": "const", "w": 32, "v": 1}}])  # fmt: skip
        cons = events(roles["warps1_1"]["body"])
        self.assertEqual([e["ev"] for e in cons], ["wait", "gstore"])
        self.assertEqual(cons[0]["parity"], {"e": "const", "w": 32, "v": 0})
        # The per-row stores run in a loop over the 32 lanes of warp 1, under the row mask.
        thread_loop = [
            s
            for s in loops(roles["warps1_1"]["body"])
            if s["ub"] == {"e": "const", "w": 32, "v": 32}
        ]
        self.assertTrue(thread_loop)
        self.assertEqual(k["case"]["inputs"][0]["params"]["mOut_shape0"], 100)
        verdicts = lean_verdicts(self, report[0]["kir"])
        self.assertEqual(
            verdicts, {("rows100", f"P0-{i}"): "PASS" for i in range(1, 6)}
        )

    def test_unelected_producer_arrives_on_every_lane(self):
        report, kirs, _ = run_case("tma_no_elect", "run_tma_pipeline", "args_tma_pipeline",
                                   [{"name": "rows64", "rows": 64, "elect_producer": False}],
                                   {"host": {"intended_writes": {"mOut": "all"}}})  # fmt: skip
        self.assert_valid(report)
        (k,) = kirs.values()
        role = next(r for r in k["roles"] if r["name"] == "warps0_0")
        evs = events(role["body"])
        arrive = next(e for e in evs if e["ev"] == "arrive")
        self.assertEqual(
            arrive.get("n", 1) * role["threads_per_issue"], 32
        )  # 32 arrivals per phase
        # expect_tx runs once per lane: a loop over 32 issuers around one event.
        issuer_loop = next(
            s
            for s in loops(role["body"])
            if events(s["body"]) and events(s["body"])[0]["ev"] == "expect_tx"
        )
        self.assertEqual(issuer_loop["ub"], {"e": "const", "w": 32, "v": 32})
        # The TMA copy is still issued once: its lowering elects one lane per warp.
        self.assertEqual(sum(e["ev"] == "tma_load" for e in evs), 1)
        # Lean: 32 arrivals complete the count-1 barrier 32 times for one wait (P0-2) and
        # the expected bytes are 32 x the bytes the copy lands (P0-4).
        verdicts = lean_verdicts(self, report[0]["kir"])
        want = {f"P0-{i}": "PASS" for i in range(1, 6)} | {
            "P0-2": "FAIL",
            "P0-4": "FAIL",
        }
        self.assertEqual(verdicts, {("rows64", p): v for p, v in want.items()})

    def test_tma_load_through_an_oversized_view_fails_bounds(self):
        # The descriptor claims 64 rows past the 100-row tensor; the second tile's load reads
        # rows 100..127 of it.
        report, kirs, _ = run_case("tma_oversized", "run_tma_pipeline", "args_tma_pipeline",
                                   [{"name": "rows100", "rows": 100, "extra_rows": 64}],
                                   {"host": {"intended_writes": {"mOut": "all"}}})  # fmt: skip
        self.assert_valid(report)
        (k,) = kirs.values()
        desc = next(t for t in k["tensors"] if t["kind"] == "desc")
        self.assertEqual(desc["box"], [64, 64])
        verdicts = lean_verdicts(self, report[0]["kir"])
        want = {f"P0-{i}": "PASS" for i in range(1, 6)} | {"P0-1": "FAIL"}
        self.assertEqual(verdicts, {("rows100", p): v for p, v in want.items()})

    def test_tma_load_from_part_of_a_warp_is_refused(self):
        # The lowering's full-mask elect.sync with 16 active lanes.
        report, kirs, _ = run_case("tma_half_warp", "run_tma_pipeline", "args_tma_pipeline",
                                   [{"name": "rows64", "rows": 64, "copy_lanes": 16}],
                                   {"host": {"intended_writes": {"mOut": "all"}}})  # fmt: skip
        self.assertEqual(kirs, {})
        self.assertIn(
            "TMA load (lowered to elect.sync) with only part of warp(s) [0] active",
            report[0]["error"],
        )

    def test_strided_metadata_is_recorded_by_offset(self):
        # `load` indices are element offsets from the data pointer.
        for strided, contents in (
            (True, [0, 99, 10, 99, 30, 99, 60]),
            (False, [0, 10, 30, 60]),
        ):
            with self.subTest(strided=strided):
                report, kirs, _ = run_case(f"first_segment_{strided}", "run_first_segment", "args_first_segment",
                                           [{"name": "seg0", "strided": strided}],
                                           {"_first_segment_host": {"intended_writes": {"mY": [[0, 10]]}}})  # fmt: skip
                self.assert_valid(report)
                (k,) = kirs.values()
                self.assertEqual(k["case"]["inputs"][0]["metadata"]["mCu"], contents)
                verdicts = lean_verdicts(self, report[0]["kir"])
                self.assertEqual(
                    verdicts, {("seg0", f"P0-{i}"): "PASS" for i in range(1, 6)}
                )

    def test_host_read_of_a_skipped_write_is_refused(self):
        # The host sizes the fill grid with cnt[0].item(), which the skipped count launch
        # writes; src is only read by it. Also through the shape of repeat_interleave(), and
        # when the count launch passes CuTe runtime tensors (they are the torch tensors they
        # wrap).
        notes = {
            "_count_host": {"intended_writes": {"mCnt": "all"}},
            "_fill_host": {"intended_writes": {"mOut": "all"}},
        }
        for how, op, cute_args in (("item", "TensorBase.item", False), ("repeat_interleave", "aten.repeat_interleave.Tensor", False),
                                   ("item", "TensorBase.item", True)):  # fmt: skip
            with self.subTest(how=how, cute_args=cute_args):
                inp = {"name": "n4", "how": how, "cute_args": cute_args}
                report, kirs, recorded = run_case(f"host_reads_cnt_{how}_{cute_args}", "run_host_reads", "args_host_reads",
                                                  [{**inp, "read": "cnt"}], notes)  # fmt: skip
                fill = next(e for e in report if e["kernel"] == "_fill_host")
                self.assertRegex(
                    fill["error"],
                    rf"input n4: before launch 1 the host reads data \({op} at "
                    r"kernels.py:\d+\) that launch 0 \(kernels._count_host\) may write",
                )
                self.assertIn(
                    "_count_host", kirs
                )  # translated: its runtime-tensor arguments are tensors
                self.assertEqual(recorded["inputs"][0]["host_reads"][0]["after"], 1)
                report, kirs, _ = run_case(f"host_reads_src_{how}_{cute_args}", "run_host_reads", "args_host_reads",
                                           [{**inp, "read": "src"}], notes)  # fmt: skip
                self.assert_valid(report)
                self.assertEqual(
                    kirs["_fill_host"]["case"]["inputs"][0]["params"]["num_tiles"], 4
                )

    def test_data_cached_from_an_earlier_input_is_checked(self):
        # The inputs share the process; the second input (its own fill variant: f16) sizes its
        # grid with a count a skipped launch of the first input wrote (refused), or one the
        # host computed (accepted).
        notes = {
            "_count_host": {"intended_writes": {"mCnt": "all"}},
            "_fill_host": {"intended_writes": {"mOut": "all"}},
        }
        for cache in ("kernel", "host"):
            with self.subTest(cache=cache):
                report, kirs, _ = run_case(f"cached_{cache}", "run_cached", "args_cached",
                                           [{"name": "first", "dtype": "float32", "cache": cache},
                                            {"name": "second", "dtype": "float16", "cache": cache}], notes)  # fmt: skip
                fills = {
                    e["inputs"][0]: e for e in report if e["kernel"] == "_fill_host"
                }
                if cache == "host":
                    self.assert_valid(report)
                    continue
                self.assertRegex(
                    fills["second"]["error"],
                    r"input second: before launch 2 the host reads data "
                    r"\(TensorBase.item at kernels.py:\d+\) that launch 0 "
                    r"\(kernels._count_host\) may write",
                )

    def test_writes_through_an_address_argument_are_tracked(self):
        # An untranslated kernel (unknown write set) gets the address of the offsets in an
        # int64 pointer table, then they steer the first-segment kernel.
        notes = {"_first_segment_host": {"intended_writes": {"mY": [[0, 10]]}}}
        report, _, recorded = run_case(
            "ptr_table_cu",
            "run_ptr_table",
            "args_ptr_table",
            [{"name": "t", "target": "cu"}],
            notes,
        )
        segment = next(e for e in report if e["kernel"] == "_first_segment_host")
        self.assertEqual(
            segment["error"],
            "metadata tensor mCu of input t may hold data that launch 0 "
            "(kernels._ptr_table_host) writes (directly or through host computation), "
            "but the CPU host skipped that launch",
        )
        cu = recorded["inputs"][0]["launches"][1]["args"][1]["storage"][0]
        self.assertEqual(recorded["inputs"][0]["launches"][0]["hinted"], [cu])
        report, kirs, _ = run_case(
            "ptr_table_other",
            "run_ptr_table",
            "args_ptr_table",
            [{"name": "t", "target": "other"}],
            notes,
        )
        self.assertIn("_first_segment_host", kirs)

    def test_failures_outside_the_host_sources_fail_the_dump(self):
        # Only an exception the exported sources raise (or a failed torch check) is the host
        # refusing the input; a direct @cute.jit call (no cute.compile) or a CUDA call on the
        # CPU-only host is a failure of the dump, not a rejected input.
        notes = {"_fill_host": {"intended_writes": {"mOut": "all"}}}
        with self.assertRaises(Exception) as direct:  # noqa: B017
            run_case("direct", "run_direct", "args_out", [{"name": "n128"}], notes)
        # host_patches is not imported here: the DSL must be imported after the dump pins the environment.
        self.assertEqual(
            type(direct.exception).__name__, "_Captured"
        )  # raised by the patched compile_and_cache (P5)
        with self.assertRaisesRegex(RuntimeError, "NVIDIA driver"):
            run_case("sync", "run_sync", "args_out", [{"name": "n128"}], notes)

    def test_launch_argument_over_untracked_memory_fails_the_dump(self):
        # A runtime tensor is tracked as the torch tensor it wraps; one over other memory
        # (numpy) cannot be.
        with self.assertRaisesRegex(
            RuntimeError,
            "a launch argument is a CuTe runtime tensor that wraps no torch tensor",
        ):
            run_case(
                "numpy_arg",
                "run_numpy_arg",
                "args_out",
                [{"name": "n128"}],
                {"_fill_host": {"intended_writes": {"mOut": "all"}}},
            )

    def test_elision_flags_reach_the_kir(self):
        # Cases name the variant switches that elide masks; the KIR case carries them, and
        # P0-5 takes the P0-1 and P0-3 checks of the inputs that set one.
        for flags, p01, p05 in (([], "FAIL", "PASS"), (["full_tiles"], "PASS", "FAIL")):
            with self.subTest(flags=flags):
                notes = {
                    "_FillTiles": {
                        "intended_writes": {"mOut": "all"},
                        "elision_flags": flags,
                    }
                }
                report, kirs, _ = run_case(f"fill_tiles_{bool(flags)}", "run_fill_tiles", "args_fill_tiles",
                                           [{"name": "partial", "n": 100}], notes)  # fmt: skip
                self.assert_valid(report)
                case = kirs["_FillTiles"]["case"]
                self.assertEqual(
                    (case.get("elision_flags", []), case["inputs"][0]["flags"]),
                    (flags, {"full_tiles": True}),
                )
                verdicts = lean_verdicts(self, report[0]["kir"])
                self.assertEqual(
                    (verdicts[("partial", "P0-1")], verdicts[("partial", "P0-5")]),
                    (p01, p05),
                )
        notes = {
            "_FillTiles": {
                "intended_writes": {"mOut": "all"},
                "elision_flags": ["full_tile"],
            }
        }
        report, _, _ = run_case(
            "fill_tiles_typo",
            "run_fill_tiles",
            "args_fill_tiles",
            [{"name": "partial", "n": 100}],
            notes,
        )
        self.assertIn(
            "_FillTiles: elision_flags ['full_tile'] are not flags of the variant (['full_tiles'])",
            report[0]["error"],
        )

    def test_environment_is_pinned_to_the_case(self):
        # Shell values are ignored and recorded; the case's env applies.
        saved = dict(os.environ)
        try:
            for env, tiles in (({}, 2), ({"LKV_CUTE_TEST_TILES": "4"}, 4)):
                with self.subTest(env=env):
                    os.environ.update(
                        {"LKV_CUTE_TEST_TILES": "1", "CUTE_DSL_LKV_TEST": "1"}
                    )  # the shell's
                    report, kirs, recorded = run_case(f"env_{tiles}", "run_env_fill", "args_env_fill", [{"name": "i"}],
                                                      {"_fill_host": {"intended_writes": {"mOut": "all"}}}, env)  # fmt: skip
                    self.assert_valid(report)
                    self.assertEqual(
                        kirs["_fill_host"]["case"]["inputs"][0]["params"]["num_tiles"],
                        tiles,
                    )
                    rec = recorded["env"]
                    self.assertEqual(rec["case"], env)
                    self.assertEqual(rec["unset"].get("CUTE_DSL_LKV_TEST"), "1")
                    self.assertEqual("LKV_CUTE_TEST_TILES" in rec["unset"], not env)
                    self.assertEqual(rec["effective"]["CUTE_DSL_ARCH"], "sm_100a")
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_environment_reads_the_scan_cannot_see_are_refused(self):
        # The case builder reads a variable through an alias of os.environ.
        with self.assertRaisesRegex(
            host_env.EnvError,
            r"kernels.py:\d+ reads environment variable LKV_CUTE_TEST_ALIAS",
        ):
            run_case("env_alias", "run_env_fill", "args_env_alias", [{"name": "i"}])

    def test_metadata_bounded_copies(self):
        report, kirs, _ = run_case(
            "segment_copy",
            "run_segment_copy",
            "args_segment_copy",
            [{"name": "mixed", "lens": [200, 0, 37]}],
        )
        self.assert_valid(report)
        (k,) = kirs.values()
        cu = next(t for t in k["tensors"] if t["name"] == "mCu")
        self.assertTrue(cu["metadata"])
        self.assertEqual(k["case"]["inputs"][0]["metadata"]["mCu"], [0, 200, 200, 237])
        self.assertEqual(
            {e["ev"] for e in events(k["roles"][0]["body"])}, {"gload", "gstore"}
        )
        # The trip count ceil_div(n, 128) is read from metadata: a KIR for over a let.
        self.assertTrue(
            any(
                s["op"] == "for" and s["ub"]["e"] == "var"
                for s in loops(k["roles"][0]["body"])
            )
        )

    def test_input_the_host_rejects_is_reported(self):
        report, kirs, recorded = run_case("host_rejects", "run_segment_copy_no_empty", "args_segment_copy",
                                          [{"name": "has_empty", "lens": [10, 0]}, {"name": "ok", "lens": [10, 5]}])  # fmt: skip
        self.assertEqual(
            ["host_error" in i for i in recorded["inputs"]], [True, False]
        )  # the next input still runs
        (host,) = [e for e in report if e["kernel"] == "(host)"]
        self.assertEqual(host["inputs"], ["has_empty"])
        self.assertRegex(
            host["error"],
            r"the host rejected input has_empty: ValueError: empty segments are not supported at kernels\.py:\d+",
        )
        (k,) = kirs.values()
        self.assertEqual([i["name"] for i in k["case"]["inputs"]], ["ok"])
        # An input for which the host launches nothing is recorded the same way.
        report, kirs, recorded = run_case("host_skips", "run_segment_copy_skip_empty", "args_segment_copy",
                                          [{"name": "no_rows", "lens": [0, 0]}, {"name": "ok", "lens": [10, 5]}])  # fmt: skip
        self.assertEqual(
            [i.get("host_error") for i in recorded["inputs"]], ["no launch", None]
        )
        (host,) = [e for e in report if e["kernel"] == "(host)"]
        self.assertEqual(
            (host["inputs"], host["error"]),
            (["no_rows"], "the host rejected input no_rows: no launch"),
        )
        (k,) = kirs.values()
        self.assertEqual([i["name"] for i in k["case"]["inputs"]], ["ok"])

    def test_warp_cooperative_search(self):
        report, kirs, _ = run_case(
            "warp_search",
            "run_warp_search",
            "args_warp_search",
            [{"name": "jagged", "lens": [40, 0, 70, 5]}],
        )
        self.assert_valid(report)
        (k,) = kirs.values()
        (role,) = k[
            "roles"
        ]  # `tidx < 32 and row < end` is one per-thread mask: no warp split
        self.assertEqual((role["name"], role["warps"]), ("warps0_1", 2))
        # The per-lane cu_seqlens loads feed shuffles and a ballot: they are bound once per
        # lane (a lane table), not copied into every use.
        per_lane = [
            s
            for s in role["body"]
            if s["op"] == "let" and '"tensor": "mCu"' in json.dumps(s["expr"])
        ]
        self.assertGreaterEqual(len(per_lane), 32)
        self.assertEqual({e["ev"] for e in events(role["body"])}, {"gload", "gstore"})
        store_loop = next(
            s
            for s in loops(role["body"])
            if any(e["ev"] == "gstore" for e in events(s["body"]))
        )
        self.assertEqual(store_loop["ub"], {"e": "const", "w": 32, "v": 64})
        self.assertEqual(store_loop["body"][-1]["op"], "if")  # the per-thread row mask
        self.assertEqual(k["case"]["inputs"][0]["params"]["num_tiles"], 6)

    def test_seeded_data_dependency_is_rejected(self):
        report, kirs, _ = run_case(
            "data_branch", "run_data_branch", "args_data_branch", [{"name": "one"}]
        )
        self.assertEqual(kirs, {})
        (entry,) = report
        self.assertIn("slice soundness", entry["error"])
        self.assertIn("arith.cmpf", entry["error"])
        self.assertIn("kernels.py:", entry["error"])

    def test_warp_wide_tma_store_is_issued_once_per_distinct_operands(self):
        # ptxas issues a warp's cp.async.bulk.tensor store once per distinct (coordinates,
        # shared-memory source) among its lanes (a waterfall loop; SASS checked, see
        # ir_ops_cute.md "Issuing threads"). Warp-uniform operands: one store, as with
        # elect_one. Lanes 16-31 storing the CTA's next tile: two stores, every element
        # written once. Lanes 16-31 storing a second buffer to the same tile: two stores
        # of the same elements, each written twice.
        notes = {"host": {"intended_writes": {"mC": "all"}}}
        for mode, rows, p03 in (("warp", 128, "PASS"), ("elect", 128, "PASS"), ("lane_coords", 256, "PASS"),
                                ("lane_smem", 128, "FAIL")):  # fmt: skip
            with self.subTest(mode=mode):
                report, kirs, _ = run_case(f"tma_store_{mode}", "run_tma_store", "args_tma_store",
                                           [{"name": "i", "rows": rows, "mode": mode}], notes)  # fmt: skip
                self.assert_valid(report)
                (k,) = kirs.values()
                (role,) = [r for r in k["roles"] if events(r["body"])]
                (store,) = events(role["body"])
                self.assertEqual(store["ev"], "gstore")
                lane_loops = [
                    s
                    for s in loops(role["body"])
                    if s["ub"] == {"e": "const", "w": 32, "v": 32}
                ]
                if mode.startswith(
                    "lane"
                ):  # a loop over the lanes, issuing on the first lane of each tuple
                    (loop,) = lane_loops
                    self.assertEqual(loop["body"][-1]["op"], "if")
                else:  # one event, no loop over the lanes
                    self.assertEqual(lane_loops, [])
                verdicts, out = lean_check(self, report[0]["kir"])
                self.assertEqual(verdicts[("i", "P0-3")], p03)
                if p03 == "FAIL":
                    self.assertRegex(
                        out,
                        r"tensor mC: 8192 element\(s\) written more than once .* 2 times",
                    )

    def test_warp_wide_tma_reduce_is_issued_once_per_warp(self):
        # The DSL lowers cp.reduce.async.bulk.tensor behind a full-mask elect.sync, so a
        # warp issues it once, as under elect_one, even with every lane executing the copy
        # (PTX and SASS checked, see ir_ops_cute.md "Issuing threads").
        notes = {
            "host": {
                "intended_writes": {"mC": "unchecked"},
                "why": "written by reduce-adds only",
            }
        }
        report, kirs, _ = run_case(
            "tma_reduce",
            "run_tma_store",
            "args_tma_store",
            [{"name": "i", "rows": 128, "mode": "reduce"}],
            notes,
        )
        self.assert_valid(report)
        (k,) = kirs.values()
        (role,) = [r for r in k["roles"] if events(r["body"])]
        (red,) = events(role["body"])
        self.assertEqual(red["ev"], "gatomic")
        lane_loops = [
            s
            for s in loops(role["body"])
            if s["ub"] == {"e": "const", "w": 32, "v": 32}
        ]
        self.assertEqual(lane_loops, [])  # one event, no loop over the lanes

    def test_tma_reduce_needs_a_full_warp_and_lane_uniform_operands(self):
        # The lowering's full-mask elect.sync reached by 16 lanes, or by the one lane of an
        # enclosing elect_one; and coordinates that depend on which lane it elects.
        notes = {
            "host": {
                "intended_writes": {"mC": "unchecked"},
                "why": "written by reduce-adds only",
            }
        }
        partial = (
            "TMA reduce (lowered to elect.sync) with only part of warp(s) [0] active"
        )
        for mode, rows, error in (
            ("reduce_half", 128, partial),
            ("reduce_elect", 128, partial),
            ("reduce_lane_coords", 256, "an operand under elect.sync depends on which lane is elected"),
        ):  # fmt: skip
            with self.subTest(mode=mode):
                report, kirs, _ = run_case(f"tma_{mode}", "run_tma_store", "args_tma_store",
                                           [{"name": "i", "rows": rows, "mode": mode}], notes)  # fmt: skip
                self.assertEqual(kirs, {})
                self.assertIn(error, report[0]["error"])

    def test_inline_ptx_global_store_is_a_gstore(self):
        report, kirs, _ = run_case("sf_store", "run_sf_store", "args_sf_store", [{"name": "rows200", "rows": 200}],
                                   {"host": {"intended_writes": {"mSF": "all"}}})  # fmt: skip
        self.assert_valid(report)
        (k,) = kirs.values()
        (store,) = events(k["roles"][0]["body"])
        self.assertEqual(
            (store["ev"], store["tensor"], store["offsets"]["shape"]),
            ("gstore", "mSF", [2]),
        )
        self.assertIn("kernels.py:", store["loc"])
        verdicts = lean_verdicts(self, report[0]["kir"])
        self.assertEqual(
            verdicts, {("rows200", f"P0-{i}"): "PASS" for i in range(1, 6)}
        )

    def test_inline_ptx_global_reduction_is_refused(self):
        report, kirs, _ = run_case(
            "sf_red",
            "run_sf_store",
            "args_sf_store",
            [{"name": "rows64", "rows": 64, "use_red": True}],
        )
        self.assertEqual(kirs, {})
        self.assertIn("llvm.inline_asm at kernels.py:", report[0]["error"])
        self.assertIn("red.global.add.u32", report[0]["error"])

    def test_metadata_written_by_the_kernel_is_refused(self):
        report, kirs, _ = run_case(
            "cu_self_write",
            "run_cu_self_write",
            "args_cu_self_write",
            [{"name": "two", "lens": [100, 30]}],
        )
        self.assertEqual(kirs, {})
        self.assertIn("writes its metadata tensors ['mCu']", report[0]["error"])

    def test_cluster_of_two_ctas(self):
        # Cluster dims from the launch, the cluster rank register, the cluster barrier and a
        # remote mbarrier arrive (mapa) reach the KIR; rank 1 arriving on its own barrier
        # instead leaves rank 0's wait hanging (P0-2).
        inputs = [
            {"name": "ok", "rows": 64},
            {"name": "local", "rows": 64, "remote": False},
        ]
        for inp in inputs:
            report, kirs, _ = run_case("cluster", "run_cluster_ready", "args_cluster_ready", [inp],
                                       {"host": {"intended_writes": {"mOut": "all"}}})  # fmt: skip
            self.assert_valid(report)
            (k,) = kirs.values()
            self.assertEqual(k["cluster"], [2, 1, 1])
            self.assertIn({"name": "cluster_barrier", "count": {"e": "const", "w": 32, "v": 128},
                           "num": {"e": "const", "w": 32, "v": 1}}, k["barriers"])  # fmt: skip
            remote = [e for r in k["roles"] for e in events(r["body"])
                      if e["ev"] == "arrive" and e["bar"] != "cluster_barrier" and "cta" in e]  # fmt: skip
            self.assertEqual(len(remote), 1 if inp.get("remote", True) else 0)
            verdicts = lean_verdicts(self, report[0]["kir"])
            want = "FAIL" if inp["name"] == "local" else "PASS"
            self.assertEqual(verdicts[(inp["name"], "P0-2")], want, verdicts)
            self.assertEqual(verdicts[(inp["name"], "P0-4")], "PASS", verdicts)


if __name__ == "__main__":
    unittest.main()
