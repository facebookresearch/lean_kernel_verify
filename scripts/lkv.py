# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""lkv: lean-kernel-verify end to end, one command per case.

    python3 -I lkv.py run CASE.json [--src DIR] [--out DIR] [--inputs a,b]
                                    [--target sm_100|sm_103] [--jobs N] [--no-p1] [--p1-strict]
    python3 -I lkv.py check KIR.json... --out DIR [--src DIR] [--jobs N] [--p1-strict]
    python3 -I lkv.py export REV OUT_DIR PATH...

run    dump with the case's frontend (a buck2 binary: CuTe DSL if the case has a
       "builder", else TTGIR for TLX/Triton; only TTGIR compiles for sm_103) -> to_kir ->
       frontend/kir.py -> frontend/emit_lean.py -> scripts/run_lean_check.sh, then
       <out>/report.json and <out>/report.md. Default out:
       /tmp/lkv_runs/<case dir name>/<case name>. The case's `env` reaches the frontend
       with the case file; the frontend pins the environment to it before importing the
       compiler and the kernel (frontend/ttgir/host_env.py) and the report repeats its
       record (dump.json "env": the case's env, the shell values it ignored, the
       effective values, Triton's cache-invalidating variables).
check  the Layer-2 half of `run` on existing KIR files (e.g. tests/lean fixtures).
export write fbsource PATHs at REV under OUT_DIR with read-only `sl files`/`sl cat`: a
       file with the first-party modules it imports at OUT_DIR/<path below fbcode/>
       (TLX/Triton cases), a directory as a package at OUT_DIR/<basename>/ (CuTe cases),
       or with --package-root R at OUT_DIR/<path below R>/ (a package that imports
       itself by its full path).
       OUT_DIR/LKV_EXPORT.json records the revision; `run --src OUT_DIR` reports it.

P1-1 (TMEM reuse ordered, CuTe only): the CuTe frontend also writes a P1 KIR per variant
(`<kernel>__<digest>.p1.json`); run checks it like a P0 KIR (unless --no-p1) and reports
its verdicts in a section of their own and in report.json "variants_p1" / "totals_p1" /
"p1_not_checked" (refused, failed or not applicable P1 translations; TTGIR variants are
"not checked"). P1-1's FAIL and UNDECIDED verdicts count toward the outcome only with
--p1-strict; its ERRORs always do (they are failures of LKV).

A verdict is per (KIR variant, input, spec), and it has one of four outcomes:
  PASS       the theorem `KV.Holds` was proved by native_decide (a variant gets verdicts
             only if Lean's sole errors are refuted theorems and its exit status says the
             same);
  FAIL       a violation certain in the extracted model: the theorem was refuted with a
             certain counterexample (each IR loc resolved to the line of the exported
             source), or the input's evaluation stopped at a certain error of the kernel
             (Lean's `evaluation error (kernel error)`: a failed device assert, a barrier
             index out of range, a loop that never ends, a negative grid);
  ERROR      LKV, the case or the environment failed and must be fixed: an INCONSISTENT
             verdict (Lean's finder and decision procedure disagree: a Layer 2 bug), an
             `evaluation error (unexpected)` (e.g. a `while` over `max_iters`, an IR
             inconsistency), or a variant that failed to translate or check (a frontend
             error, invalid KIR, a Lean error or timeout: every spec of each of its inputs);
  UNDECIDED  not decided where LKV has a known limit, and not an error: the theorem was
             refuted without a certain counterexample (Lean prints
             `UNDECIDED <input> <spec>`: an `evaluation error (model limit)` such as a
             division by zero, a role trace stopped at an out-of-bounds metadata load
             before anything certain was found, or P1-1 pairs that only the lockstep
             contents or named barriers used with two counts might order), or the variant
             is a construct the frontend refuses by design (every spec of its inputs).
Notes are findings that are not certain violations (references/specs.md); they are listed
per spec and source line and never change a verdict. Verdicts cover only the case's
inputs.

Inputs the case's host code rejects are listed in the report, and the other inputs still
run (both frontends): the exported sources raised for the input (a validation `raise` or
`assert`, or a failed torch check such as `torch._assert_async`; the report gives the
exception and its source line, and launches made before it are still checked), or the host
made no launch (TTGIR: of the case's `kernel`). Any other exception of the host call fails
the frontend.

Outcome of a run (report.md "Result", report.json "result") and exit status, from the P0
verdicts, P1-1's ERRORs, and P1-1's FAIL and UNDECIDED with --p1-strict:
  1 FAIL       some verdict FAILs: a certain violation, whatever else failed;
  2 ERROR      no FAIL, but something failed: an ERROR verdict, a variant that failed to
               translate or check, a pipeline error, or no KIR at all;
  0 UNDECIDED  no FAIL and no ERROR, but something is not decided by a known limit (also a
               case whose every variant the frontend refuses by design);
  0 PASS       every spec is proved on every input of every variant.
Rejected inputs count as none of these. lkv.py's own errors (usage, export, an exception)
also exit 2, so exit status 1 always means a certain violation and 0 means nothing needs a
fix. report.json's keys and counts: references/specs.md, "report.json".
"""

import argparse
import concurrent.futures
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True  # keep the skill tree free of __pycache__
SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "frontend"))
import emit_lean  # noqa: E402
import kir  # noqa: E402

# The fbcode directory of the checkout holding the skill: `run` builds the frontends there
# with buck2. None for a copy outside a checkout, where only `check` works.
FBCODE = next((p for p in SKILL.parents if p.name == "fbcode"), None)
TARGETS = f"fbcode//{SKILL.relative_to(FBCODE)}" if FBCODE else None
SPECS = {
    "bounds": "P0-1",
    "tripCount": "P0-2",
    "coverage": "P0-3",
    "barrier": "P0-4",
    "maskElision": "P0-5",
    "tmemReuse": "P1-1",
}
EXPORT_MARK = "LKV_EXPORT.json"
TITLES = {
    "P0-1": "global accesses in bounds",
    "P0-2": "role trip-count agreement",
    "P0-3": "output coverage exactly once",
    "P0-4": "mbarrier protocol",
    "P0-5": "mask-elision soundness",
    "P1-1": "TMEM reuse ordered",
}
LABELS = ["P0-1", "P0-2", "P0-3", "P0-4", "P0-5"]
# P1 specs: decided on the P1 KIR (`<kernel>__<digest>.p1.json`, CuTe only) and reported
# in their own section; they never change the exit status unless --p1-strict.
LABELS_P1 = ["P1-1"]
# Outcomes of a verdict and of a run (module doc), and the run's exit status.
OUTCOMES = ("PASS", "FAIL", "ERROR", "UNDECIDED")
EXIT_STATUS = {"PASS": 0, "UNDECIDED": 0, "FAIL": 1, "ERROR": 2}


def log(msg):
    print(f"[lkv] {msg}", file=sys.stderr, flush=True)


def die(msg):
    """Stop without verdicts: exit status 2 (ERROR, a failure to fix), never 1 (a FAIL)."""
    print(msg, file=sys.stderr, flush=True)
    sys.exit(EXIT_STATUS["ERROR"])


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------- export


def export(rev, out, paths, package_root=None):
    root = Path(
        subprocess.run(
            ["sl", "root"], cwd=SKILL, capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    node = subprocess.run(
        ["sl", "log", "-r", rev, "-l", "1", "-T", "{node|short}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    mark = out / EXPORT_MARK
    done = (
        json.loads(mark.read_text()) if mark.is_file() else {"rev": node, "paths": []}
    )
    if done["rev"] != node:
        die(f"export: {out} holds an export of {done['rev']}, not {node}")
    export_src = load_module("lkv_export_src", SKILL / "frontend/ttgir/export_src.py")
    cute_main = load_module("lkv_cute_main", SKILL / "frontend/cute/main.py")
    os.chdir(root)  # export_src runs `sl cat` in the current directory
    for path in (p.strip("/") for p in paths):
        listing = subprocess.run(
            ["sl", "files", "-r", rev, "--include", f"path:{path}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        if any(f.startswith(path + "/") for f in listing):
            dest = out
            if package_root:  # keep the package path below the root (absolute imports)
                rel = Path(path).relative_to(package_root.strip("/"))
                dest = out / rel.parent
                dest.mkdir(parents=True, exist_ok=True)
            cute_main.export(rev, path, dest, root)
        elif path in listing:
            for rel in export_src.export(rev, out, [path]):
                print(out / rel)
        else:
            die(f"export: {path} does not exist at {rev}")
        done["paths"] = sorted({*done["paths"], path})
        mark.write_text(json.dumps(done, indent=1))


# ----------------------------------------------------------------- frontends


def buck_run(target, args, log_path):
    """Run a frontend binary; its output goes to log_path. Returns the exit status."""
    cmd = ["buck2", "run", f"{TARGETS}/{target}", "--", *map(str, args)]
    log(f"{' '.join(cmd)}  (log: {log_path})")
    with open(log_path, "w") as f:
        return subprocess.run(
            cmd, cwd=FBCODE, stdout=f, stderr=subprocess.STDOUT
        ).returncode


def tail(path, n=15):
    return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])


def frontend_of(case):
    return "cute" if "builder" in case else "ttgir"


def src_root(case_path, case, src):
    """The exported source tree: --src, else the case's, relative to the case file."""
    if src:
        return Path(src).resolve()
    given = case["source"]["src"] if frontend_of(case) == "cute" else case["src_root"]
    return (case_path.parent / given).resolve()


def source_rev(case, root, src):
    """The revision of the analyzed sources: the one `export` recorded in the tree, else
    the case's when its own tree is analyzed, else unknown (None)."""
    mark = root / EXPORT_MARK
    if mark.is_file():
        return json.loads(mark.read_text())["rev"]
    if src:
        return None
    return case["source"].get("rev") if frontend_of(case) == "cute" else case.get("rev")


def run_frontend(case_path, case, root, out, inputs, target):
    """Dump and translate; returns ([KIR paths], [{"variant", "kernel", "inputs", "kind",
    "error"}] for the variants without a KIR, [{"input", "error"}] for the inputs the host
    code rejected: the dump's `host_error`, an exception the exported sources raised or no
    launch, in either frontend). Raises RuntimeError if the frontend itself failed."""
    fe = out / "frontend"
    logs = out / "logs"
    fe.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    sel = ["--inputs", inputs] if inputs else []
    if frontend_of(case) == "cute":
        kirs, failures = run_cute(case_path, root, fe, logs, sel)
    else:
        kirs, failures = run_ttgir(case_path, root, fe, logs, sel, target)
    rejected = [
        {"input": i["name"], "error": i["host_error"]}
        for i in json.loads((fe / "dump.json").read_text())["inputs"]
        if "host_error" in i
    ]
    return kirs, failures, rejected


def run_cute(case_path, root, fe, logs, sel):
    """The CuTe frontend (dump and to_kir in one binary): ([KIR paths], variants without
    a KIR, each with its kind: "gap" or "error")."""
    status = buck_run(
        "frontend/cute:lkv_cute",
        ["run", case_path, "--src", root, "--out", fe, *sel],
        logs / "frontend.log",
    )
    report = fe / "kir" / "report.json"
    if status != 0 or not report.exists():
        raise RuntimeError(f"CuTe frontend failed:\n{tail(logs / 'frontend.log')}")
    kirs, failures = [], []
    for e in json.loads(report.read_text()):
        if "kir" in e:
            kirs.append(Path(e["kir"]))
        elif e["kernel"] != "(host)":  # a rejected input, read from dump.json
            failures.append(
                {
                    "variant": f"{e['kernel']}__{e['digest']}",
                    "kernel": e["kernel"],
                    "inputs": e.get("inputs", []),
                    "kind": e.get("kind", "gap"),
                    "error": e["error"],
                }
            )
        p1 = e.get("p1")
        if p1 and p1["status"] == "checked":
            kirs.append(Path(p1["kir"]))
        elif p1:
            P1_NOT_CHECKED.append(
                {
                    "variant": f"{e['kernel']}__{e['digest']}",
                    "kernel": e["kernel"],
                    "inputs": e.get("inputs", []),
                    "status": p1["status"],
                    "error": p1.get("error", ""),
                }
            )
    return kirs, failures


# P1 translations without a P1 KIR (refused, failed: "error", not applicable), filled by
# the frontends.
P1_NOT_CHECKED = []


def run_ttgir(case_path, root, fe, logs, sel, target):
    """The TTGIR frontend (dump, then to_kir): ([KIR paths], variants without a KIR, each
    with its kind: "gap" or "error")."""
    status = buck_run(
        "frontend/ttgir:dump",
        ["--case", case_path, "--src", root, "--out", fe, "--target", target, *sel],
        logs / "dump.log",
    )
    if status != 0:
        raise RuntimeError(f"TTGIR dump failed:\n{tail(logs / 'dump.log')}")
    for old in fe.glob("*.kir.json"):
        old.unlink()
    status = buck_run(
        "frontend/ttgir:to_kir",
        ["--dump", fe / "dump.json", "--case", case_path],
        logs / "to_kir.log",
    )
    dump = json.loads((fe / "dump.json").read_text())
    failures = []
    for line in (logs / "to_kir.log").read_text().splitlines():
        m = re.match(r"\[to_kir\] (\S+): FAILED \((gap|error)\): (.*)", line)
        if m:
            inputs = [
                i["name"]
                for i in dump["inputs"]
                if any(la["spec"] == m.group(1) for la in i.get("launches", []))
            ]
            failures.append(
                {
                    "variant": m.group(1),
                    "kernel": dump["kernel"],
                    "inputs": inputs,
                    "kind": m.group(2),
                    "error": m.group(3),
                }
            )
    if status != 0 and not failures:
        raise RuntimeError(f"TTGIR to_kir failed:\n{tail(logs / 'to_kir.log')}")
    specs = dump["specs"]
    kirs = [fe / f"{k}.kir.json" for k in specs if (fe / f"{k}.kir.json").exists()]
    failed = {f["variant"] for f in failures}
    lost = [k for k in specs if not (fe / f"{k}.kir.json").exists() and k not in failed]
    if lost:
        raise RuntimeError(
            f"TTGIR to_kir wrote neither a KIR nor a failure for {lost}:\n"
            f"{tail(logs / 'to_kir.log')}"
        )
    P1_NOT_CHECKED.extend(
        {
            "variant": k.name.removesuffix(".kir.json"),
            "status": "not_checked",
            "error": "P1-1 not checked: the TTGIR frontend has no TMEM events",
        }
        for k in kirs
    )
    return kirs, failures


# ------------------------------------------------------------------- Layer 2


class Sources:
    """IR loc `path:line` -> (exported file:line, text of that line)."""

    def __init__(self, root):
        self.root = root
        self.files = {}

    def resolve(self, loc):
        path, _, line = loc.rpartition(":")
        if self.root is None or not line.isdigit():
            return None, None
        for cand in (self.root / path, self.root / path.removeprefix("fbcode/")):
            if cand.is_file():
                if cand not in self.files:
                    self.files[cand] = cand.read_text(errors="replace").splitlines()
                lines = self.files[cand]
                text = (
                    lines[int(line) - 1].strip()
                    if 0 < int(line) <= len(lines)
                    else None
                )
                return f"{cand}:{line}", text
        return None, None


REPORT_HEAD = re.compile(r"== .* / input (\S+)( \(flags .*\))?$")
SPEC_LINE = re.compile(r"(P[01]-\d) .*: (PASS|FAIL|INCONSISTENT)$")
NOTES_LINE = re.compile(r"NOTES (\S+) (P[01]-\d) (\d+)$")
# A refuted spec without a certain counterexample (lean/KV/Check.lean `Report.render`).
UNDECIDED_LINE = re.compile(r"UNDECIDED (\S+) (P[01]-\d)$")
# An input whose evaluation stopped, with the error's kind (lean/KV/Eval.lean `ErrKind`).
EVAL_ERROR_LINE = re.compile(
    r"evaluation error \((kernel error|model limit|unexpected)\): (.*)$"
)
EVAL_KINDS = {
    "kernel error": "kernel",
    "model limit": "limit",
    "unexpected": "unexpected",
}
# What the error means for the input's specs, as Lean appends it (`ErrKind.consequence`);
# the report's section headings say it instead.
EVAL_CONSEQUENCE = re.compile(
    r" \((a certain error of the kernel|a limit of the model|the case or LKV needs a fix)"
    r"[^()]*\)$"
)
MORE = re.compile(r"\.\.\. (\d+) more$")
REFUTED = re.compile(
    r"error: Tactic `native_decide` evaluated that the proposition\s+"
    r"KV\.Holds KV\.SpecId\.(\w+) kernel input(\d+)_\S*\s+is false"
)


def new_spec(report=None):
    return {
        "report": report,
        "counterexamples": [],
        "more": 0,
        "notes": [],
        "notes_more": 0,
        "note_count": 0,
        "lean_undecided": False,
    }


def add_line(spec, body):
    """One indented line under a spec: a counterexample `loc: context: message`, a note
    `note: loc: context: message`, or a count of the unprinted ones, `[note: ]... N more`."""
    kind, more = "counterexamples", "more"
    if body.startswith("note: "):
        kind, more, body = "notes", "notes_more", body.removeprefix("note: ")
    if m := MORE.match(body):
        spec[more] = int(m.group(1))
    elif not body.startswith("internal error"):
        loc, context, msg = (body.split(": ", 2) + ["", ""])[:3]
        spec[kind].append({"loc": loc, "context": context, "msg": msg})


def parse_report(text, names):
    """Per input, the `#eval KV.runCase` report (lean/KV/Check.lean `Report.render`)."""
    inputs = {
        n: {"name": n, "eval_error": None, "eval_kind": None, "specs": {}}
        for n in names
    }
    cur = spec = None
    for line in text.splitlines():
        if m := REPORT_HEAD.match(line):
            cur, spec = inputs.get(m.group(1)), None
        elif (m := NOTES_LINE.match(line)) and m.group(1) in inputs:
            specs = inputs[m.group(1)]["specs"]
            specs.setdefault(m.group(2), new_spec())["note_count"] = int(m.group(3))
        elif (m := UNDECIDED_LINE.match(line)) and m.group(1) in inputs:
            specs = inputs[m.group(1)]["specs"]
            specs.setdefault(m.group(2), new_spec())["lean_undecided"] = True
        elif cur is None or line.startswith("VERDICT "):
            cur = spec = None
        elif m := EVAL_ERROR_LINE.match(line):
            cur["eval_kind"] = EVAL_KINDS[m.group(1)]
            cur["eval_error"] = EVAL_CONSEQUENCE.sub("", m.group(2))
        elif m := SPEC_LINE.match(line):
            spec = cur["specs"][m.group(1)] = new_spec(m.group(2))
        elif spec is not None and line.startswith("  "):
            add_line(spec, line[2:])
    return inputs


def parse_lean(text, k, status):
    """Per input of KIR `k`: every spec's verdict (its native_decide theorem, FAIL iff
    refuted), its outcome (PASS; FAIL for a refuted theorem Lean does not mark `UNDECIDED`:
    a certain counterexample or a kernel error; ERROR for an INCONSISTENT verdict and,
    after an unexpected evaluation error, for an `UNDECIDED` one; else UNDECIDED: a known
    limit), the report's counterexamples and notes. Raises RuntimeError on any other Lean
    error, on an exit status that disagrees with the refuted theorems (e.g. Lean killed
    before checking them all), on a note count the report does not list and on an
    `UNDECIDED` proved theorem."""
    names = [i["name"] for i in k["case"]["inputs"]]
    inputs = parse_report(text, names)
    refuted = {(names[int(i)], SPECS[spec]) for spec, i in REFUTED.findall(text)}
    errors = len(re.findall(r"^\S+:\d+:\d+: error", text, re.M))
    if errors != len(refuted):
        raise RuntimeError(
            f"Lean reported {errors - len(refuted)} error(s) besides refuted theorems"
        )
    if status != (1 if refuted else 0):
        raise RuntimeError(
            f"Lean exited with status {status} and {len(refuted)} refuted theorem(s)"
        )
    for n, inp in inputs.items():
        for label in LABELS_P1 if "tmem" in k else LABELS:
            s = inp["specs"].setdefault(label, new_spec())
            theorem = "FAIL" if (n, label) in refuted else "PASS"
            # The theorem is the verdict; the #eval report must agree with it.
            s["verdict"] = theorem if s.pop("report") == theorem else "INCONSISTENT"
            undecided = s.pop("lean_undecided")
            if undecided and theorem == "PASS":
                raise RuntimeError(f"input {n} {label}: UNDECIDED, but proved")
            if s["verdict"] in ("PASS", "INCONSISTENT"):
                s["outcome"] = "PASS" if s["verdict"] == "PASS" else "ERROR"
            elif not undecided:
                s["outcome"] = "FAIL"  # a certain counterexample or kernel error
            elif inp["eval_kind"] == "unexpected":
                s["outcome"] = "ERROR"
            else:
                s["outcome"] = "UNDECIDED"  # a model limit, or a known limit of a spec
            if s["note_count"] != len(s["notes"]) + s["notes_more"]:
                raise RuntimeError(
                    f"input {n} {label}: NOTES gives {s['note_count']} notes, the "
                    f"report lists {len(s['notes']) + s['notes_more']}"
                )
    return [inputs[n] for n in names]


def check_kir(path, lean_dir, sources, timeout):
    """Validate, emit and check one KIR file; returns its report entry."""
    entry = {
        "variant": path.name.removesuffix(".json").removesuffix(".kir"),
        "kir": str(path),
    }
    k = json.loads(path.read_text())
    entry.update(
        kernel=k.get("kernel"),
        flags=k.get("case", {}).get("inputs", [{}])[0].get("flags", {}),
        p1="tmem" in k,
        input_names=[i.get("name") for i in k.get("case", {}).get("inputs", [])],
    )
    errors = kir.validate(k)
    if errors:
        return dict(entry, status="invalid_kir", error="; ".join(errors[:10]))
    lean = lean_dir / f"{entry['variant']}.lean"
    lean.write_text(emit_lean.emit(k, path.name))
    start = time.time()
    try:
        r = subprocess.run(
            [SKILL / "scripts/run_lean_check.sh", lean],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return dict(
            entry, status="lean_error", error=f"Lean timed out after {timeout} s"
        )
    out = lean.with_suffix(".out")
    out.write_text(r.stdout + r.stderr)
    entry.update(
        lean=str(lean), lean_output=str(out), seconds=round(time.time() - start, 1)
    )
    try:
        inputs = parse_lean(r.stdout + r.stderr, k, r.returncode)
    except RuntimeError as e:
        return dict(entry, status="lean_error", error=f"{e}; see {out}")
    for inp in inputs:
        for s in inp["specs"].values():
            for f in s["counterexamples"] + s["notes"]:
                f["source"], f["line_text"] = sources.resolve(f["loc"])
    return dict(entry, status="checked", inputs=inputs)


def layer2(kirs, out, sources, jobs, timeout):
    lean_dir = out / "lean"
    lean_dir.mkdir(parents=True, exist_ok=True)
    for old in [*lean_dir.glob("*.lean"), *lean_dir.glob("*.out")]:
        old.unlink()
    build = subprocess.run(
        [SKILL / "scripts/build_lean.sh"], capture_output=True, text=True
    )
    if build.returncode != 0:
        raise RuntimeError(f"scripts/build_lean.sh failed:\n{build.stderr[-2000:]}")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(check_kir, p, lean_dir, sources, timeout): p for p in kirs
        }
        for n, f in enumerate(concurrent.futures.as_completed(futures), 1):
            e = f.result()
            specs = [s for i in e.get("inputs", []) for s in i["specs"].values()]
            count = Counter(s["outcome"] for s in specs)
            notes = sum(s["note_count"] for s in specs)
            state = (
                f"{count['FAIL']} FAIL, {count['ERROR']} ERROR, {count['UNDECIDED']} "
                f"UNDECIDED, {notes} notes"
                if e["status"] == "checked"
                else e["status"]
            )
            log(
                f"lean {n}/{len(kirs)} {e['variant']}: {state} ({e.get('seconds', '-')} s)"
            )
            results.append(e)
    return sorted(results, key=lambda e: e["variant"])


# ------------------------------------------------------------------- reports


def unchecked(report, p1=False):
    """The variants without verdicts, each {variant, kernel, inputs, why, outcome}: ERROR for
    a P0 (with `p1`: P1) KIR whose check did not finish and for a frontend failure (kind
    "error"; P1: status "error"), UNDECIDED for a construct the frontend refuses by design
    (kind "gap"; P1: status "refused"). `inputs` names the inputs that launched the variant,
    if the frontend says; every spec of each of them has the variant's outcome."""
    out = [
        {
            "variant": v["variant"],
            "kernel": v.get("kernel"),
            "inputs": v.get("input_names", []),
            "why": f"{v['status']}: {v.get('error', '')}",
            "outcome": "ERROR",
        }
        for v in (report.get("variants_p1", []) if p1 else report["variants"])
        if v["status"] != "checked"
    ]
    if p1:
        failed = [
            (x, x["status"] == "error")
            for x in report.get("p1_not_checked", [])
            if x["status"] in ("refused", "error")
        ]
    else:
        failed = [(x, x.get("kind") == "error") for x in report["frontend_failures"]]
    for x, err in failed:
        what = (
            ("failed" if err else "refused") if p1 else ("error" if err else "refused")
        )
        out.append(
            {
                "variant": x["variant"],
                "kernel": x.get("kernel") or x["variant"].split("__")[0],
                "inputs": x.get("inputs", []),
                "why": f"{'P1 translation' if p1 else 'frontend'} {what}: {x['error']}",
                "outcome": "ERROR" if err else "UNDECIDED",
            }
        )
    return out


def totals(variants, labels=None, no_verdicts=()):
    """Per spec, over (variant, input) runs: the outcomes (every spec of each input of a
    variant in `no_verdicts` has the variant's outcome), the INCONSISTENT verdicts among the
    ERROR ones, and the runs with notes."""
    labels = labels or LABELS
    t = {
        label: {**dict.fromkeys(OUTCOMES, 0), "INCONSISTENT": 0, "with_notes": 0}
        for label in labels
    }
    for v in variants:
        for inp in v.get("inputs", []):
            for label, s in inp["specs"].items():
                t[label][s["outcome"]] += 1
                t[label]["INCONSISTENT"] += s["verdict"] == "INCONSISTENT"
                t[label]["with_notes"] += s["note_count"] > 0
    for u in no_verdicts:
        for label in labels:
            t[label][u["outcome"]] += len(u["inputs"])
    return t


def tally(variants):
    """Over the checked variants' runs: certain violations per spec (counterexamples), runs
    whose evaluation stopped, per error kind, INCONSISTENT verdicts, and the specs refuted
    without a certain counterexample per spec (UNDECIDED, no evaluation error)."""
    fails, evals, refuted = Counter(), Counter(), Counter()
    inconsistent = 0
    for v in variants:
        for inp in v.get("inputs", []):
            if inp["eval_error"]:
                evals[inp["eval_kind"]] += 1
                continue
            for label, s in inp["specs"].items():
                if s["verdict"] == "INCONSISTENT":
                    inconsistent += 1
                elif s["outcome"] == "FAIL":
                    fails[label] += 1
                elif s["outcome"] == "UNDECIDED":
                    refuted[label] += 1
    return fails, evals, inconsistent, refuted


def per_spec(counts):
    return ", ".join(f"{label}: {n}" for label, n in sorted(counts.items()))


def names(us):
    return ", ".join(code(u["variant"]) for u in us)


def reasons(prefix, variants, no_verdicts, counts):
    """{outcome: report lines} of one part of a run (P0, or P1-1 with `prefix`): its
    failures (ERROR) always, its certain violations and known limits only if `counts`."""
    fails, evals, inconsistent, refuted = tally(variants)
    errs, gaps = (
        [u for u in no_verdicts if u["outcome"] == o] for o in ("ERROR", "UNDECIDED")
    )
    lines = {
        "FAIL": [
            (fails, f"{sum(fails.values())} certain violation(s) ({per_spec(fails)})"),
            (
                evals["kernel"],
                f"{evals['kernel']} (variant, input) run(s) stopped at a certain error "
                "of the kernel",
            ),
        ],
        "ERROR": [
            (
                errs,
                f"{len(errs)} variant(s) that failed to translate or check: {names(errs)}",
            ),
            (
                evals["unexpected"],
                f"{evals['unexpected']} (variant, input) run(s) whose evaluation stopped "
                "unexpectedly (the case or LKV needs a fix)",
            ),
            (inconsistent, f"{inconsistent} INCONSISTENT verdict(s) (a Layer 2 bug)"),
        ],
        "UNDECIDED": [
            (
                gaps,
                f"{len(gaps)} variant(s) the frontend refuses by design: {names(gaps)}",
            ),
            (
                evals["limit"],
                f"{evals['limit']} (variant, input) run(s) whose evaluation stopped at a "
                "limit of the model",
            ),
            (
                refuted,
                f"{sum(refuted.values())} spec(s) refuted without a certain "
                f"counterexample ({per_spec(refuted)})",
            ),
        ],
    }
    return {
        o: [prefix + text for cond, text in ls if cond and (counts or o == "ERROR")]
        for o, ls in lines.items()
    }


def run_reason(report):
    """(outcome, report line) for a run whose pipeline failed or in which no variant
    reached the Lean checks, unless its variants' own reasons say it all; else None."""
    if report.get("error"):
        return "ERROR", f"pipeline error: {report['error'].splitlines()[0]}"
    if any(v["status"] == "checked" for v in report["variants"]):
        return None
    missing = unchecked(report)
    if not missing:
        return "ERROR", (
            "no variant reached the Lean checks: the frontend made no KIR (see the "
            "inputs the host rejected)"
        )
    if all(u["outcome"] == "UNDECIDED" for u in missing):
        return "UNDECIDED", (
            "no variant reached the Lean checks: the frontend refuses every variant by "
            "design"
        )
    return None


def result(report):
    """The run's outcome (module doc), its exit status and why, as report lines: "fail"
    (certain violations), "error" (failures to fix) and "undecided" (known limits, not
    errors). P1-1's FAIL and UNDECIDED count only with --p1-strict (report["p1_strict"]);
    its errors always do: they are failures of LKV."""
    lines = {"FAIL": [], "ERROR": [], "UNDECIDED": []}
    parts = [
        ("", report["variants"], unchecked(report), True),
        (
            "P1-1: ",
            report.get("variants_p1", []),
            unchecked(report, p1=True),
            bool(report.get("p1_strict")),
        ),
    ]
    for part in parts:
        for o, ls in reasons(*part).items():
            lines[o] += ls
    if extra := run_reason(report):
        lines[extra[0]].append(extra[1])
    outcome = next((o for o in ("FAIL", "ERROR", "UNDECIDED") if lines[o]), "PASS")
    return {
        "outcome": outcome,
        "exit_status": EXIT_STATUS[outcome],
        "fail": lines["FAIL"],
        "error": lines["ERROR"],
        "undecided": lines["UNDECIDED"],
    }


def exit_status(report):
    return result(report)["exit_status"]


def code(s):
    return "`" + s.replace("`", "'") + "`"


def runs_of(variants):
    """(variant, input) pairs of the checked variants."""
    return [(v, i) for v in variants for i in v.get("inputs", [])]


def md_result(report):
    r = result(report)
    out = [f"## Result: {r['outcome']} (exit status {r['exit_status']})", ""]
    out.append(
        {
            "PASS": "Every spec is proved on every input of every compiled variant.",
            "FAIL": "Certain violations in the extracted model: confirm them in the source.",
            "ERROR": "No certain violation, but something failed: fix the case, LKV or the "
            "environment.",
            "UNDECIDED": "No certain violation and no failure; not decided where LKV has "
            "known limits (not an error):",
        }[r["outcome"]]
    )
    out += [f"- FAIL: {x}" for x in r["fail"]]
    out += [f"- ERROR: {x}" for x in r["error"]]
    out += [f"- not decided (known limit): {x}" for x in r["undecided"]]
    return out


def md_summary(report):
    t = report["totals"]
    if not any(x[o] for x in t.values() for o in OUTCOMES):
        return ["## Summary", "", "No variant reached the Lean checks (see below)."]
    out = [
        "## Summary over (variant, input) runs",
        "",
        "| spec | PASS | FAIL | ERROR | UNDECIDED | runs with notes |",
        "|---|---|---|---|---|---|",
    ]
    for label, x in t.items():
        bad = f" ({x['INCONSISTENT']} inconsistent)" if x["INCONSISTENT"] else ""
        out.append(
            f"| {label} {TITLES[label]} | {x['PASS']} | {x['FAIL']} | {x['ERROR']}{bad} "
            f"| {x['UNDECIDED']} | {x['with_notes']} |"
        )
    return out


def md_per_input(variants, no_verdicts=(), labels=None):
    labels = labels or LABELS
    cells = defaultdict(lambda: defaultdict(list))  # (kernel, input) -> spec -> [s]
    for v, inp in runs_of(variants):
        for label, s in inp["specs"].items():
            cells[(v["kernel"], inp["name"])][label].append(
                (s["outcome"], s["note_count"] > 0)
            )
    for u in no_verdicts:
        for name in u["inputs"]:
            for label in labels:
                cells[(u["kernel"], name)][label].append((u["outcome"], False))
    if not cells:
        return []
    out = [
        "## Verdicts per input",
        "",
        "Over the n compiled variants of the kernel that ran the input: **FAIL** (k/n) if k",
        "of them have a certain violation, **ERROR** (e/n) if e of them failed to decide it,",
        "UNDECIDED (u/n) if u of them hit a known limit (a variant without verdicts counts",
        "for every spec), else PASS; notes (j/n): j of them have notes for the spec.",
        "",
        "| kernel | input | " + " | ".join(labels) + " |",
        "|---|---|" + "---|" * len(labels),
    ]
    for (kernel, name), specs in cells.items():
        row = []
        for label in labels:
            n = len(specs[label])
            count = Counter(o for o, _ in specs[label])
            noted = sum(x for _, x in specs[label])
            parts = [f"**FAIL** ({count['FAIL']}/{n})"] if count["FAIL"] else []
            parts += [f"**ERROR** ({count['ERROR']}/{n})"] if count["ERROR"] else []
            parts += (
                [f"UNDECIDED ({count['UNDECIDED']}/{n})"] if count["UNDECIDED"] else []
            )
            cell = ", ".join(parts) or "PASS"
            row.append(cell + (f"; notes ({noted}/{n})" if noted else ""))
        out.append(f"| {kernel} | {code(name)} | " + " | ".join(row) + " |")
    return out


def md_findings(variants, kind, title, intro, labels=None, keep=None, level="###"):
    """Findings of `kind` ("counterexamples" or "notes") of the specs `keep(input, spec)`
    selects, grouped by spec and source line (IR loc; findings without one, such as
    coverage holes, by their context): per group the resolved source line, one example
    and the runs it occurs in."""
    groups = defaultdict(lambda: {"runs": [], "example": None})  # (spec, where)
    for v, inp in runs_of(variants):
        for label, s in inp["specs"].items():
            if keep and not keep(inp, s):
                continue
            for f in s[kind]:
                where = (
                    code(f["loc"])
                    if f["loc"] != "?"
                    else f"(no IR location) {f['context']}"
                )
                g = groups[(label, where)]
                run = f"{inp['name']} [{v['variant']}]"
                if run not in g["runs"]:
                    g["runs"].append(run)
                g["example"] = g["example"] or f
    if not groups:
        return []
    out = [title, "", *intro, ""]
    labels = labels or LABELS
    for label in labels:
        mine = [(where, g) for (lb, where), g in groups.items() if lb == label]
        if mine and len(labels) > 1:
            out += [f"{level} {label} {TITLES[label]}", ""]
        for where, g in sorted(mine, key=lambda x: -len(x[1]["runs"])):
            f = g["example"]
            src = f" ({code(f['source'])})" if f["source"] else ""
            out.append(f"- {where}{src}: {len(g['runs'])} run(s)")
            if f["line_text"]:
                out.append(f"  - source: {code(f['line_text'])}")
            out.append(f"  - e.g. {f['context']}: {f['msg']}")
            more = f" and {len(g['runs']) - 6} more" if len(g["runs"]) > 6 else ""
            out.append(f"  - runs: {', '.join(map(code, g['runs'][:6]))}{more}")
        if mine:
            out.append("")
    return out


def failing(inp, s):
    return s["outcome"] == "FAIL" and not inp["eval_error"]


def refuted_open(inp, s):
    """UNDECIDED: refuted without a certain counterexample, by no evaluation error."""
    return s["outcome"] == "UNDECIDED" and not inp["eval_error"]


def eval_errors(variants, kind):
    """`- kernel input (n variant(s)): error` lines of the runs whose evaluation stopped
    with an error of `kind`."""
    errors = defaultdict(list)  # (kernel, input, error) -> variants
    for v, inp in runs_of(variants):
        if inp["eval_error"] and inp["eval_kind"] == kind:
            errors[(v["kernel"], inp["name"], inp["eval_error"])].append(v["variant"])
    return [
        f"- {kernel} {code(name)} ({len(vs)} variant(s)): {e}"
        for (kernel, name, e), vs in errors.items()
    ]


def md_fail(variants, labels=None, title="## FAIL", sub="###"):
    """Certain violations: counterexamples by spec and source line, and the runs whose
    evaluation stopped at a certain error of the kernel."""
    out = md_findings(
        variants,
        "counterexamples",
        f"{sub} Counterexamples by spec and source line",
        [
            "Certain violations in the extracted model (references/specs.md): confirm each in",
            "the source. A `not decided on this input` line is not a violation: it says which",
            "part of that spec is not decided, and why.",
        ],
        labels,
        keep=failing,
        level=sub + "#",
    )
    kernel = eval_errors(variants, "kernel")
    if kernel:
        out += [
            f"{sub} Evaluation stopped at a certain error of the kernel",
            "",
            "Every spec of the input FAILs: the kernel or its host certainly fails on it.",
            "",
            *kernel,
            "",
        ]
    return [title, ""] + out if out else []


def md_errors(variants, no_verdicts, labels=None, title="## ERROR", sub="###"):
    """Failures of LKV, the case or the environment: variants without verdicts by a
    failure, unexpected evaluation errors, INCONSISTENT verdicts."""
    out = []
    errs = [u for u in no_verdicts if u["outcome"] == "ERROR"]
    if errs:
        out += [f"{sub} Variants that failed to translate or check", ""]
        for u in errs:
            ins = (
                f" (inputs {', '.join(map(code, u['inputs']))})" if u["inputs"] else ""
            )
            out.append(f"- {code(u['variant'])}{ins}: {u['why']}")
        out.append("")
    unexpected = eval_errors(variants, "unexpected")
    if unexpected:
        out += [
            f"{sub} Evaluation errors",
            "",
            "The evaluation stopped at something the case or LKV must fix; no spec of the",
            "input is decided.",
            "",
            *unexpected,
            "",
        ]
    inconsistent = [
        f"- {v['kernel']} {code(inp['name'])} {label} ({code(v['variant'])})"
        for v, inp in runs_of(variants)
        for label, s in inp["specs"].items()
        if s["verdict"] == "INCONSISTENT"
    ]
    if inconsistent:
        out += [
            f"{sub} INCONSISTENT verdicts",
            "",
            "The counterexample finder and the decision procedure disagree: a Layer 2 bug.",
            "",
            *inconsistent,
            "",
        ]
    if not out:
        return []
    intro = "Failures to fix in the case, LKV or the environment; none of it is a pass."
    return [title, "", intro, ""] + out


def md_gaps(variants, no_verdicts, labels=None, title="## UNDECIDED", sub="###"):
    """Known limits of LKV (not errors): variants the frontend refuses by design, model
    limits of the evaluation, specs refuted without a certain counterexample."""
    out = []
    gaps = [u for u in no_verdicts if u["outcome"] == "UNDECIDED"]
    if gaps:
        out += [f"{sub} Variants the frontend refuses by design", ""]
        for u in gaps:
            ins = (
                f" (inputs {', '.join(map(code, u['inputs']))})" if u["inputs"] else ""
            )
            out.append(f"- {code(u['variant'])}{ins}: {u['why']}")
        out.append("")
    limit = eval_errors(variants, "limit")
    if limit:
        out += [
            f"{sub} Evaluation stopped at a limit of the model",
            "",
            "No spec of the input is decided (e.g. a division by zero: the GPU computes an",
            "unspecified value, which the kernel may never use).",
            "",
            *limit,
            "",
        ]
    out += md_findings(
        variants,
        "counterexamples",
        f"{sub} Specs refuted without a certain counterexample, by source line",
        [
            "A `not decided on this input` line says why each is not decided; nothing listed",
            "here is a certain violation (references/specs.md).",
        ],
        labels,
        keep=refuted_open,
        level=sub + "#",
    )
    if not out:
        return []
    intro = "Not decided where LKV has a known limit: not errors, and not a pass."
    return [title, "", intro, ""] + out


def md_notes(report):
    return md_findings(
        report["variants"],
        "notes",
        "## Notes by spec and source line",
        [
            "Not violations: findings that can be harmless depending on data values or on",
            "how the kernel uses the result. They never change a verdict.",
        ],
    )


def md_variants(report):
    out = []
    if report["host_rejected"]:
        out += ["## Inputs the host code rejected", ""]
        out += [f"- {code(r['input'])}: {r['error']}" for r in report["host_rejected"]]
        out.append("")
    if report["variants"]:
        out += [
            "## Variants",
            "",
            "| variant | kernel | flags on (others off) | inputs | Lean s |",
            "|---|---|---|---|---|",
        ]
        for v in report["variants"]:
            flags = ", ".join(k for k, b in v.get("flags", {}).items() if b)
            out.append(
                f"| {code(v['variant'])} | {v.get('kernel')} | {flags} "
                f"| {len(v.get('inputs', []))} | {v.get('seconds', '-')} |"
            )
    return out


def md_env(report):
    """The environment the frontend ran with (dump.json "env", frontend/ttgir/host_env.py)."""
    env = report.get("env")
    if not env:
        return []

    def pairs(d):
        return ", ".join(code(f"{k}={v}") for k, v in d.items()) or "none"

    out = [
        "## Environment of the frontend run",
        "",
        f"- case `env`: {pairs(env['case'])}",
        f"- shell values ignored (unset before the compiler and kernel imports): {pairs(env['unset'])}",
        f"- read by the exported sources (unset unless effective): {', '.join(map(code, env['read_by_sources'])) or 'none'}",
        f"- effective (variables the exported sources read, compiler switches): {pairs(env['effective'])}",
    ]
    if "triton_cache_invalidating" in env:
        out.append(
            f"- Triton cache-invalidating variables: {pairs(env['triton_cache_invalidating'])}"
        )
    return out


def md_p1(report):
    """The P1 section: P1-1 verdicts per input, FAIL, ERROR and UNDECIDED findings, notes
    by source line, and the variants P1-1 does not check."""
    variants = report.get("variants_p1", [])
    nc = report.get("p1_not_checked", [])
    if not variants and not nc:
        return []
    strict = report.get("p1_strict")
    out = [
        "## P1-1 TMEM reuse ordered (P1; CuTe only; "
        + (
            "part of the outcome: --p1-strict)"
            if strict
            else "its FAIL and UNDECIDED are not part of the outcome, its ERRORs are)"
        ),
        "",
        "Decided on the P1 KIR (`.p1.json`) of each variant: every pair of TMEM accesses of",
        "different roles that touch a common cell, one of them writing, must be ordered (one",
        "completes before the other is issued) under sound mbarrier and named-barrier",
        'happens-before (references/specs.md, "P1-1"). A FAIL is a pair unordered even under',
        "the most generous lockstep reading of the ledger: confirm it in the source before",
        "calling it a race. An UNDECIDED input has pairs only the lockstep contents order.",
        "",
    ]
    no_verdicts = unchecked(report, p1=True)
    if variants or no_verdicts:
        t = totals(variants, LABELS_P1, no_verdicts)["P1-1"]
        bad = f" ({t['INCONSISTENT']} inconsistent)" if t["INCONSISTENT"] else ""
        out += [
            f"P1-1 over (variant, input) runs: {t['PASS']} PASS, {t['FAIL']} FAIL, "
            f"{t['ERROR']} ERROR{bad}, {t['UNDECIDED']} UNDECIDED, {t['with_notes']} with "
            "notes.",
            "",
        ]
        table = md_per_input(variants, no_verdicts, LABELS_P1)
        out += table[next(i for i, x in enumerate(table) if x.startswith("| kernel")) :]
        out.append("")
        out += md_fail(variants, LABELS_P1, "### P1-1 FAIL", "####")
        out += md_errors(variants, no_verdicts, LABELS_P1, "### P1-1 ERROR", "####")
        out += md_gaps(variants, no_verdicts, LABELS_P1, "### P1-1 UNDECIDED", "####")
        out += md_findings(
            variants,
            "notes",
            "### P1-1 notes by source line",
            [
                "N1 same-role hazards; N2 intra-role cross-warp pairs; N3 a blocked wait makes",
                "happens-before vacuous.",
            ],
            LABELS_P1,
        )
    if variants:
        out += [
            "### P1 variants",
            "",
            "| variant | kernel | status | inputs | Lean s |",
            "|---|---|---|---|---|",
        ]
        for v in variants:
            out.append(
                f"| {code(v['variant'])} | {v.get('kernel')} | {v['status']}"
                f"{(': ' + v['error']) if v.get('error') else ''} | {len(v.get('inputs', []))} | {v.get('seconds', '-')} |"
            )
        out.append("")
    others = [x for x in nc if x["status"] not in ("refused", "error")]
    if others:
        out += ["### Variants P1-1 does not check", ""]
        out += [f"- {code(x['variant'])}: {x['status']}: {x['error']}" for x in others]
    return out


def markdown(report):
    out = [f"# lkv report: {report['case']}", ""]
    for key in ("case_file", "frontend", "target", "rev", "src", "out"):
        if report.get(key):
            out.append(f"- {key.replace('_', ' ')}: {code(str(report[key]))}")
    if report.get("error"):
        out += ["", f"Pipeline error: {report['error']}"]
    out += [""] + md_result(report) + [""]
    out += [
        "Each verdict is per compiled variant, concrete input and spec, for that input only",
        "(not all shapes). PASS: its theorem `KV.Holds` is proved by `native_decide`. FAIL:",
        "a violation certain in the extracted model (a counterexample, or a certain error of",
        "the kernel that stops the evaluation). ERROR: LKV, the case or the environment",
        "failed (a frontend or Lean failure, an INCONSISTENT verdict, an unexpected",
        "evaluation error). UNDECIDED: not decided where LKV has a known limit (a construct",
        "the frontend refuses by design, an out-of-bounds metadata load, a division by zero,",
        "P1-1 pairs only the lockstep contents order); not an error. Exit status: 1 FAIL,",
        "2 ERROR, else 0. Notes are findings that are not violations; they never change a",
        "verdict (references/specs.md).",
        "",
    ]
    for section in (
        md_env,
        md_summary,
        lambda r: md_per_input(r["variants"], unchecked(r)),
        lambda r: md_fail(r["variants"]),
        lambda r: md_errors(r["variants"], unchecked(r)),
        lambda r: md_gaps(r["variants"], unchecked(r)),
        md_notes,
        md_variants,
        md_p1,
    ):
        lines = section(report)
        out += lines + ([""] if lines and lines[-1] else [])
    return "\n".join(out)


def write_report(report, out):
    report["variants_p1"] = [v for v in report["variants"] if v.get("p1")]
    report["variants"] = [v for v in report["variants"] if not v.get("p1")]
    report["p1_not_checked"] = list(P1_NOT_CHECKED)
    report["totals"] = totals(report["variants"], LABELS, unchecked(report))
    report["totals_p1"] = totals(
        report["variants_p1"], LABELS_P1, unchecked(report, p1=True)
    )
    report["result"] = result(report)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "report.md").write_text(markdown(report))
    t = report["totals"]
    log(
        "verdicts: "
        + ", ".join(
            f"{label} {t[label]['PASS']} PASS / {t[label]['FAIL']} FAIL / "
            f"{t[label]['ERROR']} ERROR / {t[label]['UNDECIDED']} UNDECIDED "
            f"({t[label]['with_notes']} with notes)"
            for label in LABELS
        )
    )
    if report["variants_p1"] or report["p1_not_checked"]:
        t = report["totals_p1"]["P1-1"]
        log(
            f"P1-1 ({'part of the outcome: --p1-strict' if report.get('p1_strict') else 'only its ERRORs count'}): "
            f"{t['PASS']} PASS / {t['FAIL']} FAIL / {t['ERROR']} ERROR / {t['UNDECIDED']} "
            f"UNDECIDED ({t['with_notes']} with notes), {len(report['p1_not_checked'])} "
            "variant(s) not checked"
        )
    r = report["result"]
    log(
        f"result: {r['outcome']} (exit status {r['exit_status']})"
        + "".join(f"; FAIL: {x}" for x in r["fail"])
        + "".join(f"; ERROR: {x}" for x in r["error"])
        + "".join(f"; not decided: {x}" for x in r["undecided"])
    )
    log(f"report: {out / 'report.md'} and {out / 'report.json'}")
    return r["exit_status"]


# ---------------------------------------------------------------------- main


def cmd_run(args):
    case_path = Path(args.case).resolve()
    case = json.loads(case_path.read_text())
    if frontend_of(case) == "cute" and args.target != "sm_100":
        die("lkv: the CuTe frontend compiles for sm_100 only")
    if FBCODE is None:
        die(
            f"lkv: run builds the frontend with buck2, so the skill must be inside an "
            f"fbsource checkout ({SKILL} is not); check runs anywhere"
        )
    out = Path(
        args.out or f"/tmp/lkv_runs/{case_path.parent.name}/{case['name']}"
    ).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = src_root(case_path, case, args.src)
    report = {
        "case": case["name"],
        "case_file": str(case_path),
        "frontend": frontend_of(case),
        "target": args.target,
        "rev": source_rev(case, root, args.src),
        "src": str(root),
        "out": str(out),
        "host_rejected": [],
        "frontend_failures": [],
        "variants": [],
        "p1_strict": args.p1_strict,
    }
    try:
        kirs, report["frontend_failures"], report["host_rejected"] = run_frontend(
            case_path, case, root, out, args.inputs, args.target
        )
        if not args.p1:
            kirs = [k for k in kirs if not k.name.endswith(".p1.json")]
            P1_NOT_CHECKED.clear()
        report["env"] = json.loads((out / "frontend" / "dump.json").read_text())["env"]
        report["variants"] = layer2(kirs, out, Sources(root), args.jobs, args.timeout)
    except RuntimeError as e:
        log(str(e))
        report["error"] = str(e)
    return write_report(report, out)


def cmd_check(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.src).resolve() if args.src else None
    report = {
        "case": args.name,
        "src": str(root) if root else None,
        "out": str(out),
        "host_rejected": [],
        "frontend_failures": [],
        "variants": [],
        "p1_strict": args.p1_strict,
    }
    try:
        kirs = [Path(p).resolve() for p in args.kir]
        report["variants"] = layer2(kirs, out, Sources(root), args.jobs, args.timeout)
    except RuntimeError as e:
        log(str(e))
        report["error"] = str(e)
    return write_report(report, out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "check"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("case", help="case file (cases/<frontend>/<case>.json)")
            p.add_argument("--out", help="output directory")
            p.add_argument(
                "--inputs", help="comma-separated subset of the case's inputs"
            )
            p.add_argument(
                "--target",
                choices=("sm_100", "sm_103"),
                default="sm_100",
                help="GPU the TTGIR frontend compiles for",
            )
        else:
            p.add_argument("kir", nargs="+", help="KIR JSON files")
            p.add_argument("--out", required=True, help="output directory")
            p.add_argument("--name", default="kir", help="report title")
        p.add_argument("--src", help="exported source tree (default: the case's)")
        p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
        p.add_argument(
            "--timeout", type=int, default=3600, help="seconds per Lean file"
        )
        p.add_argument(
            "--p1-strict",
            action="store_true",
            help="P1-1's FAIL and UNDECIDED also count toward the outcome and exit status "
            "(its ERRORs always do)",
        )
        if name == "run":
            p.add_argument(
                "--p1",
                dest="p1",
                action="store_true",
                default=True,
                help="also check the P1 KIRs (default; CuTe only)",
            )
            p.add_argument("--no-p1", dest="p1", action="store_false", help="skip P1-1")
    p = sub.add_parser("export")
    p.add_argument("rev")
    p.add_argument("out", type=Path)
    p.add_argument("paths", nargs="+", help="fbsource-relative files or directories")
    p.add_argument(
        "--package-root",
        help="export a directory at OUT_DIR/<its path below this fbsource directory> "
        "(e.g. fbcode, for a package that imports itself by its full path)",
    )
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.rev, args.out.resolve(), args.paths, args.package_root)
        return 0
    return cmd_run(args) if args.cmd == "run" else cmd_check(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # an error of lkv.py is an ERROR: exit 2, never 1 (a FAIL)
        traceback.print_exc()
        sys.exit(EXIT_STATUS["ERROR"])
