---
name: Bug report
about: A wrong verdict, a crash, or a refusal you did not expect
title: ''
labels: ''
assignees: ''

---

**What happened**
A clear description of the problem. For a wrong verdict, say which spec, input and
variant, the verdict LKV reported and the one you expected, and why.

**How to reproduce**
The command you ran (`scripts/lkv.py run`, `check` or a test) and the case file or KIR
file. Attach the smallest kernel and case file that show the problem, if you can.

**Output**
The relevant part of `report.md` (or `report.json`), and the error or log lines for a
crash or a refusal.

**Versions**
- LKV commit:
- Lean (`scripts/lean.sh --version`):
- Python:
- Frontend: Triton with TLX (commit), or `nvidia-cutlass-dsl` version:

**Additional context**
Anything else that helps, for example whether the kernel misbehaves on a GPU.
