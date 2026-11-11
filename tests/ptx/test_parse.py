# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the PTX reader, the op table classification and the coverage tool.

    python3 -I tests/ptx/test_parse.py      (or the buck target :test_parse)

On two hand-written fixtures (fixtures/parse_llvm.ptx in the shape of a Triton kernel,
fixtures/parse_nvvm.ptx in the shape of a CuTe DSL kernel): module directives, parameters
and performance directives, inline-asm scopes whose local labels and registers are
renamed per scope, function-scope named registers shared by asm blocks, dual
destinations, vector, memory and tensor operands, guards, .loc inlined_at chains with
function names from .debug_str and line 0, .branchtargets with brx.idx, the trailing NUL
byte, and syntax errors naming the line. Then the op table: NVVM 20/23 and LLVM spellings
meet on one mnemonic, the design's class rules, unknown mnemonics raise; and coverage.py's
exit status and JSON.
"""

import contextlib
import io
import json
import sys
import tempfile
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

from ...frontend.ptx import coverage, ops, parse  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LLVM, NVVM = FIXTURES / "parse_llvm.ptx", FIXTURES / "parse_nvvm.ptx"


def _fixture(path):
    # The CuTe DSL writes its PTX followed by a NUL byte; the reader must not mind.
    return parse.parse(path.read_text() + "\0", str(path))


def _instrs(fn, opcode):
    return [i for i in fn.instructions() if i.opcode == opcode]


def _labels(fn):
    return [x.name for x in fn.body if isinstance(x, parse.Label)]


def _coverage(*argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return coverage.main([str(a) for a in argv])


def _one(stmt, regs=".reg .b32 %r<4>; .reg .pred %p<2>; .reg .b64 %rd<2>;"):
    """The instruction `stmt` in a minimal kernel."""
    text = f".version 9.3\n.target sm_100a\n.address_size 64\n.visible .entry k()\n{{\n{regs}\n{stmt};\n}}\n"
    return parse.parse(text).functions[0].instructions()[0]


class ParseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llvm, cls.nvvm = _fixture(LLVM), _fixture(NVVM)
        cls.ws, cls.mma = (
            cls.llvm.function("ws_kernel"),
            cls.nvvm.function("kernel_mma"),
        )

    def test_module_directives_variables_and_parameters(self):
        m = self.llvm
        self.assertEqual((m.version, m.target, m.address_size), ("9.3", "sm_100a", 64))
        self.assertEqual(m.files, {1: "<source>/example.py"})  # .file after the body
        smem, s = m.vars
        self.assertEqual(
            (smem.name, smem.space, smem.extern, smem.array, smem.align),
            ("global_smem", "shared", True, 0, 16),
        )
        self.assertEqual((s.name, s.space, s.init), ("_$_str", "global", (65, 66, 0)))
        self.assertEqual([f.name for f in m.entries], ["ws_kernel"])
        self.assertEqual(self.ws.directives, {"reqntid": (256,), "maxnreg": (168,)})
        desc, n, ptr = self.ws.params
        self.assertEqual(
            (desc.type, desc.array, desc.size, desc.align, desc.ptr),
            ("b8", 128, 128, 64, None),
        )
        self.assertEqual((n.name, n.size, n.align), ("ws_kernel_param_1", 4, 4))
        self.assertEqual((ptr.size, ptr.align, ptr.ptr), (8, 8, "global"))

    def test_nvvm_header_prototype_and_nul_byte(self):
        self.assertEqual(
            (self.nvvm.version, self.nvvm.files),
            ("8.8", {1: "/src/example/mma_kernel.py"}),
        )
        self.assertEqual(
            self.mma.directives,
            {
                "reqntid": (128, 1, 1),
                "minnctapersm": (1,),
                "explicitcluster": (),
                "reqnctapercluster": (2, 1, 1),
            },
        )
        vprintf = self.nvvm.function("vprintf")
        self.assertEqual(
            (vprintf.kind, vprintf.linkage, vprintf.body), ("func", ("extern",), None)
        )
        self.assertEqual(
            [p.name for p in vprintf.rets + vprintf.params],
            ["func_retval0", "vprintf_param_0", "vprintf_param_1"],
        )
        plain = parse.parse(NVVM.read_text())
        self.assertEqual(
            [i.text for i in plain.function("kernel_mma").instructions()],
            [i.text for i in self.mma.instructions()],
        )

    def test_inline_asm_scopes_rename_local_labels_and_registers(self):
        # Two asm blocks with `.reg .pred complete; waitLoop:` each: every name is local
        # to its block, and each loop branches back to its own label.
        loops = [i for i in _instrs(self.ws, "bra.uni") if i.asm is not None]
        waits = _instrs(self.ws, "mbarrier.try_wait.parity.shared::cta.b64")
        self.assertEqual(len(loops), 2)
        for wait, back in zip(waits, loops):
            pred = wait.operands[0]
            self.assertEqual(pred[0], "reg")
            self.assertTrue(pred[1].startswith("complete#"))
            self.assertEqual(back.guard, (pred[1], True))
            self.assertEqual(
                back.operands[0], ("sym", pred[1].replace("complete", "waitLoop"))
            )
            self.assertEqual(wait.asm, back.asm)
        self.assertNotEqual(waits[0].operands[0], waits[1].operands[0])
        self.assertEqual(sum(n.startswith("waitLoop#") for n in _labels(self.ws)), 2)
        # The same with LAB_WAIT/DONE, one block written on a single line.
        nv = [x for x in _labels(self.mma) if "#" in x]
        self.assertEqual(nv, ["LAB_WAIT#1", "DONE#1", "LAB_WAIT#2", "DONE#2"])
        one_line = [i for i in self.mma.instructions() if i.line == 72]
        self.assertEqual(
            [i.opcode for i in one_line],
            ["mbarrier.try_wait.parity.shared.b64", "bra.uni", "bra.uni"],
        )
        self.assertEqual(one_line[1].operands, (("sym", "DONE#2"),))
        # asm blocks are numbered; asm_locs holds the location where each begins.
        self.assertEqual(len(self.ws.asm_locs), 4)
        self.assertEqual(self.ws.asm_locs[2].line, 60)

    def test_function_scope_named_registers_are_shared_by_asm_blocks(self):
        regs = self.mma.regs
        self.assertEqual(
            (regs["q_desc_lo"], regs["q_desc_0"], regs["q_desc_1"]),
            ("b32", "b64", "b64"),
        )
        self.assertNotIn("q_desc_2", regs)
        self.assertIn("desc_b_1#3", regs)  # a range declared inside a braced block
        update = [
            i
            for i in _instrs(self.mma, "mov.b64")
            if i.operands[0] == ("reg", "q_desc_0")
        ]
        self.assertEqual(
            len(update), 2
        )  # set by the unbraced block, updated by the braced one
        self.assertEqual(update[1].operands[1][1][0], ("reg", "desc_a_lo#3"))
        mma = _instrs(self.mma, "tcgen05.mma.cta_group::1.kind::f16")[0]
        self.assertEqual(
            mma.operands[1:3], (("reg", "q_desc_0"), ("reg", "desc_b_0#3"))
        )
        self.assertEqual(mma.guard, ("leader_thread#3", False))

    def test_operands(self):
        elect = _instrs(self.ws, "elect.sync")[0]
        self.assertEqual(
            elect.operands, (("pair", ("reg", "%r5"), ("reg", "%p2")), ("imm", -1))
        )
        sink = _instrs(self.mma, "elect.sync")[0]
        self.assertEqual(
            sink.operands[0], ("pair", parse.SINK, ("reg", "leader_thread#3"))
        )
        ld = _instrs(self.ws, "ld.global.b32")[0]
        self.assertEqual(
            ld.operands,
            (("vec", (("reg", "%r11"),)), ("mem", ("reg", "%rd2"), 0, None)),
        )
        st = _instrs(self.mma, "st.global.v4.b32")[0]
        self.assertEqual(
            st.operands,
            (
                ("mem", ("reg", "%rd1"), -8, None),
                ("vec", (("reg", "%r1"), ("reg", "%r2"), ("imm", 0), ("imm", -1))),
            ),
        )
        tma = [i for i in self.mma.instructions() if i.base == "cp"][0]
        self.assertEqual(
            tma.operands[1],
            ("mem", ("reg", "%rd2"), 0, ("vec", (("reg", "%r9"), ("reg", "%r4")))),
        )
        a_tmem = _instrs(self.mma, "tcgen05.mma.cta_group::1.kind::f16")[1]
        self.assertEqual(
            a_tmem.operands[1], ("mem", ("reg", "%r4"), 16, None)
        )  # [%r4 + 0x10]
        self.assertEqual(
            _instrs(self.ws, "st.shared.b32")[0].operands,
            (("mem", ("sym", "global_smem"), 24, None), ("imm", 33554689)),
        )
        self.assertEqual(
            _instrs(self.mma, "mov.u32")[0].operands[1], ("sreg", "%ctaid.x")
        )
        self.assertEqual(
            _instrs(self.mma, "mov.f32")[0].operands[1], ("f32", 0x3F800000)
        )
        self.assertEqual(
            _instrs(self.mma, "mov.f64")[0].operands[1], ("f64", 0x3FF0000000000000)
        )
        self.assertEqual(
            _instrs(self.mma, "mov.b64")[0].operands[1],
            ("vec", (("reg", "%r1"), ("imm", 0x40004040))),
        )
        call = _instrs(self.mma, "call.uni")[0]
        self.assertEqual(
            call.operands,
            (
                ("list", (("sym", "retval0#4"),)),
                ("sym", "vprintf"),
                ("list", (("sym", "param0#4"), ("sym", "param1#4"))),
            ),
        )
        self.assertEqual(
            [v.name for v in self.mma.vars], ["param0#4", "param1#4", "retval0#4"]
        )
        self.assertEqual(
            _one("selp.b32 %r1, 7, 017, !%p1").operands[1:],
            (("imm", 7), ("imm", 15), ("not", ("reg", "%p1"))),
        )

    def test_guards(self):
        self.assertEqual(_instrs(self.ws, "bra")[0].guard, ("%p1", True))  # @!%p1
        self.assertEqual(
            _instrs(self.ws, "mbarrier.init.shared::cta.b64")[0].guard, ("%p4", False)
        )
        self.assertIsNone(_instrs(self.ws, "shr.u32")[0].guard)

    def test_loc_chain_function_names_and_line_zero(self):
        frames = lambda i: [(f.file, f.line, f.col, f.func) for f in i.loc.chain()]  # noqa: E731
        src = "<source>/example.py"
        self.assertEqual(
            frames(_instrs(self.ws, "elect.sync")[0]),
            [(src, 20, 7, "helper"), (src, 50, 9, None)],
        )
        inner = [(src, 10, 3, "inner"), (src, 20, 7, "helper"), (src, 50, 9, None)]
        self.assertEqual(frames(_instrs(self.ws, "and.pred")[0]), inner)
        # `.loc 1 0 3` is no location: the next instructions keep the previous one.
        self.assertEqual(frames(_instrs(self.ws, "mov.b32")[0]), inner)
        self.assertEqual(_instrs(self.ws, "ld.param::entry.b32")[0].loc.line, 50)
        # Inside asm, the asm-internal .loc applies; NVVM's leading `.loc 1 0 0`: no loc.
        self.assertEqual(_instrs(self.ws, "ld.global.b32")[0].loc.line, 53)
        self.assertIsNone(self.mma.instructions()[0].loc)
        self.assertEqual(
            _instrs(self.mma, "mov.u32")[0].loc,
            parse.Loc("/src/example/mma_kernel.py", 12, 9),
        )

    def test_branchtargets_and_brx_idx(self):
        self.assertEqual(self.ws.branchtargets, {"JTI0_0": ("$L__BB0_4", "$L__BB0_5")})
        self.assertNotIn("JTI0_0", _labels(self.ws))  # a table, not a code position
        self.assertEqual(
            _instrs(self.ws, "brx.idx")[0].operands, (("reg", "%r8"), ("sym", "JTI0_0"))
        )
        self.assertEqual(self.ws.pragmas, [(76, "nounroll")])

    def test_syntax_errors_name_the_line(self):
        bad = [
            ("add.s32 %r9, %r1, 1", "undeclared register %r9"),
            (".reg .b32 %r1", "%r1 declared twice in one scope"),
            (".maxnreg 4", "unsupported declaration"),
            ("mov.b32 %r1, %r2 +", "unsupported operand"),
        ]
        for stmt, msg in bad:
            with (
                self.subTest(stmt=stmt),
                self.assertRaisesRegex(parse.PtxSyntaxError, f"^<ptx>:7: {msg}"),
            ):
                _one(stmt)
        with self.assertRaisesRegex(
            parse.PtxSyntaxError, "<ptx>:1: no size for type .pred"
        ):
            parse.parse(".visible .entry k(.param .pred p)\n{\nret;\n}\n")
        with self.assertRaisesRegex(parse.PtxSyntaxError, "undefined .file 2"):
            parse.parse(".visible .entry k()\n{\n.loc 2 1 1\nret;\n}\n")
        with self.assertRaisesRegex(
            parse.PtxSyntaxError, "unsupported function directive .pragma"
        ):
            parse.parse('.visible .entry k()\n.pragma "nounroll"\n{\nret;\n}\n')
        with self.assertRaisesRegex(parse.PtxSyntaxError, ":3: unterminated statement"):
            parse.parse(".visible .entry k()\n{\nret\n")


class OpsTest(unittest.TestCase):
    def test_spellings_meet_on_one_mnemonic(self):
        same = [
            (
                "mbarrier.arrive.release.cta.shared::cta.b64",
                "mbarrier.arrive.shared.b64",
                "mbarrier.arrive.shared::cta.b64",
            ),
            (
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64",
                "mbarrier.arrive.expect_tx.shared.b64",
            ),
            (
                "mbarrier.try_wait.parity.shared::cta.b64",
                "mbarrier.try_wait.parity.shared.b64",
            ),
            ("ld.global.u32", "ld.global.b32", "ld.global.v4.b32"),
            ("ld.param::entry.b32", "ld.param.u32", "ld.param.v2.u8"),
            ("setp.u32.eq", "setp.eq.s32", "setp.lt.u64"),
            (
                "tcgen05.cp.cta_group::1.warpx4.32x128b",
                "tcgen05.cp.cta_group::1.32x128b.warpx4",
            ),
            (
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X",
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16",
            ),
            (
                "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X",
                "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32",
            ),
            (
                "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint",
                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes",
            ),
            ("bra.uni", "bra"),
            ("add.rn.f32x2", "add.f32", "add.rm.ftz.f32"),
        ]
        for group in same:
            with self.subTest(group=group[0]):
                self.assertEqual(len({ops.mnemonic(o) for o in group}), 1)
        self.assertNotEqual(
            ops.mnemonic("mbarrier.arrive.relaxed.cluster.shared::cta.b64"),
            ops.mnemonic("mbarrier.arrive.shared.b64"),
        )
        self.assertNotEqual(ops.mnemonic("add.s32"), ops.mnemonic("add.f32"))

    def test_classes_follow_the_design(self):
        cases = {
            "setmaxnreg.dec.sync.aligned.u32 120": "drop",  # no warpgroup consistency check
            "ld.shared.b8 %r1, [%r2+4]": "drop",
            "st.shared.v4.b32 [%r1], {%r1, %r1, %r1, %r1}": "drop",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0, %r1, %r2, %r3}, [%r1]": "drop",
            "stmatrix.sync.aligned.m8n8.x1.shared.b16 [%r1], {%r2}": "drop",
            "add.f32 %r1, %r2, 0f3F800000": "drop",
            "cvt.rn.bf16x2.f32 %r1, %r2, %r3": "drop",
            "fence.proxy.async.shared::cta": "drop",
            "add.s32 %r1, %r2, 1": "value",
            "cvt.u64.u32 %rd1, %r1": "value",
            "mov.u32 %r1, %tid.x": "value",
            "ld.param.u32 %r1, [k_param_0]": "value",
            "elect.sync %r1|%p1, -1": "value",
            "ld.global.b32 %r1, [%rd1]": "event",
            "mbarrier.init.shared::cta.b64 [%r1], 1": "event",
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %p1, %rd1": "event",
            "barrier.sync 1": "event",
            "bra $L1": "control",
            "brx.idx %r1, T": "control",
            "ret": "control",
            "mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r1], %r2": "refuse",
            "mov.u32 %r1, %warpid": "refuse",
            "cp.async.bulk.tensor.2d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%r1], [%rd1, {%r1, %r2}], [%r1], {%r3}": "refuse",
            "call.uni __assertfail, (%rd1)": "event",
            "call.uni (%r1), vprintf, (%rd1, %rd1)": "drop",
            "call.uni (%r1), __nv_expf, (%r2)": "drop",
            "call.uni (%r1), my_helper, (%r2)": "refuse",
        }
        for stmt, cls in cases.items():
            with self.subTest(stmt=stmt):
                self.assertEqual(ops.classify(_one(stmt))[0], cls)

    def test_table_rows_are_normalized_and_disjoint(self):
        keys = [k for _, _, ks in ops._ROWS for k in ks]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual([k for k in keys if ops.mnemonic(k) != k], [])

    def test_unknown_mnemonic_raises(self):
        with self.assertRaisesRegex(
            ops.UnknownInstruction, r"no op table row for mad\.hi"
        ):
            ops.classify(_one("mad.hi.s32 %r1, %r2, %r3, %r1"))
        with self.assertRaisesRegex(ops.UnknownInstruction, "ld.global.nc"):
            ops.classify(_one("ld.global.nc.b32 %r1, [%rd1]"))

    def test_every_fixture_instruction_is_classified(self):
        for path in (LLVM, NVVM):
            for fn in _fixture(path).functions:
                for i in fn.instructions():
                    self.assertIn(ops.classify(i)[0], ops.CLASSES)


class CoverageTest(unittest.TestCase):
    def test_fixtures_pass_and_the_json_adds_up(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "cov.json"
            self.assertEqual(_coverage(LLVM, NVVM, "--json", out), 0)
            report = json.loads(out.read_text())
        files, total = report["files"], report["total"]
        self.assertEqual(total["instructions"], sum(f["instructions"] for f in files))
        self.assertEqual(total["instructions"], sum(total["classes"].values()))
        self.assertEqual((total["errors"], total["unknown"]), (0, {}))
        self.assertEqual(
            total["mnemonics"]["mbarrier.try_wait.parity.shared"], {"refuse": 4}
        )
        self.assertEqual(total["mnemonics"]["call"], {"drop": 1})

    def test_unknown_mnemonics_and_parse_errors_fail(self):
        head = ".version 9.3\n.target sm_100a\n.address_size 64\n.visible .entry k()\n{\n.reg .b32 %r<2>;\n"
        with tempfile.TemporaryDirectory() as d:
            unknown, broken = Path(d) / "unknown.ptx", Path(d) / "broken.ptx"
            unknown.write_text(head + "mad.hi.s32 %r1, %r1, %r1, %r1;\nret;\n}\n")
            broken.write_text(head + "add.s32 %r7, %r1, 1;\n}\n")
            out = Path(d) / "cov.json"
            self.assertEqual(_coverage(unknown, broken, "--json", out), 1)
            report = json.loads(out.read_text())
        u, b = report["files"]
        self.assertEqual(
            u["unknown"], {"mad.hi": {"count": 1, "line": 7, "opcode": "mad.hi.s32"}}
        )
        self.assertEqual(u["classes"], {"unknown": 1, "control": 1})
        self.assertIn("undeclared register %r7", b["error"])


if __name__ == "__main__":
    unittest.main()
