import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from slothctf import execution as ex
from slothctf.execution import (
    DockerLinuxBackend,
    ExecResult,
    NativeBackend,
    ResourceLimits,
    SandboxUnavailable,
    build_docker_run_argv,
    get_backend,
    resolve_backend,
)
from slothctf.targets import ELF, PE, get_target, host_native_target

_HAS_GCC = shutil.which("gcc") is not None


class DockerArgvTests(unittest.TestCase):
    """The isolation flags are the whole point; assert them without a daemon."""

    def _argv(self, **kw):
        return build_docker_run_argv(
            "img", ["gcc", "-o", "/work/a", "/work/a.c"], Path("."),
            ResourceLimits(cpus=2.0, memory="256m", pids=99), "cname", **kw)

    def test_isolation_flags_present(self):
        a = " ".join(self._argv())
        self.assertIn("--network none", a)
        self.assertIn("--user 65534:65534", a)
        self.assertIn("--cap-drop ALL", a)
        self.assertIn("no-new-privileges", a)
        self.assertIn("--read-only", a)
        self.assertIn("--tmpfs", a)
        self.assertIn("--pids-limit 99", a)
        self.assertIn("--memory 256m", a)
        self.assertIn("--cpus 2.0", a)
        self.assertIn("--rm", a)
        self.assertIn("--name cname", a)

    def test_workspace_mounted_rw_and_inner_last(self):
        argv = self._argv()
        mount = [argv[i + 1] for i, t in enumerate(argv) if t == "-v"][0]
        self.assertTrue(mount.endswith(":/work:rw"))
        # the image precedes the inner command, which is last
        self.assertEqual(argv[-4:], ["gcc", "-o", "/work/a", "/work/a.c"])
        self.assertIn("img", argv)

    def test_stdin_flag_only_when_requested(self):
        self.assertNotIn("-i", self._argv())
        self.assertIn("-i", self._argv(stdin=True))


class FactoryTests(unittest.TestCase):
    def test_default_is_docker_and_raises_without_fallback(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
             mock.patch.object(ex, "docker_available", return_value=False):
            with self.assertRaises(SandboxUnavailable):
                get_backend()  # default docker, unavailable -> explicit error, NO native fallback

    def test_docker_selected_when_available(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
             mock.patch.object(ex, "docker_available", return_value=True):
            self.assertIsInstance(get_backend(), DockerLinuxBackend)

    def test_native_requires_opt_in(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SandboxUnavailable):
                get_backend("native")
            self.assertIsInstance(get_backend("native", allow_native=True), NativeBackend)

    def test_native_opt_in_via_env(self):
        with mock.patch.dict("os.environ", {"SLOTHCTF_ALLOW_NATIVE": "1"}, clear=True):
            self.assertIsInstance(get_backend("native"), NativeBackend)

    def test_native_backend_is_not_safe(self):
        self.assertFalse(NativeBackend().safe)
        self.assertTrue(DockerLinuxBackend().safe)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            get_backend("vm")


class ResolveBackendTests(unittest.TestCase):
    """The (target, isolation) resolver: both targets first-class, no silent fallback."""

    def test_isolated_elf_needs_docker(self):
        with mock.patch.object(ex, "docker_available", return_value=True):
            self.assertIsInstance(resolve_backend("elf", "isolated"), DockerLinuxBackend)
        with mock.patch.object(ex, "docker_available", return_value=False):
            with self.assertRaises(SandboxUnavailable):
                resolve_backend("elf", "isolated")  # explicit error, no native fallback

    def test_isolated_pe_resolves_to_wincompat(self):
        # Phase B: pe + isolated -> DockerWinCompatBackend when Docker is up and the
        # Windows-compat image is built.
        from slothctf.execution import DockerWinCompatBackend
        with mock.patch.object(ex, "docker_available", return_value=True), \
             mock.patch.object(ex, "win_image_available", return_value=True):
            self.assertIsInstance(resolve_backend("pe", "isolated"), DockerWinCompatBackend)

    def test_isolated_pe_without_docker_errors(self):
        with mock.patch.object(ex, "docker_available", return_value=False):
            with self.assertRaises(SandboxUnavailable):
                resolve_backend("pe", "isolated")  # no fallback to native

    def test_isolated_pe_without_win_image_errors(self):
        # Docker up but the MinGW+Wine image not built -> explicit error, no silent
        # unsandboxed build and no PE->ELF conversion.
        with mock.patch.object(ex, "docker_available", return_value=True), \
             mock.patch.object(ex, "win_image_available", return_value=False):
            with self.assertRaises(SandboxUnavailable):
                resolve_backend("pe", "isolated")

    def test_unsafe_native_requires_opt_in(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SandboxUnavailable):
                resolve_backend(host_native_target(), "unsafe-native")
            b = resolve_backend(host_native_target(), "unsafe-native", allow_native=True)
            self.assertIsInstance(b, NativeBackend)

    def test_native_cannot_build_foreign_target(self):
        # the non-host-native target cannot be built natively here -> explicit error
        foreign = ELF if host_native_target().name == "pe" else PE
        with self.assertRaises(SandboxUnavailable):
            resolve_backend(foreign, "unsafe-native", allow_native=True)

    def test_unknown_isolation(self):
        with self.assertRaises(ValueError):
            resolve_backend("elf", "sorta-isolated")


class TargetSupportTests(unittest.TestCase):
    def test_docker_linux_supports_only_elf(self):
        b = DockerLinuxBackend()
        self.assertTrue(b.can_build(ELF))
        self.assertFalse(b.can_build(PE))

    def test_native_supports_host_target_only(self):
        b = NativeBackend()
        self.assertIn(host_native_target().name, b.supported_targets)
        self.assertEqual(len(b.supported_targets), 1)

    def test_docker_refuses_pe_compile(self):
        b = DockerLinuxBackend()
        with self.assertRaises(SandboxUnavailable):
            b.compile_c(Path("."), "src/c.c", "bin/c.exe", PE)


class DockerInnerCommandTests(unittest.TestCase):
    """Container paths are /work-relative; verify the inner argv the backend runs."""

    def test_compile_maps_paths_into_work(self):
        b = DockerLinuxBackend()
        with TemporaryDirectory() as tmp, \
             mock.patch.object(b, "_exec", return_value=ExecResult(True, "", "")) as m:
            b.compile_c(Path(tmp), "src/challenge.c", "bin/challenge", ELF)
            inner = m.call_args.args[0]
            self.assertEqual(inner, ["gcc", "-O2", "-s", "-o", "/work/bin/challenge",
                                     "/work/src/challenge.c"])

    def test_run_python_uses_python3_isolated(self):
        b = DockerLinuxBackend()
        with TemporaryDirectory() as tmp, \
             mock.patch.object(b, "_exec", return_value=ExecResult(True, "", "")) as m:
            b.run_python(Path(tmp), "organizer/extractor_1.py", ["/work/evidence/x.log"])
            inner = m.call_args.args[0]
            self.assertEqual(inner[:3], ["python3", "-I", "/work/organizer/extractor_1.py"])


class TruncateTests(unittest.TestCase):
    def test_output_capped(self):
        out, trunc = ex._truncate("x" * 100, 10)
        self.assertTrue(trunc)
        self.assertTrue(out.startswith("x" * 10))
        self.assertIn("truncated", out)

    def test_under_limit_untouched(self):
        self.assertEqual(ex._truncate("short", 100), ("short", False))


@unittest.skipUnless(_HAS_GCC, "needs a C compiler")
class NativeBackendRealTests(unittest.TestCase):
    def test_compile_and_run(self):
        b = NativeBackend()
        with TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "src").mkdir()
            (ws / "src" / "c.c").write_text(
                '#include <stdio.h>\nint main(){printf("sloth{ok}");return 0;}',
                encoding="utf-8")
            tgt = host_native_target()
            out_rel = f"bin/{tgt.out_name}"
            c = b.compile_c(ws, "src/c.c", out_rel, tgt)
            self.assertTrue(c.ok, c.stderr)
            r = b.run_binary(ws, out_rel, tgt)
            self.assertTrue(r.ok)
            self.assertIn("sloth{ok}", r.stdout)


_ELF_READY = ex.docker_available() and ex.image_available()
_PE_READY = ex.docker_available() and ex.win_image_available()
_TRIVIAL_C = '#include <stdio.h>\nint main(){printf("hello");return 0;}\n'


@unittest.skipUnless(_ELF_READY, "needs Docker + the ELF sandbox image")
class StaticAnalysisElfTests(unittest.TestCase):
    def test_static_analysis_is_bounded_and_static(self):
        b = DockerLinuxBackend()
        with TemporaryDirectory() as tmp:
            ws = Path(tmp); (ws / "src").mkdir()
            (ws / "src" / "c.c").write_text(_TRIVIAL_C, encoding="utf-8")
            self.assertTrue(b.compile_c(ws, "src/c.c", "bin/challenge", get_target("elf")).ok)
            ev = b.static_analysis(ws, "bin/challenge", get_target("elf"))
        self.assertIn("=== sections", ev)
        self.assertIn("=== strings", ev)
        self.assertIn("=== disassembly", ev)
        # bounded to the (rebalanced) context budget: 600+700+500+6000 nominal max
        self.assertLessEqual(len(ev), 8500)


@unittest.skipUnless(_PE_READY, "needs Docker + the Windows-compat sandbox image")
class StaticAnalysisPeTests(unittest.TestCase):
    def test_static_analysis_pe_uses_mingw_tools(self):
        from slothctf.execution import DockerWinCompatBackend
        b = DockerWinCompatBackend()
        with TemporaryDirectory() as tmp:
            ws = Path(tmp); (ws / "src").mkdir()
            (ws / "src" / "c.c").write_text(_TRIVIAL_C, encoding="utf-8")
            self.assertTrue(b.compile_c(ws, "src/c.c", "bin/challenge.exe", get_target("pe")).ok)
            ev = b.static_analysis(ws, "bin/challenge.exe", get_target("pe"))
        self.assertIn("=== sections", ev)
        self.assertIn("=== disassembly", ev)
        self.assertLessEqual(len(ev), 8500)


# --------------------------------------------------------------------------- #
# Disassembly SELECTION (ex._compact_disassembly / _entry_point_from_objdump_f /
# _clip_lines_then_chars): deterministic, symbol-independent boilerplate removal
# so the solver sees application instructions, not ELF/PLT/CRT startup noise.
# Pure-Python unit tests first (hand-built objdump-shaped text, no Docker needed),
# then Docker integration tests against real compiled binaries.
# --------------------------------------------------------------------------- #
_SAMPLE_OBJDUMP_D = (
    "Disassembly of section .plt:\n\n"
    "0000000000001020 <puts@plt-0x10>:\n"
    "    1020:\tpush   QWORD PTR [rip+0x2fca]\n"
    "    1026:\tjmp    QWORD PTR [rip+0x2fcc]\n\n"
    "0000000000001030 <puts@plt>:\n"
    "    1030:\tjmp    QWORD PTR [rip+0x2fca]\n"
    "    1036:\tpush   0x0\n\n"
    "Disassembly of section .plt.got:\n\n"
    "0000000000001060 <__cxa_finalize@plt>:\n"
    "    1060:\tjmp    QWORD PTR [rip+0x2f7a]\n\n"
    "Disassembly of section .text:\n\n"
    "0000000000001100 <.text>:\n"
    "    1100:\txor    ebp,ebp\n"
    "    1102:\tmov    r9,rdx\n"
    "    1104:\thlt\n"
    "    1105:\tpush   rbx\n"
    "    1106:\txor    ebx,ebx\n"
    "    1107:\tcall   1030 <puts@plt>\n"
    "    110c:\tret\n"
)


class DisassemblyCompactionTests(unittest.TestCase):
    """Pure-Python: ex._compact_disassembly / _entry_point_from_objdump_f /
    _clip_lines_then_chars, against hand-built objdump-shaped text."""

    def test_entry_point_parsed_from_objdump_f(self):
        text = ("\nfile.exe:     file format elf64-x86-64\n"
                "architecture: i386:x86-64, flags 0x00000150:\n"
                "start address 0x0000000000001100\n")
        self.assertEqual(ex._entry_point_from_objdump_f(text), 0x1100)

    def test_entry_point_missing_returns_none(self):
        self.assertIsNone(ex._entry_point_from_objdump_f("no such field here"))

    def test_plt_family_sections_compacted_to_one_line_index(self):
        compacted, notes = ex._compact_disassembly(_SAMPLE_OBJDUMP_D, entry_point=0x1100)
        self.assertNotIn("push   QWORD PTR", compacted)   # trampoline body gone
        self.assertIn("puts@plt-0x10", compacted)          # but the entry is still named
        self.assertIn("puts@plt>", compacted)
        self.assertIn("__cxa_finalize@plt", compacted)
        self.assertTrue(any(".plt" in n for n in notes))
        self.assertTrue(any(".plt.got" in n for n in notes))

    def test_entry_point_block_compacted_up_to_hlt(self):
        compacted, notes = ex._compact_disassembly(_SAMPLE_OBJDUMP_D, entry_point=0x1100)
        # the 3 instructions at/after the entry point, up to and including hlt, are gone...
        self.assertNotIn("1100:\txor    ebp,ebp", compacted)
        self.assertNotIn("1104:\thlt", compacted)
        # ...but application code AFTER that point is fully preserved
        self.assertIn("1105:\tpush   rbx", compacted)
        self.assertIn("1106:\txor    ebx,ebx", compacted)
        self.assertIn("1107:\tcall   1030 <puts@plt>", compacted)
        self.assertIn("110c:\tret", compacted)
        self.assertTrue(any("entry point" in n for n in notes))

    def test_no_entry_point_leaves_everything_untouched_except_plt(self):
        # safer to show too much than to guess and hide application code
        compacted, notes = ex._compact_disassembly(_SAMPLE_OBJDUMP_D, entry_point=None)
        self.assertIn("1100:\txor    ebp,ebp", compacted)
        self.assertIn("1104:\thlt", compacted)
        self.assertFalse(any("entry point" in n for n in notes))

    def test_entry_point_outside_the_dump_is_a_noop(self):
        compacted, notes = ex._compact_disassembly(_SAMPLE_OBJDUMP_D, entry_point=0xdeadbeef)
        self.assertIn("1100:\txor    ebp,ebp", compacted)
        self.assertFalse(any("entry point" in n for n in notes))

    def test_addresses_mnemonics_operands_and_relocation_comments_pass_through_unchanged(self):
        raw = ("Disassembly of section .text:\n\n0000000000002000 <.text>:\n"
              "    2000:\tmov    rax,QWORD PTR [rip+0x10]        # 3010 <stdin@GLIBC_2.2.5>\n")
        compacted, _ = ex._compact_disassembly(raw, entry_point=None)
        self.assertIn("2000:\tmov    rax,QWORD PTR [rip+0x10]        # 3010 <stdin@GLIBC_2.2.5>",
                      compacted)

    def test_clip_never_cuts_a_line_in_half(self):
        line_len = len("line000 " + "x" * 20)   # 28 chars
        text = "\n".join(f"line{i:03d} " + "x" * 20 for i in range(50))
        clipped = ex._clip_lines_then_chars(text, "disassembly", max_lines=100, max_chars=137)
        for ln in clipped.splitlines():
            if ln.startswith("line"):
                self.assertEqual(len(ln), line_len)   # every kept content line is whole, never sliced
        self.assertIn("truncated to fit 137 chars", clipped)

    def test_clip_line_budget_reports_omitted_count(self):
        text = "\n".join(f"l{i}" for i in range(10))
        clipped = ex._clip_lines_then_chars(text, "sections", max_lines=4, max_chars=10_000)
        self.assertIn("6 more lines truncated", clipped)

    def test_clip_under_budget_is_unchanged(self):
        text = "a\nb\nc"
        self.assertEqual(ex._clip_lines_then_chars(text, "x", max_lines=10, max_chars=100), text)


_FC_TRANSFORM_FOR_SELECTION = (
    "void slothctf_transform(const unsigned char *in, size_t n, unsigned char *out) {\n"
    "    unsigned char s = 0x5a;\n"
    "    for (size_t i = 0; i < n; i++) {\n"
    "        unsigned char v = (unsigned char)(in[i] ^ s);\n"
    "        out[i] = v;\n"
    "        s = (unsigned char)(s + in[i] + (unsigned char)i);\n"
    "    }\n"
    "}\n")
_FC_CHALLENGE_C_FOR_SELECTION = (
    "#include <stdio.h>\n#include <string.h>\n#include <stddef.h>\n"
    + _FC_TRANSFORM_FOR_SELECTION +
    "static const unsigned char ENC[] = {0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,"
    "0x0b,0x0c,0x0d,0x0e,0x0f,0x10,0x11,0x12,0x13};\n"
    "#define FLAG_LEN (sizeof(ENC))\n"
    "int main(void){\n"
    "  unsigned char buf[FLAG_LEN]; size_t n=0; int c;\n"
    "  while((c=fgetc(stdin))!=EOF && c!='\\n'){ if(n<FLAG_LEN) buf[n]=(unsigned char)c; n++; }\n"
    "  if(n!=FLAG_LEN){ printf(\"WRONG_INPUT\\n\"); return 0; }\n"
    "  unsigned char computed[FLAG_LEN]; slothctf_transform(buf, FLAG_LEN, computed);\n"
    "  if(memcmp(computed, ENC, FLAG_LEN)==0) printf(\"FLAG_RECOVERED\\n\");\n"
    "  else printf(\"WRONG_INPUT\\n\");\n"
    "  return 0;\n}\n")
# A tiny inline-friendly variant: the transform is ONE line, trivial enough that gcc -O2
# may choose to inline it directly into main() instead of emitting a separate function —
# the selection logic must not assume either shape.
_INLINE_FRIENDLY_TRANSFORM = (
    "static inline void slothctf_transform(const unsigned char *in, size_t n, unsigned char *out) {\n"
    "    for (size_t i = 0; i < n; i++) out[i] = (unsigned char)(in[i] ^ 0x5a);\n"
    "}\n")
_INLINE_FRIENDLY_CHALLENGE_C = (
    "#include <stdio.h>\n#include <string.h>\n#include <stddef.h>\n"
    + _INLINE_FRIENDLY_TRANSFORM +
    "static const unsigned char ENC[] = {0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a};\n"
    "#define FLAG_LEN (sizeof(ENC))\n"
    "int main(void){\n"
    "  unsigned char buf[FLAG_LEN]; size_t n=0; int c;\n"
    "  while((c=fgetc(stdin))!=EOF && c!='\\n'){ if(n<FLAG_LEN) buf[n]=(unsigned char)c; n++; }\n"
    "  if(n!=FLAG_LEN){ printf(\"WRONG_INPUT\\n\"); return 0; }\n"
    "  unsigned char computed[FLAG_LEN]; slothctf_transform(buf, FLAG_LEN, computed);\n"
    "  if(memcmp(computed, ENC, FLAG_LEN)==0) printf(\"FLAG_RECOVERED\\n\");\n"
    "  else printf(\"WRONG_INPUT\\n\");\n"
    "  return 0;\n}\n")
# Deliberately pulls in many extra PLT imports to stress-test trampoline compaction.
_LONG_BOILERPLATE_C = (
    "#include <stdio.h>\n#include <string.h>\n#include <stdlib.h>\n#include <ctype.h>\n"
    "int main(int argc, char **argv){\n"
    "  char b[64]; if(!fgets(b,sizeof b,stdin)) return 0;\n"
    "  size_t n = strlen(b); if(n && b[n-1]=='\\n') b[--n]=0;\n"
    "  if(strcmp(b,\"x\")==0) puts(\"eq\");\n"
    "  if(strncmp(b,\"y\",1)==0) puts(\"pre\");\n"
    "  if(isalpha((unsigned char)b[0])) puts(\"alpha\");\n"
    "  printf(\"%d %s %ld\\n\", (int)n, b, strtol(b,NULL,10));\n"
    "  qsort(b, n, 1, (int(*)(const void*,const void*))strcmp);\n"
    "  return 0;\n}\n")


def _assemble(binrel, c_src, backend, tgt, workspace, *, stripped=True):
    (workspace / "src").mkdir(exist_ok=True); (workspace / "bin").mkdir(exist_ok=True)
    (workspace / "src" / "c.c").write_text(c_src, encoding="utf-8")
    if stripped:
        res = backend.compile_c(workspace, "src/c.c", binrel, tgt)
    else:
        # bypass compile_c's hardcoded -s so a REAL symbol table is present — the
        # selection logic must not rely on it either way.
        from pathlib import PurePosixPath
        src = PurePosixPath("/work") / "src" / "c.c"
        out = PurePosixPath("/work") / binrel
        res = backend._exec(["gcc", "-O2", "-o", str(out), str(src)], workspace, 60.0, ResourceLimits())
    return res


@unittest.skipUnless(_ELF_READY, "needs Docker + the ELF sandbox image")
class DisassemblySelectionElfDockerTests(unittest.TestCase):
    """Real compiles: stripped/non-stripped, inline/separate transform, long
    boilerplate, truncation, and a real citation still verifying against the new
    (--no-show-raw-insn) format — against ACTUAL binaries, not hand-built text."""

    def _evidence(self, c_src, *, stripped=True, binrel="bin/challenge"):
        b = DockerLinuxBackend()
        with TemporaryDirectory() as tmp:
            ws = Path(tmp)
            res = _assemble(binrel, c_src, b, get_target("elf"), ws, stripped=stripped)
            self.assertTrue(res.ok, res.stderr)
            return b.static_analysis(ws, binrel, get_target("elf"))

    def test_stripped_separate_transform_function(self):
        ev = self._evidence(_FC_CHALLENGE_C_FOR_SELECTION, stripped=True)
        idx = ev.find("=== disassembly")
        self.assertNotEqual(idx, -1)
        disasm = ev[idx:]
        self.assertIn("xor", disasm.lower())
        self.assertIn("0x13", disasm)              # FLAG_LEN=19=0x13, as a real immediate
        self.assertNotIn("endbr64\n    ", disasm[:300])  # entry-point block was compacted away

    def test_non_stripped_real_symbols_do_not_break_selection(self):
        ev = self._evidence(_FC_CHALLENGE_C_FOR_SELECTION, stripped=False)
        idx = ev.find("=== disassembly")
        disasm = ev[idx:]
        self.assertIn("<main>", disasm)                    # a real symbol now exists...
        self.assertIn("<slothctf_transform>", disasm)       # ...for both functions
        self.assertIn("xor", disasm.lower())
        self.assertIn("0x13", disasm)

    def test_inline_friendly_transform_still_fully_visible(self):
        ev = self._evidence(_INLINE_FRIENDLY_CHALLENGE_C, stripped=True)
        idx = ev.find("=== disassembly")
        disasm = ev[idx:]
        self.assertIn("xor", disasm.lower())
        self.assertIn("0x5a", disasm.lower())   # the inline-friendly XOR key, 0x5a
        self.assertIn("0xa", disasm.lower())    # FLAG_LEN=10=0xa, as a real immediate

    def test_long_boilerplate_plt_is_compacted_not_shown_verbatim(self):
        ev = self._evidence(_LONG_BOILERPLATE_C, stripped=True, binrel="bin/boiler")
        idx = ev.find("=== disassembly")
        disasm = ev[idx:]
        self.assertIn("import-trampoline entries", disasm)
        # the harness's own extra imports (strtol, qsort, isalpha, strncmp...) are named
        # in the compacted index, not shown with their full (boilerplate) trampoline body
        self.assertIn("strtol@plt", disasm)
        self.assertNotIn("QWORD PTR [rip+", disasm[disasm.find("import-trampoline"):
                                                    disasm.find("import-trampoline") + 400])

    def test_citation_against_the_new_format_still_verifies(self):
        import slothctf.agentic as ag
        ev = self._evidence(_FC_CHALLENGE_C_FOR_SELECTION, stripped=True)
        lines = ag._parse_disasm_lines(ev)
        self.assertTrue(lines, "no instruction lines parsed from the new evidence format")
        addr, text = next(iter(lines.items()))
        insn = text.split("\t")[-1].strip()
        payload = {"analysis": {"evidence": [{"addr": addr, "insn": insn, "claim": "c"}],
                                "insufficient_evidence": False}, "solver_code": "print(1)"}

        class FS:
            binary_evidence = ev
        self.assertIsNone(ag._check_solver_evidence(payload, FS()))
        # a fabricated address must still be rejected against the new format
        bad = {"analysis": {"evidence": [{"addr": "0xdeadbeef", "insn": insn, "claim": "c"}],
                            "insufficient_evidence": False}, "solver_code": "print(1)"}
        self.assertIsNotNone(ag._check_solver_evidence(bad, FS()))

    def test_tight_budget_truncates_on_whole_lines_and_reports_it(self):
        b = DockerLinuxBackend()
        with TemporaryDirectory() as tmp:
            ws = Path(tmp)
            res = _assemble("bin/challenge", _FC_CHALLENGE_C_FOR_SELECTION, b, get_target("elf"), ws)
            self.assertTrue(res.ok, res.stderr)
            ev = b._disassembly_evidence("/work/bin/challenge", "", ws, ResourceLimits())
        # the method uses its own fixed budget; verify the GENERAL truncation helper
        # directly with a tiny budget against this run's real compacted text instead
        idx = ev.find("=== disassembly")
        real_compacted_disasm = ev[idx:]
        tight = ex._clip_lines_then_chars(real_compacted_disasm, "disassembly",
                                          max_lines=1000, max_chars=300)
        for ln in tight.splitlines():
            self.assertFalse(ln.startswith("    ") and len(ln) > 200, ln)  # no absurdly long partial line
        self.assertIn("truncated to fit 300 chars", tight)
        # no mid-line cut: every content line kept is a complete, valid line from the original
        kept_lines = [l for l in tight.splitlines() if not l.startswith("[...")]
        for l in kept_lines:
            if l.strip() and not l.startswith("["):
                self.assertIn(l, real_compacted_disasm)


@unittest.skipUnless(_PE_READY, "needs Docker + the Windows-compat sandbox image")
class DisassemblySelectionPeDockerTests(unittest.TestCase):
    """Same proof on PE (MinGW, with its own CRT startup) — the selection logic must
    generalize, not just happen to work for ELF's smaller dynamic-linking glue."""

    def test_entry_point_is_detected_on_pe_too(self):
        # the entry-point detection mechanism itself (objdump -f parsing) works
        # identically on PE -- this is NOT the same claim as "the transform is visible"
        # (see the documented limitation below).
        from slothctf.execution import DockerWinCompatBackend
        b = DockerWinCompatBackend()
        with TemporaryDirectory() as tmp:
            ws = Path(tmp)
            res = _assemble("bin/challenge.exe", _FC_CHALLENGE_C_FOR_SELECTION, b,
                            get_target("pe"), ws)
            self.assertTrue(res.ok, res.stderr)
            fr = b._run_tool(["x86_64-w64-mingw32-objdump", "-f", "/work/bin/challenge.exe"],
                             ws, 30.0, ResourceLimits())
        self.assertIsNotNone(ex._entry_point_from_objdump_f(fr.stdout))

    def test_KNOWN_LIMITATION_static_mingw_crt_can_still_bury_the_transform(self):
        """Documents a known, UNFIXED limitation (same pattern as
        ForwardCheckCollisionRiskTests): our PE builds use `-static -pthread`
        (DockerWinCompatBackend.compile_c; needed so the .exe has no missing-DLL
        problem under Wine). A statically-linked MinGW CRT is far larger than ELF's
        dynamic-linking glue -- thousands of lines of locale/exception/init machinery
        -- and _compact_disassembly only removes the entry-point block and PLT-style
        import trampolines (ELF has none to remove here, since everything is linked
        in). For a trivial transform this can still leave the actual application code
        past the 6000-char disassembly budget, confirmed below. This is not claimed
        to be solved here; building PE dynamically (no `-static`) would likely give it
        the same small, PLT-like import-thunk shape ELF has, but changes the Wine
        compatibility trade-off `-static` was added for, so it is a build-configuration
        decision outside this fix's scope, documented rather than changed."""
        from slothctf.execution import DockerWinCompatBackend
        b = DockerWinCompatBackend()
        with TemporaryDirectory() as tmp:
            ws = Path(tmp)
            res = _assemble("bin/challenge.exe", _FC_CHALLENGE_C_FOR_SELECTION, b,
                            get_target("pe"), ws)
            self.assertTrue(res.ok, res.stderr)
            ev = b.static_analysis(ws, "bin/challenge.exe", get_target("pe"))
        idx = ev.find("=== disassembly")
        disasm = ev[idx:]
        self.assertIn("more lines truncated", disasm)   # the CRT dump itself is this large
        self.assertNotIn("0x13", disasm, "if this starts failing, the limitation may be "
                         "resolved -- re-check before deleting this test")

    def test_pe_citation_against_the_new_format_still_verifies(self):
        import slothctf.agentic as ag
        from slothctf.execution import DockerWinCompatBackend
        b = DockerWinCompatBackend()
        with TemporaryDirectory() as tmp:
            ws = Path(tmp)
            res = _assemble("bin/challenge.exe", _FC_CHALLENGE_C_FOR_SELECTION, b,
                            get_target("pe"), ws)
            self.assertTrue(res.ok, res.stderr)
            ev = b.static_analysis(ws, "bin/challenge.exe", get_target("pe"))
        lines = ag._parse_disasm_lines(ev)
        self.assertTrue(lines)
        addr, text = next(iter(lines.items()))
        insn = text.split("\t")[-1].strip()
        payload = {"analysis": {"evidence": [{"addr": addr, "insn": insn, "claim": "c"}],
                                "insufficient_evidence": False}, "solver_code": "print(1)"}

        class FS:
            binary_evidence = ev
        self.assertIsNone(ag._check_solver_evidence(payload, FS()))


class NoSecretLeakageTests(unittest.TestCase):
    """The selection functions only ever process the PUBLIC disassembly text of the
    DISTRIBUTED binary — they take no flag, source, or organizer data as input at all,
    so there is nothing for them to leak. Documented structurally (signature-level),
    not re-proving the existing flag-leak check."""

    def test_compact_disassembly_signature_takes_no_organizer_data(self):
        import inspect
        params = list(inspect.signature(ex._compact_disassembly).parameters)
        self.assertEqual(params, ["raw", "entry_point"])

    def test_static_analysis_signature_takes_no_flag_or_source(self):
        import inspect
        params = list(inspect.signature(ex.ExecutionBackend.static_analysis).parameters)
        self.assertEqual(params, ["self", "workspace", "binary_rel", "target", "limits"])


# --------------------------------------------------------------------------- #
# Validates the exact technique Stage D's prompt now points the solver at:
# `objdump -h` for section (file-offset, size), then a PLAIN file read/slice —
# no new tool, no new container; objdump/strings are already on PATH in the
# same sandbox run_solver() uses. These tests exercise ONLY the mechanism (the
# harness's own use of the tool), not whether an LLM-written solver gets it
# right — see tests/test_agentic.py for an end-to-end fixture doing that.
# --------------------------------------------------------------------------- #
import re as _re  # noqa: E402  (test-local, kept near its single use below)

_DATA_C = (
    '#include <stdio.h>\n#include <string.h>\n'
    'static const unsigned char DATA[] = {0xde,0xad,0xbe,0xef,0x11,0x22,0x33,0x44,0x55,0x66};\n'
    'int main(void){\n'
    '  char b[256];\n'
    '  if(!fgets(b,sizeof b,stdin)) return 0;\n'
    '  if(strlen(b)==sizeof(DATA) && memcmp(b,DATA,sizeof DATA)==0) printf("Correct\\n");\n'
    '  else printf("No\\n");\n'
    '  return 0;\n'
    '}\n'
)
_NEEDLE = bytes([0xde, 0xad, 0xbe, 0xef])


def _parse_objdump_sections(objdump_h_text):
    """Exactly the parsing shown in the Stage-D prompt guidance (agentic._SOLVER_TOOLING_HINT)."""
    out = {}
    for line in objdump_h_text.splitlines():
        m = _re.match(r"^\s*\d+\s+(\S+)\s+([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)", line)
        if m:
            out[m.group(1)] = (int(m.group(3), 16), int(m.group(2), 16))  # name -> (fileoff, size)
    return out


@unittest.skipUnless(_ELF_READY, "needs Docker + the ELF sandbox image")
class SolverSectionLookupElfTests(unittest.TestCase):
    def _build(self, ws, binrel="bin/t"):
        b = DockerLinuxBackend()
        (ws / "src").mkdir(); (ws / "bin").mkdir()
        (ws / "src" / "c.c").write_text(_DATA_C, encoding="utf-8")
        self.assertTrue(b.compile_c(ws, "src/c.c", binrel, get_target("elf")).ok)
        return b, binrel

    def test_section_offset_and_raw_slice_locate_real_bytes(self):
        with TemporaryDirectory() as tmp:
            ws = Path(tmp); b, binrel = self._build(ws)
            h = b._run_tool(["objdump", "-h", f"/work/{binrel}"], ws, 30.0, ResourceLimits())
            sections = _parse_objdump_sections(h.stdout)
            raw = (ws / binrel).read_bytes()
            real_off = raw.find(_NEEDLE)
            self.assertNotEqual(real_off, -1)
            hit = next((n for n, (off, size) in sections.items() if off <= real_off < off + size), None)
            self.assertIsNotNone(hit, sections)
            self.assertIn(hit, (".rodata", ".data"))

    def test_missing_section_yields_no_match_not_a_crash(self):
        with TemporaryDirectory() as tmp:
            ws = Path(tmp); b, binrel = self._build(ws)
            h = b._run_tool(["objdump", "-h", "-j", ".this_section_does_not_exist", f"/work/{binrel}"],
                            ws, 30.0, ResourceLimits())
            sections = _parse_objdump_sections(h.stdout)
            self.assertEqual(sections, {})   # empty, not an exception

    def test_malformed_binary_handled_gracefully(self):
        b = DockerLinuxBackend()
        with TemporaryDirectory() as tmp:
            ws = Path(tmp); (ws / "bin").mkdir()
            (ws / "bin" / "junk").write_bytes(b"not an ELF or PE file at all, just garbage bytes")
            h = b._run_tool(["objdump", "-h", "/work/bin/junk"], ws, 30.0, ResourceLimits())
            # objdump reports an error (non-zero / stderr), never hangs or crashes the caller
            sections = _parse_objdump_sections(h.stdout)
            self.assertEqual(sections, {})
            self.assertFalse(h.timed_out)


@unittest.skipUnless(_PE_READY, "needs Docker + the Windows-compat sandbox image")
class SolverSectionLookupPeTests(unittest.TestCase):
    def test_section_offset_and_raw_slice_locate_real_bytes_pe(self):
        from slothctf.execution import DockerWinCompatBackend
        b = DockerWinCompatBackend()
        with TemporaryDirectory() as tmp:
            ws = Path(tmp); (ws / "src").mkdir(); (ws / "bin").mkdir()
            (ws / "src" / "c.c").write_text(_DATA_C, encoding="utf-8")
            self.assertTrue(b.compile_c(ws, "src/c.c", "bin/t.exe", get_target("pe")).ok)
            # the solver always runs objdump from the LINUX sandbox (run_solver's image),
            # not the win-compat one — confirm PLAIN objdump (no mingw prefix) still reads PE.
            linux = DockerLinuxBackend()
            h = linux._run_tool(["objdump", "-h", "/work/bin/t.exe"], ws, 30.0, ResourceLimits())
            sections = _parse_objdump_sections(h.stdout)
            raw = (ws / "bin" / "t.exe").read_bytes()
            real_off = raw.find(_NEEDLE)
            self.assertNotEqual(real_off, -1)
            hit = next((n for n, (off, size) in sections.items() if off <= real_off < off + size), None)
            self.assertIsNotNone(hit, sections)
            self.assertIn(hit, (".rdata", ".data"))


class _EofPipe:
    """A pipe that is already at EOF — read() always yields no bytes."""
    def read(self, _n):
        return b""

    def close(self):
        pass


class _HangingProc:
    """Stands in for a child that never exits on its own: wait() times out once,
    then returns after it is killed. Lets the timeout path be tested without a
    real long-running process or a compiler."""
    def __init__(self):
        self.stdout = _EofPipe()
        self.stderr = _EofPipe()
        self.stdin = None
        self.returncode = None
        self._waits = 0

    def wait(self, timeout=None):
        self._waits += 1
        if self._waits == 1:
            raise ex.subprocess.TimeoutExpired(cmd="x", timeout=timeout)
        self.returncode = -9
        return -9

    def kill(self):
        self.returncode = -9


class CappedCaptureTests(unittest.TestCase):
    """The bounded-capture runner used by both backends: a timeout is reported
    cleanly, and output is capped in memory, not buffered whole then trimmed."""

    def test_run_timeout_is_clean(self):
        b = NativeBackend()
        with mock.patch.object(ex.subprocess, "Popen", return_value=_HangingProc()):
            r = b.run_binary(Path("."), "bin/c", host_native_target())
        self.assertFalse(r.ok)
        self.assertTrue(r.timed_out)

    def test_native_output_memory_is_bounded(self):
        # The process prints ~16 MB but we keep at most output_bytes of it: the
        # retained string must stay tiny, proving output_bytes caps real memory
        # during the run rather than trimming an already-buffered flood.
        b = NativeBackend()
        with TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "spew.py").write_text(
                "import sys\n"
                "chunk=b'A'*65536\n"
                "for _ in range(256):\n"       # ~16 MB total
                "    sys.stdout.buffer.write(chunk)\n", encoding="utf-8")
            r = b.run_python(ws, "spew.py", limits=ResourceLimits(output_bytes=1000))
        self.assertTrue(r.truncated)
        self.assertFalse(r.timed_out)
        # capped bytes (1000) + a short truncation marker, and never the 16 MB.
        self.assertLessEqual(len(r.stdout.encode("utf-8")), 1000 + 128)


if __name__ == "__main__":
    unittest.main()
