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
        self.assertLessEqual(len(ev), 6000)                    # bounded to the context budget


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
        self.assertLessEqual(len(ev), 6000)


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
