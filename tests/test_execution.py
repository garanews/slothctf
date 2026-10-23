import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from slothctf import execution as ex
from slothctf.execution import (
    DockerBackend,
    ExecResult,
    NativeBackend,
    ResourceLimits,
    SandboxUnavailable,
    build_docker_run_argv,
    get_backend,
)

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
            self.assertIsInstance(get_backend(), DockerBackend)

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
        self.assertTrue(DockerBackend().safe)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            get_backend("vm")


class DockerInnerCommandTests(unittest.TestCase):
    """Container paths are /work-relative; verify the inner argv the backend runs."""

    def test_compile_maps_paths_into_work(self):
        b = DockerBackend()
        with TemporaryDirectory() as tmp, \
             mock.patch.object(b, "_exec", return_value=ExecResult(True, "", "")) as m:
            b.compile_c(Path(tmp), "src/challenge.c", "bin/challenge")
            inner = m.call_args.args[0]
            self.assertEqual(inner, ["gcc", "-O2", "-s", "-o", "/work/bin/challenge",
                                     "/work/src/challenge.c"])

    def test_run_python_uses_python3_isolated(self):
        b = DockerBackend()
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
            c = b.compile_c(ws, "src/c.c", "bin/c")
            self.assertTrue(c.ok, c.stderr)
            # MinGW may append .exe
            rel = "bin/c" if (ws / "bin" / "c").exists() else "bin/c.exe"
            r = b.run_binary(ws, rel)
            self.assertTrue(r.ok)
            self.assertIn("sloth{ok}", r.stdout)

    def test_run_timeout_is_clean(self):
        b = NativeBackend()
        with TemporaryDirectory() as tmp, \
             mock.patch.object(ex.subprocess, "run",
                               side_effect=ex.subprocess.TimeoutExpired("x", 1)):
            r = b.run_binary(Path(tmp), "bin/c")
            self.assertFalse(r.ok)
            self.assertTrue(r.timed_out)


if __name__ == "__main__":
    unittest.main()
