"""Phase B — Windows-compat backend (MinGW-w64 + Wine).

Unit tests here need no Docker daemon: they assert the argv/isolation/mount shape
the backend builds. Live container tests (compile a real PE, run it under Wine,
isolation holds) live in test_docker_backend.py, gated on the built win image, and
the compatibility study over the existing 32 PE challenges is its own tool/report.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from slothctf.execution import (
    DockerWinCompatBackend,
    ExecResult,
    ResourceLimits,
    SandboxUnavailable,
    build_docker_run_argv,
)
from slothctf.targets import ELF, PE


def _mounts_of(argv):
    return [argv[i + 1] for i, t in enumerate(argv) if t == "-v"]


class WinArgvTests(unittest.TestCase):
    """build_docker_run_argv keeps the Phase A security flags while allowing the
    tighter mount / tmpfs / env a Wine run needs."""

    def test_default_shape_unchanged(self):
        argv = build_docker_run_argv("img", ["gcc"], Path("."), ResourceLimits(), "cn")
        s = " ".join(argv)
        self.assertIn(":/work:rw", s)                       # whole workspace, as before
        self.assertIn("--tmpfs /tmp:rw,size=64m,noexec", s)  # default tmpfs
        self.assertNotIn("-e ", s)                           # no env injected by default

    def test_custom_mounts_env_tmpfs(self):
        argv = build_docker_run_argv(
            "img", ["winrun", "/run/challenge.exe"], Path("."),
            ResourceLimits(memory="1g"), "cn",
            mounts=[("./a/b.exe", "/run/challenge.exe", "ro")],
            tmpfs="/tmp:rw,size=384m",
            env=[("HOME", "/tmp"), ("WINEPREFIX", "/tmp/wineprefix")])
        s = " ".join(argv)
        # fixed isolation flags survive
        for flag in ("--network none", "--read-only", "--cap-drop ALL",
                     "no-new-privileges", "--user 65534:65534"):
            self.assertIn(flag, s)
        # only the single read-only mount, and NOT the rw workspace mount
        self.assertNotIn(":/work:rw", s)
        mounts = _mounts_of(argv)
        self.assertEqual(len(mounts), 1)
        self.assertTrue(mounts[0].endswith(":/run/challenge.exe:ro"))
        self.assertIn("--tmpfs /tmp:rw,size=384m", s)
        self.assertIn("-e HOME=/tmp", s)
        self.assertIn("-e WINEPREFIX=/tmp/wineprefix", s)


class WinBackendUnitTests(unittest.TestCase):
    def test_supports_only_pe(self):
        b = DockerWinCompatBackend()
        self.assertTrue(b.can_build(PE))
        self.assertFalse(b.can_build(ELF))
        self.assertEqual(b.fidelity, "wine")
        self.assertTrue(b.safe)

    def test_compile_uses_mingw_cross_and_writable_workspace(self):
        b = DockerWinCompatBackend()
        with TemporaryDirectory() as tmp, \
             mock.patch.object(b, "_capped", return_value=ExecResult(True, "", "")) as m:
            b.compile_c(Path(tmp), "src/challenge.c", "bin/challenge.exe", PE)
            argv = m.call_args.args[0]
        self.assertIn("x86_64-w64-mingw32-gcc", argv)
        self.assertIn("/work/bin/challenge.exe", argv)
        self.assertIn("/work/src/challenge.c", argv)
        self.assertIn(":/work:rw", " ".join(argv))  # compile must write the .exe back
        # self-contained link: no external libwinpthread-1.dll / libgcc DLL imports
        self.assertIn("-static", argv)
        self.assertIn("-pthread", argv)

    def test_run_mounts_only_the_exe_and_uses_winrun(self):
        b = DockerWinCompatBackend()
        with TemporaryDirectory() as tmp, \
             mock.patch.object(b, "_capped", return_value=ExecResult(True, "", "")) as m:
            (Path(tmp) / "bin").mkdir()
            (Path(tmp) / "bin" / "challenge.exe").write_bytes(b"MZ\x90\x00")
            b.run_binary(Path(tmp), "bin/challenge.exe", PE)
            argv = m.call_args.args[0]
        s = " ".join(argv)
        self.assertIn("winrun", argv)
        self.assertIn("/run/challenge.exe", argv)
        self.assertNotIn(":/work:rw", s)  # the rest of the workspace stays invisible
        mounts = _mounts_of(argv)
        self.assertEqual(len(mounts), 1)
        self.assertTrue(mounts[0].endswith(":/run/challenge.exe:ro"))
        self.assertIn("--tmpfs /tmp:rw,size=384m", s)  # writable, exec-capable Wine tmpfs
        self.assertIn("-e WINEPREFIX=/tmp/wineprefix", s)

    def test_default_memory_floor_raised_for_wine(self):
        # A default (limits=None) run gets the larger Wine/tmpfs memory floor.
        b = DockerWinCompatBackend()
        with TemporaryDirectory() as tmp, \
             mock.patch.object(b, "_capped", return_value=ExecResult(True, "", "")) as m:
            (Path(tmp) / "bin").mkdir()
            (Path(tmp) / "bin" / "challenge.exe").write_bytes(b"MZ")
            b.run_binary(Path(tmp), "bin/challenge.exe", PE)
            argv = m.call_args.args[0]
        self.assertIn("--memory 1g", " ".join(argv))

    def test_refuses_elf_target(self):
        b = DockerWinCompatBackend()
        with self.assertRaises(SandboxUnavailable):
            b.compile_c(Path("."), "src/c.c", "bin/c", ELF)

    def test_run_python_is_refused(self):
        b = DockerWinCompatBackend()
        with self.assertRaises(SandboxUnavailable):
            b.run_python(Path("."), "x.py")


if __name__ == "__main__":
    unittest.main()
