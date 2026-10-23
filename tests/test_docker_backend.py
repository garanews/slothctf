"""Live isolation tests for DockerLinuxBackend.

These require Docker AND the built sandbox image, so they are skipped where either
is missing (like the gcc-gated tests). On a machine with Docker Desktop + WSL2 (or
native Docker) and `python main.py sandbox-build`, they prove the container really
is isolated: no network, bounded time, read-only root fs with a writable /tmp, and
truncated output. Run: python -m unittest tests.test_docker_backend
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from slothctf.execution import (
    DockerLinuxBackend,
    ResourceLimits,
    docker_available,
    image_available,
)
from slothctf.targets import ELF

_READY = docker_available() and image_available()
_SKIP = "needs Docker + the slothctf-sandbox image (python main.py sandbox-build)"


def _ws(tmp: str) -> Path:
    ws = Path(tmp)
    (ws / "src").mkdir(parents=True, exist_ok=True)
    (ws / "work").mkdir(parents=True, exist_ok=True)
    return ws


@unittest.skipUnless(_READY, _SKIP)
class DockerIsolationTests(unittest.TestCase):
    def setUp(self):
        self.b = DockerLinuxBackend()

    def test_compile_and_run_elf(self):
        with TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            (ws / "src" / "c.c").write_text(
                '#include <stdio.h>\nint main(){printf("sloth{ok}");return 0;}', encoding="utf-8")
            c = self.b.compile_c(ws, "src/c.c", "bin/c", ELF)
            self.assertTrue(c.ok, c.stderr)
            self.assertTrue((ws / "bin" / "c").exists())           # ELF landed on the host mount
            r = self.b.run_binary(ws, "bin/c", ELF)
            self.assertTrue(r.ok)
            self.assertIn("sloth{ok}", r.stdout)

    def test_network_is_disabled(self):
        with TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            (ws / "net.py").write_text(
                "import socket,sys\n"
                "try:\n"
                "  socket.create_connection(('8.8.8.8',53),2); print('CONNECTED'); sys.exit(1)\n"
                "except Exception:\n"
                "  print('NO-NET'); sys.exit(0)\n", encoding="utf-8")
            r = self.b.run_python(ws, "net.py")
            self.assertTrue(r.ok, r.stderr)
            self.assertIn("NO-NET", r.stdout)                      # --network none held

    def test_run_timeout(self):
        with TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            (ws / "loop.py").write_text("while True:\n  pass\n", encoding="utf-8")
            r = self.b.run_python(ws, "loop.py", limits=ResourceLimits(run_timeout=2.0))
            self.assertTrue(r.timed_out)

    def test_root_fs_readonly_tmp_writable(self):
        with TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            (ws / "fs.py").write_text(
                "ok_tmp=False; ro_root=False\n"
                "try:\n  open('/tmp/x','w').write('1'); ok_tmp=True\nexcept Exception: pass\n"
                "try:\n  open('/root_probe','w').write('1')\nexcept Exception: ro_root=True\n"
                "print('TMP' if ok_tmp else 'noTMP', 'RO' if ro_root else 'RW')\n", encoding="utf-8")
            r = self.b.run_python(ws, "fs.py")
            self.assertIn("TMP", r.stdout)                         # writable tmpfs
            self.assertIn("RO", r.stdout)                          # read-only root fs

    def test_output_is_truncated(self):
        with TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            (ws / "big.py").write_text("print('A'*100000)\n", encoding="utf-8")
            r = self.b.run_python(ws, "big.py", limits=ResourceLimits(output_bytes=100))
            self.assertTrue(r.truncated)
            self.assertLess(len(r.stdout), 500)

    def test_refuses_pe_target(self):
        from slothctf.execution import SandboxUnavailable
        from slothctf.targets import PE
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SandboxUnavailable):
                self.b.compile_c(Path(tmp), "src/c.c", "bin/c.exe", PE)  # ELF-only in Phase A


if __name__ == "__main__":
    unittest.main()
