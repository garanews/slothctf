"""Live isolation tests for DockerLinuxBackend.

These require Docker AND the built sandbox image, so they are skipped where either
is missing (like the gcc-gated tests). On a machine with Docker Desktop + WSL2 (or
native Docker) and `python main.py sandbox-build`, they prove the container really
is isolated: no network, bounded time, read-only root fs with a writable /tmp, and
truncated output. Run: python -m unittest tests.test_docker_backend
"""
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from slothctf.execution import (
    DockerLinuxBackend,
    DockerWinCompatBackend,
    ResourceLimits,
    SandboxUnavailable,
    docker_available,
    image_available,
    win_image_available,
)
from slothctf.targets import ELF, PE, detect_target_from_binary

_READY = docker_available() and image_available()
_SKIP = "needs Docker + the slothctf-sandbox image (python main.py sandbox-build)"
_WIN_READY = docker_available() and win_image_available()
_WIN_SKIP = "needs Docker + the slothctf-sandbox-win image (python main.py sandbox-build --win)"


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


@unittest.skipUnless(_READY, _SKIP)
class EndToEndElfTests(unittest.TestCase):
    """Full generate -> Docker compile/run -> validate path for an ELF challenge,
    driven by a PRE-RECORDED model response (no Ollama). This is the integration
    seam the unit tests stub out: here the model's C is really compiled and run in
    the sandbox container, both while generating and again while validating. It is
    NOT a substitute for a real end-to-end run with Qwen — it fixes the model's
    output so the test is deterministic; a live-model run stays a separate check.
    """

    # A minimal, self-consistent decode challenge: the encoder XORs the flag with
    # 0x5a and prints the bytes; the challenge embeds those bytes and XORs them back.
    _FLAG = "sloth{elf_e2e}"
    _RESPONSE = {
        "title": "XOR E2E",
        "category": "Reverse Engineering",
        "difficulty": "easy",
        "description": "Recover the flag printed by the binary.",
        "flag": _FLAG,
        "challenge_kind": "decode",
        "encoder_code": (
            "#include <stdio.h>\n"
            "int main(int c,char**v){for(char*p=v[1];*p;++p)"
            "printf(\"0x%02x,\",(unsigned char)(*p^0x5a));return 0;}\n"
        ),
        "challenge_code": (
            "#include <stdio.h>\n"
            "static unsigned char t[]={ /*__ENCODED_ARRAY__*/ };\n"
            "int main(void){for(unsigned long i=0;i<sizeof t;i++)"
            "putchar(t[i]^0x5a);return 0;}\n"
        ),
    }

    def test_generate_then_validate_in_sandbox(self):
        from slothctf.generator import generate_challenge
        from slothctf.validator import validate_challenge

        def fake_caller(prompt, model):
            return dict(self._RESPONSE)

        prev = os.getcwd()
        with TemporaryDirectory() as tmp:
            os.chdir(tmp)  # generate_challenge writes under ./generated/<name>
            try:
                report = generate_challenge(
                    "elf_e2e",
                    prompt="easy XOR reverse",
                    caller=fake_caller,
                    backend=DockerLinuxBackend(),   # real sandbox, no silent fallback
                    target="elf",
                    max_attempts=1,
                    return_report=True,
                )
                self.assertTrue(report.success, report.as_text())
                challenge_dir = Path("generated") / "elf_e2e"
                # The ELF the container compiled is readable on the host mount.
                self.assertTrue((challenge_dir / "bin" / "challenge").exists())
                for name in ("challenge.json", "README.md", "solution.md"):
                    self.assertTrue((challenge_dir / name).exists(), name)

                # Re-validate from scratch through the same sandbox.
                result = validate_challenge(challenge_dir, backend=DockerLinuxBackend())
                self.assertTrue(result.success, str(result))
            finally:
                os.chdir(prev)


@unittest.skipUnless(_WIN_READY, _WIN_SKIP)
class WinCompatIsolationTests(unittest.TestCase):
    """Live tests for the MinGW-w64 + Wine backend: a real PE is cross-compiled and
    run under Wine in the isolated container. Gated on the built win image."""

    def setUp(self):
        self.b = DockerWinCompatBackend()

    def test_compile_and_run_pe_under_wine(self):
        with TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            (ws / "src" / "c.c").write_text(
                '#include <stdio.h>\nint main(){printf("sloth{pe_ok}");return 0;}', encoding="utf-8")
            c = self.b.compile_c(ws, "src/c.c", "bin/challenge.exe", PE)
            self.assertTrue(c.ok, c.stderr)
            exe = ws / "bin" / "challenge.exe"
            self.assertTrue(exe.exists())                              # PE landed on the host mount
            self.assertEqual(detect_target_from_binary(exe).name, "pe")  # real x86-64 PE
            r = self.b.run_binary(ws, "bin/challenge.exe", PE)
            self.assertTrue(r.ok, r.stderr)
            self.assertIn("sloth{pe_ok}", r.stdout)

    def test_run_exposes_only_the_exe(self):
        # The run mounts ONLY the PE, so the rest of the workspace is not in the
        # container: a program reaching for a sibling file cannot find it.
        with TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            (ws / "secret.txt").write_text("TOPSECRET", encoding="utf-8")
            (ws / "src" / "peek.c").write_text(
                '#include <stdio.h>\n'
                'int main(){FILE*f=fopen("Z:\\\\work\\\\secret.txt","r");'
                'if(!f){printf("NO-ACCESS");return 0;}printf("LEAKED");return 0;}', encoding="utf-8")
            c = self.b.compile_c(ws, "src/peek.c", "bin/challenge.exe", PE)
            self.assertTrue(c.ok, c.stderr)
            r = self.b.run_binary(ws, "bin/challenge.exe", PE)
            self.assertIn("NO-ACCESS", r.stdout)
            self.assertNotIn("LEAKED", r.stdout)

    def test_refuses_elf_target(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SandboxUnavailable):
                self.b.compile_c(Path(tmp), "src/c.c", "bin/c", ELF)  # PE-only backend

    def test_threaded_pe_runs_under_wine(self):
        # Regression for the compatibility study's one failure: a pthread-using PE
        # must load and run under Wine. The backend links -static -pthread, so
        # winpthread is embedded instead of imported as libwinpthread-1.dll (which
        # the Wine prefix does not have). Without the fix this fails at load.
        with TemporaryDirectory() as tmp:
            ws = _ws(tmp)
            (ws / "src" / "c.c").write_text(
                '#include <stdio.h>\n#include <pthread.h>\n'
                'static void* w(void* a){ return a; }\n'
                'int main(void){ pthread_t t; pthread_create(&t,0,w,0); pthread_join(t,0);'
                ' printf("sloth{threads_ok}"); return 0; }\n', encoding="utf-8")
            c = self.b.compile_c(ws, "src/c.c", "bin/challenge.exe", PE)
            self.assertTrue(c.ok, c.stderr)
            r = self.b.run_binary(ws, "bin/challenge.exe", PE)
            self.assertTrue(r.ok, r.stderr)
            self.assertIn("sloth{threads_ok}", r.stdout)

    def test_generate_then_validate_pe(self):
        from slothctf.generator import generate_challenge
        from slothctf.validator import validate_challenge
        resp = {
            "title": "XOR PE", "category": "Reverse Engineering", "difficulty": "easy",
            "description": "Recover the flag.", "flag": "sloth{pe_e2e}",
            "challenge_kind": "decode",
            "encoder_code": ("#include <stdio.h>\nint main(int c,char**v){for(char*p=v[1];*p;++p)"
                             "printf(\"0x%02x,\",(unsigned char)(*p^0x5a));return 0;}\n"),
            "challenge_code": ("#include <stdio.h>\nstatic unsigned char t[]={ /*__ENCODED_ARRAY__*/ };\n"
                               "int main(void){for(unsigned long i=0;i<sizeof t;i++)"
                               "putchar(t[i]^0x5a);return 0;}\n"),
        }
        prev = os.getcwd()
        with TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                report = generate_challenge(
                    "pe_e2e", prompt="easy XOR reverse",
                    caller=lambda p, m: dict(resp),
                    backend=DockerWinCompatBackend(), target="pe",
                    max_attempts=1, return_report=True)
                self.assertTrue(report.success, report.as_text())
                cd = Path("generated") / "pe_e2e"
                self.assertTrue((cd / "bin" / "challenge.exe").exists())
                result = validate_challenge(cd, backend=DockerWinCompatBackend())
                self.assertTrue(result.success, str(result))
            finally:
                os.chdir(prev)


if __name__ == "__main__":
    unittest.main()
