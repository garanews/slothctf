"""Live isolation tests for DockerLinuxBackend.

These require Docker AND the built sandbox image, so they are skipped where either
is missing (like the gcc-gated tests). On a machine with Docker Desktop + WSL2 (or
native Docker) and `python main.py sandbox-build`, they prove the container really
is isolated: no network, bounded time, read-only root fs with a writable /tmp, and
truncated output. Run: python -m unittest tests.test_docker_backend
"""
import contextlib
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


# --- recover genre: shared pre-recorded model response (XOR 0x5a) --------------
# encoder prints flag^0x5a as bytes; challenge embeds them and VALIDATES stdin
# against them (never prints the flag); solver XORs the whole binary back and
# finds sloth{...}. Works for any flag, so the harness flag-swap round-trips.
_REC_ENCODER = ("#include <stdio.h>\nint main(int c,char**v){for(char*p=v[1];*p;++p)"
                "printf(\"0x%02x,\",(unsigned char)(*p^0x5a));return 0;}\n")
_REC_CHALLENGE = ("#include <stdio.h>\n#include <string.h>\n"
                  "static unsigned char data[]={ /*__ENCODED_ARRAY__*/ };\n"
                  "int main(void){char b[256];if(!fgets(b,sizeof b,stdin)){printf(\"no input\\n\");return 0;}"
                  "size_t n=strlen(b);if(n&&b[n-1]=='\\n')b[--n]=0;"
                  "if(n!=sizeof data){printf(\"Denied\\n\");return 0;}"
                  "for(size_t i=0;i<n;i++)if(((unsigned char)b[i]^0x5a)!=data[i]){printf(\"Denied\\n\");return 0;}"
                  "printf(\"Correct\\n\");return 0;}\n")
_REC_SOLVER = ("import sys, re\n"
               "b = open(sys.argv[1], 'rb').read()\n"
               "dec = bytes(x ^ 0x5a for x in b)\n"
               "m = re.search(rb'sloth\\{[^}]{1,64}\\}', dec)\n"
               "print(m.group(0).decode() if m else '')\n")


def _recover_response(flag, challenge=_REC_CHALLENGE, solver=_REC_SOLVER, marker="Correct"):
    return {"title": "XOR Recover", "category": "Reverse Engineering", "difficulty": "easy",
            "description": "Recover the flag from the binary.", "flag": flag,
            "challenge_kind": "recover", "success_marker": marker, "encoder_code": _REC_ENCODER,
            "challenge_code": challenge, "solver_code": solver}


# A challenge that IGNORES its input and always prints the marker — a broken
# validator that must be rejected even though encoder and solver are fine. It uses
# `data` (via a volatile sum) so the array is NOT optimised away: the embedded
# bytes are present, so verification reaches the functional check and rejects it
# there (a wrong input is accepted), not merely for missing data.
_REC_IGNORES_INPUT = ("#include <stdio.h>\n"
                      "static unsigned char data[]={ /*__ENCODED_ARRAY__*/ };\n"
                      "int main(void){ volatile unsigned s=0;"
                      "for(unsigned i=0;i<sizeof data;i++) s+=data[i];"
                      "(void)s; printf(\"Correct\\n\"); return 0; }\n")

# A solver that tries to EXECUTE the artifact instead of reading it — must fail
# (the artifact is mounted read-only with no execute bit; /tmp is noexec).
_REC_EXEC_SOLVER = ("import os, sys\n"
                    "os.execv(sys.argv[1], [sys.argv[1]])  # attempt to run the binary\n"
                    "print('unreachable')\n")

# A CORRECT validator that stores the encoded bytes in an AUTOMATIC (stack) array,
# exactly like the campaign's elf-easy#rep3. gcc materialises such an array as
# fragmented immediate stores at every -O level, so the encoder's exact sequence is
# NOT present contiguously in the binary — even though the data is there and the
# program works. The old 'data-missing' gate rejected this outright; under the fix
# the (non-fatal) contiguity signal is recorded and the SOLVER decides recoverability.
_REC_LOCAL_ARRAY = ("#include <stdio.h>\n#include <string.h>\n"
                    "int main(void){unsigned char data[]={ /*__ENCODED_ARRAY__*/ };"
                    "char b[256];if(!fgets(b,sizeof b,stdin)){printf(\"Denied\\n\");return 0;}"
                    "size_t n=strlen(b);if(n&&b[n-1]=='\\n')b[--n]=0;"
                    "if(n!=sizeof data){printf(\"Denied\\n\");return 0;}"
                    "for(size_t i=0;i<n;i++)if(((unsigned char)b[i]^0x5a)!=data[i]){printf(\"Denied\\n\");return 0;}"
                    "printf(\"Correct\\n\");return 0;}\n")

# A validator that embeds and USES the array (data present, static, contiguous) but
# decodes with the WRONG key (0x5b vs the encoder's 0x5a), so it rejects even the
# correct flag — the functional contract must catch this (recover-rejects-correct).
_REC_REJECTS_CORRECT = ("#include <stdio.h>\n#include <string.h>\n"
                        "static unsigned char data[]={ /*__ENCODED_ARRAY__*/ };"
                        "int main(void){char b[256];if(!fgets(b,sizeof b,stdin)){printf(\"Denied\\n\");return 0;}"
                        "size_t n=strlen(b);if(n&&b[n-1]=='\\n')b[--n]=0;"
                        "if(n!=sizeof data){printf(\"Denied\\n\");return 0;}"
                        "for(size_t i=0;i<n;i++)if(((unsigned char)b[i]^0x5b)!=data[i]){printf(\"Denied\\n\");return 0;}"
                        "printf(\"Correct\\n\");return 0;}\n")


@contextlib.contextmanager
def _generate_recover(name, target, backend, response):
    """Generate a recover challenge in a throwaway cwd; yields (report, dir) while
    the workspace still exists so the caller can inspect files, then cleans up."""
    from slothctf.generator import generate_challenge
    prev = os.getcwd()
    with TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            report = generate_challenge(name, prompt="xor recover", genre="recover",
                                        caller=lambda p, m: dict(response), backend=backend,
                                        target=target, max_attempts=1, return_report=True)
            yield report, Path(tmp) / "generated" / name
        finally:
            os.chdir(prev)


@unittest.skipUnless(_READY, _SKIP)
class RecoverGenreTests(unittest.TestCase):
    """Live recover-genre verification on real binaries (ELF via DockerLinux)."""

    def test_recover_elf_end_to_end(self):
        with _generate_recover("rec_elf", "elf", DockerLinuxBackend(),
                               _recover_response("sloth{rec_elf}")) as (report, cdir):
            self.assertTrue(report.success, report.as_text())
            self.assertTrue((cdir / "bin" / "challenge").exists())
            self.assertTrue((cdir / "organizer" / "solver.py").exists())

    def test_hardcoded_solver_is_rejected_by_flag_swap(self):
        # Solver prints the ORIGINAL flag literally: the first recovery looks OK,
        # but the harness flag-swap (a different binary) exposes it as hardcoded.
        flag = "sloth{rec_hc}"
        with _generate_recover("rec_hc", "elf", DockerLinuxBackend(),
                               _recover_response(flag, solver=f"print({flag!r})\n")) as (report, _):
            self.assertFalse(report.success)
            self.assertEqual(report.attempts[-1].stage, "solver-hardcoded")

    def test_solver_reading_challenge_json_fails(self):
        # The isolated solver workspace has no challenge.json, so a solver that
        # tries to read it fails (file absent) — it never sees organizer data.
        peek = "import json,sys\nprint(json.load(open('challenge.json'))['flag'])\n"
        with _generate_recover("rec_peek", "elf", DockerLinuxBackend(),
                               _recover_response("sloth{rec_peek}", solver=peek)) as (report, _):
            self.assertFalse(report.success)
            self.assertEqual(report.attempts[-1].stage, "solver-error")

    def test_recover_revalidates_on_disk(self):
        from slothctf.validator import validate_challenge
        with _generate_recover("rec_val", "elf", DockerLinuxBackend(),
                               _recover_response("sloth{rec_val}")) as (report, cdir):
            self.assertTrue(report.success, report.as_text())
            result = validate_challenge(cdir, backend=DockerLinuxBackend())
            self.assertTrue(result.success, str(result))

    def test_program_ignoring_input_is_rejected(self):
        # encoder + solver are fine, but the binary ignores stdin and always prints
        # the marker -> the functional check rejects it (a wrong input is accepted).
        with _generate_recover("rec_ignore", "elf", DockerLinuxBackend(),
                               _recover_response("sloth{rec_ignore}", challenge=_REC_IGNORES_INPUT)
                               ) as (report, _):
            self.assertFalse(report.success)
            self.assertEqual(report.attempts[-1].stage, "recover-accepts-wrong")

    def test_solver_cannot_execute_the_binary(self):
        # A solver that tries os.execv on the ELF is stopped by the stripped exec bit
        # on the read-only mount, so it fails to recover the flag -> rejected. (This
        # blocks the naive exec path; the ld-loader bypass is not claimed to be
        # prevented — the container sandbox is the real boundary. See run_solver.)
        with _generate_recover("rec_exec", "elf", DockerLinuxBackend(),
                               _recover_response("sloth{rec_exec}", solver=_REC_EXEC_SOLVER)
                               ) as (report, _):
            self.assertFalse(report.success)
            self.assertIn(report.attempts[-1].stage, {"solver-error", "solver-mismatch"})

    # --- P1: contiguity is diagnostic-only; the solver+swap decide recoverability ---

    def test_contiguous_recoverable_passes_and_reports_contiguity_diagnostic(self):
        # A static-array challenge: the encoded bytes ARE contiguous and a simple
        # solver recovers them. It passes, and the (now non-fatal) contiguity check
        # is recorded as a diagnostic signal on the passing attempt.
        with _generate_recover("rec_diag", "elf", DockerLinuxBackend(),
                               _recover_response("sloth{rec_diag}")) as (report, _):
            self.assertTrue(report.success, report.as_text())
            self.assertIs(report.attempts[-1].diagnostics.get("encoded_bytes_contiguous"), True)

    def test_noncontiguous_local_array_not_rejected_for_contiguity(self):
        # Reproduces elf-easy#rep3: an automatic-array validator whose encoded bytes
        # are NOT contiguous in the binary. The fix removes the 'data-missing' reject,
        # so verification proceeds to the solver. A naive XOR-sweep solver cannot
        # recover fragmented data, so the challenge fails at the SOLVER stage — it is
        # NOT blessed merely for being non-contiguous, and NOT rejected for contiguity.
        flag = "sloth{noncontiguous_local_demo}"
        with _generate_recover("rec_local", "elf", DockerLinuxBackend(),
                               _recover_response(flag, challenge=_REC_LOCAL_ARRAY)) as (report, cdir):
            want = bytes(c ^ 0x5A for c in flag.encode())      # the encoder's exact output
            blob = (cdir / "bin" / "challenge").read_bytes()
            self.assertNotIn(want, blob, "premise: the encoded sequence must be non-contiguous here")
            self.assertFalse(report.success)
            self.assertNotEqual(report.attempts[-1].stage, "data-missing")
            self.assertIn(report.attempts[-1].stage, {"solver-mismatch", "solver-error"})

    def test_program_rejecting_correct_flag_is_rejected(self):
        # Data is present and contiguous, but the validator decodes with the wrong key
        # and rejects the correct flag: the functional contract must catch it.
        with _generate_recover("rec_rej", "elf", DockerLinuxBackend(),
                               _recover_response("sloth{rec_rej}", challenge=_REC_REJECTS_CORRECT)
                               ) as (report, _):
            self.assertFalse(report.success)
            self.assertEqual(report.attempts[-1].stage, "recover-rejects-correct")


@unittest.skipUnless(_WIN_READY and _READY, _WIN_SKIP)
class RecoverPeTests(unittest.TestCase):
    """Recover for a Windows PE: challenge built/run under Wine, solver (Python)
    analyses the .exe bytes on the Linux backend."""

    def test_recover_pe_end_to_end(self):
        with _generate_recover("rec_pe", "pe", DockerWinCompatBackend(),
                               _recover_response("sloth{rec_pe}")) as (report, cdir):
            self.assertTrue(report.success, report.as_text())
            self.assertTrue((cdir / "bin" / "challenge.exe").exists())


if __name__ == "__main__":
    unittest.main()
