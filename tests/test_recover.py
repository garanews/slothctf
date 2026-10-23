"""Unit tests for the `recover` genre (no Docker daemon needed).

The pieces that don't need a C compiler: schema/normalization, the byte/leak
helpers, and the ISOLATED solver runner (the solver sees only its script and the
artifact — never challenge.json, sources, encoder or the expected flag). Live
compile/run end-to-end for both ELF and PE, plus the hardcoded-solver and the
reads-challenge.json negatives, are gated Docker tests in test_docker_backend.py.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from slothctf import generator as g
from slothctf.execution import NativeBackend


class SchemaAndNormalizeTests(unittest.TestCase):
    def test_schema_describes_recover_and_solver(self):
        s = g._SCHEMA_RULES_RECOVER
        self.assertIn('"challenge_kind" : "recover"', s)
        self.assertIn('"solver_code"', s)
        self.assertIn("MUST NOT print the flag", s)
        self.assertIn(g.ENCODED_PLACEHOLDER, s)

    def test_normalize_recover_sets_kind_and_keeps_solver(self):
        data = g._normalize_challenge(
            {"flag": "bare", "encoder_code": "```c\nint main(){}\n```",
             "challenge_code": "x", "solver_code": "```python\nprint(1)\n```"},
            "easy", None, genre="recover")
        self.assertEqual(data["challenge_kind"], "recover")
        self.assertEqual(data["flag"], "sloth{bare}")          # normalized
        self.assertEqual(data["encoder_code"], "int main(){}")  # fences stripped
        self.assertEqual(data["solver_code"], "print(1)")

    def test_required_fields_include_solver(self):
        # a recover response missing solver_code is fed back as a schema failure
        report = _run_loop(g, {"encoder_code": "e", "challenge_code": "c"})  # no solver_code
        self.assertFalse(report.success)
        self.assertEqual(report.attempts[0].stage, "bad-schema")


class ByteAndLeakHelperTests(unittest.TestCase):
    def test_tokens_to_bytes(self):
        self.assertEqual(g._tokens_to_bytes("0x31, 0x2e,0x00, 0xff"), bytes([0x31, 0x2e, 0x00, 0xff]))
        self.assertEqual(g._tokens_to_bytes("no tokens"), b"")

    def test_leak_detects_utf8_and_utf16le(self):
        with TemporaryDirectory() as tmp:
            p8 = Path(tmp) / "a"; p8.write_bytes(b"\x00sloth{visible}\x00")
            self.assertTrue(g._flag_leaks_recover(p8, "sloth{visible}"))
            # UTF-16LE (wide) string — matters for PE; plain utf-8 check would miss it
            p16 = Path(tmp) / "b"; p16.write_bytes(b"\x00" + "sloth{wide}".encode("utf-16-le"))
            self.assertTrue(g._flag_leaks_recover(p16, "sloth{wide}"))
            self.assertTrue(g._flag_leaks_recover(p16, "wide"))      # inner secret too
            clean = Path(tmp) / "c"; clean.write_bytes(b"\xde\xad\xbe\xef obfuscated only")
            self.assertFalse(g._flag_leaks_recover(clean, "sloth{hidden}"))


@unittest.skipUnless(__import__("shutil").which("python") or __import__("shutil").which("python3"),
                     "needs a host python for the native solver runner")
class SolverIsolationTests(unittest.TestCase):
    """The solver runs with ONLY its script + the artifact in a throwaway workspace."""

    def test_solver_recovers_from_artifact_bytes(self):
        solver = ("import sys, re\n"
                  "b = open(sys.argv[1], 'rb').read()\n"
                  "m = re.search(rb'sloth\\{[^}]{1,64}\\}', b)\n"
                  "print(m.group(0).decode() if m else '')\n")
        with TemporaryDirectory() as tmp:
            art = Path(tmp) / "bin.dat"
            art.write_bytes(b"\x01\x02sloth{found_me}\x03\x04")
            ok, out, detail = g._run_solver_isolated(solver, art, NativeBackend())
        self.assertTrue(ok, detail)
        self.assertEqual(out.strip(), "sloth{found_me}")

    def test_solver_cannot_see_challenge_json(self):
        # A solver that peeks at challenge.json must fail: the file is not in the
        # isolated workspace (FileNotFoundError -> non-zero exit).
        solver = ("import json\n"
                  "d = json.load(open('challenge.json'))\n"
                  "print(d['flag'])\n")
        with TemporaryDirectory() as tmp:
            art = Path(tmp) / "bin.dat"
            art.write_bytes(b"sloth{secret}")
            ok, out, detail = g._run_solver_isolated(solver, art, NativeBackend())
        self.assertFalse(ok)                 # it could not read the file it should not have
        self.assertIn("non-zero", detail)


def _run_loop(gmod, response: dict):
    """Drive generate_challenge's loop for recover with a fixed model response and
    verification stubbed, to exercise schema handling without a compiler."""
    from unittest import mock
    with mock.patch.object(gmod, "_write_challenge_files"), \
         mock.patch.object(gmod, "_verify_recover_challenge",
                           return_value=gmod.AttemptResult(0, True, "passed")), \
         mock.patch.object(gmod, "resolve_backend", return_value=NativeBackend()):
        return gmod.generate_challenge(
            "unit_recover", prompt="x", genre="recover",
            caller=lambda p, m: dict(response), backend=NativeBackend(),
            max_attempts=2, return_report=True)


if __name__ == "__main__":
    unittest.main()
