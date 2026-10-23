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
from slothctf.execution import ExecResult, NativeBackend


def _er(ok=True, stdout="", stderr="", exit_code=0, timed_out=False, truncated=False):
    return ExecResult(ok, stdout, stderr, exit_code=exit_code, timed_out=timed_out, truncated=truncated)


class RecoverExecutionContractTests(unittest.TestCase):
    """The shared functional contract (_recover_execution_problem), used at both
    generation and validation — tested deterministically with crafted ExecResults."""

    FLAG = "sloth{x}"

    def problem(self, good, bad, marker="Unlocked"):
        return g._recover_execution_problem(good, bad, marker, self.FLAG)

    def test_well_behaved_validator_passes(self):
        self.assertIsNone(self.problem(_er(stdout="Unlocked\n"), _er(stdout="Nope\n")))

    def test_marker_substring_is_not_a_false_accept(self):
        # marker 'Correct' must NOT be seen as present in 'Incorrect' (line-exact).
        self.assertIsNone(
            g._recover_execution_problem(_er(stdout="Correct\n"), _er(stdout="Incorrect\n"),
                                         "Correct", self.FLAG))

    def test_always_accept_rejected(self):
        self.assertEqual(self.problem(_er(stdout="Unlocked\n"), _er(stdout="Unlocked\n"))[0],
                         "recover-accepts-wrong")

    def test_wrong_input_nonzero_exit_rejected(self):
        # both runs must exit 0; a non-zero exit on wrong input is a failure
        self.assertEqual(self.problem(_er(stdout="Unlocked\n"),
                                      _er(ok=False, exit_code=1, stdout="Nope\n"))[0],
                         "challenge-bad-exit")

    def test_crash_on_correct_rejected(self):
        self.assertEqual(self.problem(_er(ok=False, exit_code=139), _er(stdout="Nope\n"))[0],
                         "challenge-crash")

    def test_timeout_rejected(self):
        self.assertEqual(self.problem(_er(timed_out=True), _er(stdout="Nope\n"))[0],
                         "challenge-timeout")

    def test_truncated_output_rejected(self):
        # incomplete output cannot prove the flag is absent
        self.assertEqual(self.problem(_er(stdout="Unlocked\n", truncated=True), _er(stdout="Nope\n"))[0],
                         "output-truncated")

    def test_flag_revealed_is_trivial(self):
        self.assertEqual(self.problem(_er(stdout="Unlocked\n" + self.FLAG), _er(stdout="Nope\n"))[0],
                         "too-trivial")

    def test_marker_matches_line_exact(self):
        self.assertTrue(g._marker_matches("foo\nUnlocked\nbar", "Unlocked"))
        self.assertTrue(g._marker_matches("Unlocked", "Unlocked"))
        self.assertFalse(g._marker_matches("Incorrect", "Correct"))
        self.assertFalse(g._marker_matches("", "Unlocked"))


class SchemaAndNormalizeTests(unittest.TestCase):
    def test_schema_describes_recover_solver_and_marker(self):
        s = g._SCHEMA_RULES_RECOVER
        self.assertIn('"challenge_kind" : "recover"', s)
        self.assertIn('"solver_code"', s)
        self.assertIn('"success_marker"', s)
        self.assertIn("MUST NOT print the flag", s)
        self.assertIn(g.ENCODED_PLACEHOLDER, s)

    def test_distinct_random_flag_never_equals_exclude(self):
        # Regression for the flag-swap bug: the second flag must differ from the
        # original (it was wrongly compared to the challenge template before).
        import unittest.mock as mock
        seq = iter(["sloth{same}", "sloth{same}", "sloth{different}"])
        with mock.patch.object(g, "_random_flag", side_effect=lambda: next(seq)):
            self.assertEqual(g._distinct_random_flag("sloth{same}"), "sloth{different}")

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
