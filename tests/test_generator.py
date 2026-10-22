import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from slothctf import generator
from slothctf.generator import (
    AttemptResult,
    ENCODED_PLACEHOLDER,
    GenerationError,
    _build_generation_prompt,
    _coerce_json,
    _flag_leaks_in_binary,
    _generate_endpoint,
    _normalize_flag,
    _parse_encoder_output,
    _render_readme,
    _render_solution,
    _SCHEMA_RULES,
    _strip_code_fences,
    generate_challenge,
)


class FlagHelperTests(unittest.TestCase):
    def test_normalize_wraps_bare_secret(self):
        self.assertEqual(_normalize_flag("devourer of destiny"), "sloth{devourer_of_destiny}")

    def test_normalize_keeps_wellformed_flag(self):
        self.assertEqual(_normalize_flag("sloth{already_good}"), "sloth{already_good}")

    def test_normalize_empty_falls_back(self):
        self.assertEqual(_normalize_flag("   "), "sloth{slow_and_steady}")

    def test_flag_leak_detection(self):
        with TemporaryDirectory() as tmp:
            binary = Path(tmp) / "challenge"
            binary.write_bytes(b"\x00\x01sloth{visible_secret}\x02padding")
            self.assertTrue(_flag_leaks_in_binary(binary, "sloth{visible_secret}"))
            self.assertTrue(_flag_leaks_in_binary(binary, "visible_secret"))

    def test_flag_leak_detection_negative(self):
        with TemporaryDirectory() as tmp:
            binary = Path(tmp) / "challenge"
            binary.write_bytes(b"\xde\xad\xbe\xef random compiled bytes")
            self.assertFalse(_flag_leaks_in_binary(binary, "sloth{hidden}"))


class PromptBuildTests(unittest.TestCase):
    def test_prompt_contains_request_and_schema(self):
        text = _build_generation_prompt("make an xor challenge", "easy", None, None)
        self.assertIn("make an xor challenge", text)
        self.assertIn('"encoder_code"', text)
        self.assertIn('"challenge_code"', text)
        self.assertIn(ENCODED_PLACEHOLDER, text)

    def test_feedback_is_included_for_repair(self):
        text = _build_generation_prompt("x", "easy", None, "gcc failed: missing semicolon")
        self.assertIn("previous attempt failed", text.lower())
        self.assertIn("missing semicolon", text)

    def test_difficulty_rubric_in_prompt(self):
        from slothctf.generator import _SCHEMA_RULES_LOG
        hard = _build_generation_prompt("x", "hard", None, None)
        self.assertIn("Target difficulty: hard", hard)
        self.assertIn("layered", hard.lower())  # reverse hard rubric
        easy_log = _build_generation_prompt("x", "easy", None, None, schema=_SCHEMA_RULES_LOG)
        self.assertIn("single line", easy_log.lower())  # log easy rubric

    def test_requested_difficulty_overrides_model(self):
        from slothctf.generator import _normalize_challenge
        data = _normalize_challenge({"difficulty": "hard", "flag": "x"}, "easy", None, "reverse")
        self.assertEqual(data["difficulty"], "easy")

    def test_coerce_json_from_fenced_block(self):
        raw = 'Sure!\n```json\n{"flag": "sloth{a}"}\n```\n'
        self.assertEqual(_coerce_json(raw)["flag"], "sloth{a}")

    def test_strip_code_fences(self):
        self.assertEqual(_strip_code_fences("```c\nint main(){}\n```"), "int main(){}")
        self.assertEqual(_strip_code_fences("int main(){}"), "int main(){}")
        # stray opening fence only
        self.assertEqual(_strip_code_fences("```c\nint x;"), "int x;")

    def test_schema_does_not_request_prose_fields(self):
        # The repair loop must stay cheap: the model is not asked for README or
        # the walkthrough (those are rendered locally on success).
        self.assertNotIn('"readme"', _SCHEMA_RULES)
        self.assertNotIn('"solution"', _SCHEMA_RULES)
        self.assertIn('"encoder_code"', _SCHEMA_RULES)
        self.assertIn('"challenge_code"', _SCHEMA_RULES)

    def test_schema_describes_both_kinds(self):
        self.assertIn('"decode"', _SCHEMA_RULES)
        self.assertIn('"check"', _SCHEMA_RULES)
        self.assertIn("success_marker", _SCHEMA_RULES)

    def test_check_readme_mentions_stdin(self):
        data = {
            "title": "Pwd", "category": "RE", "difficulty": "easy",
            "description": "d", "flag": "sloth{p}", "challenge_kind": "check",
            "success_marker": "Access Granted", "encoder_code": "x",
        }
        self.assertIn("standard input", _render_readme(data).lower())
        self.assertIn("Access Granted", _render_solution(data))

    def test_parse_encoder_output(self):
        self.assertEqual(_parse_encoder_output("0x31,0x2e,0x2d,"), "0x31, 0x2e, 0x2d")
        self.assertEqual(_parse_encoder_output("junk 0xAB more 0x0f"), "0xab, 0x0f")
        self.assertIsNone(_parse_encoder_output("no bytes here"))

    def test_prose_is_rendered_locally(self):
        data = {
            "title": "XOR Lock",
            "category": "Reverse Engineering",
            "difficulty": "easy",
            "description": "Find the key.",
            "flag": "sloth{ok}",
            "encoder_code": "int main(){/*encoder*/}",
        }
        readme = _render_readme(data)
        solution = _render_solution(data)
        self.assertIn("XOR Lock", readme)
        self.assertIn("sloth{ok}", solution)
        self.assertIn("/*encoder*/", solution)

    def test_endpoint_from_base_url(self):
        self.assertEqual(_generate_endpoint("http://10.0.0.2:11434"), "http://10.0.0.2:11434/api/generate")
        self.assertEqual(_generate_endpoint("http://10.0.0.2:11434/"), "http://10.0.0.2:11434/api/generate")
        self.assertEqual(
            _generate_endpoint("http://h:11434/api/generate"), "http://h:11434/api/generate"
        )


class GenerationLoopTests(unittest.TestCase):
    """The retry loop is tested with the compiler and verifier stubbed so the
    tests need neither Ollama nor gcc."""

    def _run(self, responses, verify_side_effect, max_attempts=5):
        calls = {"n": 0}

        def fake_caller(prompt, model):
            idx = calls["n"]
            calls["n"] += 1
            return responses[min(idx, len(responses) - 1)]

        with mock.patch.object(generator, "_find_compiler", return_value="gcc"), \
             mock.patch.object(generator, "_write_challenge_files"), \
             mock.patch.object(generator, "_verify_challenge", side_effect=verify_side_effect):
            return generate_challenge(
                "unit_test_challenge",
                prompt="make a challenge",
                caller=fake_caller,
                max_attempts=max_attempts,
                return_report=True,
            )

    def test_succeeds_on_first_attempt(self):
        good = {"encoder_code": "int main(){}", "challenge_code": "int main(){}", "flag": "sloth{x}"}
        report = self._run([good], lambda *a, **k: AttemptResult(0, True, "passed", "flag recovered"))
        self.assertTrue(report.success)
        self.assertEqual(report.attempts_used, 1)

    def test_repairs_then_succeeds(self):
        payload = {"encoder_code": "e", "challenge_code": "c", "flag": "sloth{y}"}
        outcomes = iter([
            AttemptResult(0, False, "challenge-compile", "gcc failed"),
            AttemptResult(0, True, "passed", "ok"),
        ])
        report = self._run([payload, payload], lambda *a, **k: next(outcomes))
        self.assertTrue(report.success)
        self.assertEqual(report.attempts_used, 2)
        self.assertEqual(report.attempts[0].stage, "challenge-compile")

    def test_gives_up_after_max_attempts(self):
        payload = {"encoder_code": "e", "challenge_code": "c", "flag": "sloth{z}"}
        report = self._run(
            [payload],
            lambda *a, **k: AttemptResult(0, False, "flag-mismatch", "nope"),
            max_attempts=3,
        )
        self.assertFalse(report.success)
        self.assertEqual(report.attempts_used, 3)

    def test_missing_schema_field_is_fed_back(self):
        bad = {"encoder_code": "e"}  # no challenge_code
        report = self._run([bad], lambda *a, **k: AttemptResult(0, True, "passed"), max_attempts=2)
        self.assertFalse(report.success)
        self.assertEqual(report.attempts[0].stage, "bad-schema")

    def test_raises_without_compiler(self):
        with mock.patch.object(generator, "_find_compiler", return_value=None):
            with self.assertRaises(GenerationError):
                generate_challenge("x", caller=lambda p, m: {}, return_report=True)


class LogVerifyTests(unittest.TestCase):
    """Exercises the real multi-task log synthesis verifier (no model)."""

    # Generator PRINTS a known ALERT line plus noise; count is a computed task.
    GOOD_GEN = "print('noise\\nALERT from 10.0.0.9\\nhit\\nhit')\n"
    EXT_IP = ("import sys,re\n"
              "print(re.search(r'ALERT from (\\S+)', open(sys.argv[1]).read()).group(1))\n")
    EXT_COUNT = ("import sys\n"
                 "print(sum(1 for l in open(sys.argv[1]) if 'hit' in l))\n")

    def _tasks(self):
        return [
            {"question": "attacker IP?", "type": "extractive", "answer": "10.0.0.9",
             "evidence_line": 2, "evidence_text": "ALERT from 10.0.0.9", "extractor_code": self.EXT_IP},
            {"question": "how many hits?", "type": "computed", "answer": "2", "extractor_code": self.EXT_COUNT},
        ]

    def _run(self, data):
        with TemporaryDirectory() as tmp:
            return generator._verify_log_challenge(Path(tmp) / "c", data)

    def test_passes_multi_task(self):
        data = {"generator_code": self.GOOD_GEN, "tasks": self._tasks()}
        r = self._run(data)
        self.assertTrue(r.ok, r.detail)
        self.assertEqual(len(data["tasks"]), 2)  # both kept

    def test_keeps_good_drops_bad(self):
        tasks = self._tasks(); tasks[1]["answer"] = "99"  # 2nd task wrong
        data = {"generator_code": self.GOOD_GEN, "tasks": tasks}
        r = self._run(data)
        self.assertTrue(r.ok, r.detail)          # still valid...
        self.assertEqual(len(data["tasks"]), 1)  # ...with only the good task kept

    def test_all_bad_fails(self):
        tasks = self._tasks()
        tasks[0]["extractor_code"] = "import sys\nprint('10.0.0.9')\n"  # hardcoded -> dropped
        tasks[1]["answer"] = "99"                                        # mismatch -> dropped
        r = self._run({"generator_code": self.GOOD_GEN, "tasks": tasks})
        self.assertFalse(r.ok)
        self.assertEqual(r.stage, "no-valid-tasks")

    def test_generator_must_print_log(self):
        r = self._run({"generator_code": "pass\n", "tasks": self._tasks()})
        self.assertFalse(r.ok)
        self.assertEqual(r.stage, "no-log")

    def test_no_tasks(self):
        r = self._run({"generator_code": self.GOOD_GEN, "tasks": []})
        self.assertFalse(r.ok)
        self.assertEqual(r.stage, "no-tasks")


class LogAnalysisVerifyTests(unittest.TestCase):
    """Exercises the provided-log multi-task analysis verifier."""

    LOG = ("2026-03-01 sshd: Failed password from 10.0.0.2\n"
           "2026-03-01 sshd: Accepted password from 10.0.0.42\n"
           "2026-03-01 sshd: Failed password from 10.0.0.3\n")
    EXT = ("import sys,re\n"
           "print(re.search(r'Accepted password from (\\S+)', open(sys.argv[1]).read()).group(1))\n")

    def _run(self, tasks):
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "provided.log"
            log.write_text(self.LOG, encoding="utf-8")
            return generator._verify_log_analysis(Path(tmp) / "c", {"tasks": tasks}, log)

    def test_passes_when_grounded(self):
        r = self._run([{"type": "extractive", "answer": "10.0.0.42", "evidence_line": 2,
                        "evidence_text": "2026-03-01 sshd: Accepted password from 10.0.0.42",
                        "extractor_code": self.EXT}])
        self.assertTrue(r.ok, r.detail)

    def test_not_grounded(self):
        # single bad task -> nothing survives
        r = self._run([{"type": "extractive", "answer": "9.9.9.9", "evidence_line": 2, "extractor_code": self.EXT}])
        self.assertFalse(r.ok)
        self.assertEqual(r.stage, "no-valid-tasks")

    def test_bad_evidence_line(self):
        r = self._run([{"type": "extractive", "answer": "10.0.0.42", "evidence_line": 1, "extractor_code": self.EXT}])
        self.assertFalse(r.ok)
        self.assertEqual(r.stage, "no-valid-tasks")
        self.assertIn("evidence_line", r.detail)

    def test_hardcoded_rejected(self):
        hard = "import sys\nprint('10.0.0.42')\n"
        r = self._run([{"type": "extractive", "answer": "10.0.0.42", "evidence_line": 2, "extractor_code": hard}])
        self.assertFalse(r.ok)
        self.assertEqual(r.stage, "no-valid-tasks")
        self.assertIn("hardcoded", r.detail)


if __name__ == "__main__":
    unittest.main()
