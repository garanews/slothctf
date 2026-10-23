import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import subprocess

from slothctf import generator
from slothctf.generator import (
    AttemptResult,
    ENCODED_PLACEHOLDER,
    GenerationError,
    _build_generation_prompt,
    _coerce_json,
    _compile_binary,
    _flag_leaks_in_binary,
    _generate_endpoint,
    _normalize_flag,
    _parse_encoder_output,
    _render_readme,
    _render_solution,
    _run_binary,
    _SCHEMA_RULES,
    _strip_code_fences,
    generate_challenge,
)


class RunBinaryDecodeTests(unittest.TestCase):
    """Regression: a challenge binary may print bytes that are not valid in the
    host's default codec (the campaign hit 0x9d, which cp1252 can't decode and a
    non-invertible transform emits on a random round-trip probe). _run_binary must
    decode robustly and return a string, never raise UnicodeDecodeError or return
    None. See generator._run_binary / _roundtrip_check."""

    def test_non_decodable_byte_does_not_crash(self):
        # Use the Python interpreter as a stand-in "binary" that writes a raw 0x9d
        # byte to stdout — no compiler needed, reproduces the exact failing case.
        ran, out, _err = _run_binary(
            Path(sys.executable),
            ["-c", "import sys; sys.stdout.buffer.write(b'sloth{x}\\x9d')"],
        )
        self.assertTrue(ran)
        self.assertIsInstance(out, str)           # decoded, not None
        self.assertIn("sloth{x}", out)            # the valid part survives
        # the replacement never equals a clean flag, so a wrong decode is a mismatch
        self.assertNotEqual(out.strip(), "sloth{x}")


class CompileTimeoutTests(unittest.TestCase):
    """A model can emit C that makes the compiler hang; _compile_binary must bound
    it with a timeout and return a clean failure, never block or raise."""

    def test_compile_timeout_is_a_clean_failure(self):
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "challenge.c"
            src.write_text("int main(void){return 0;}", encoding="utf-8")
            out = Path(tmp) / "bin" / "challenge"
            with mock.patch.object(
                generator.subprocess, "run",
                side_effect=subprocess.TimeoutExpired(cmd="gcc", timeout=generator._COMPILE_TIMEOUT),
            ):
                ok, stderr, produced = _compile_binary(src, out, "gcc")
            self.assertFalse(ok)
            self.assertIn("timed out", stderr)
            self.assertEqual(produced, out)


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

    def test_flag_leaks_raises_when_unreadable(self):
        from slothctf.generator import _flag_leaks_in_binary
        with TemporaryDirectory() as tmp:
            self.assertRaises(OSError, _flag_leaks_in_binary, Path(tmp) / "nope", "sloth{x}")

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

    def test_adversary_not_applicable_to_decode(self):
        from slothctf.generator import _adversary_check, _COMMON_SECRETS
        from slothctf.execution import NativeBackend
        from slothctf.targets import host_native_target
        be, tg = NativeBackend(), host_native_target()
        self.assertIn("password", _COMMON_SECRETS)
        # decode / missing marker -> gate does not apply (returns None, no run)
        self.assertIsNone(_adversary_check(Path("nope"), "bin/x", be, tg, "decode", "", "hard"))
        self.assertIsNone(_adversary_check(Path("nope"), "bin/x", be, tg, "check", "", "hard"))

    def test_default_flag_policy(self):
        from slothctf.generator import _random_flag
        rf = _random_flag()
        self.assertTrue(rf.startswith("sloth{") and rf.endswith("}"))
        text = _build_generation_prompt("x", "easy", None, None, default_flag=rf)
        self.assertIn("FLAG POLICY", text)
        self.assertIn(rf, text)
        # without a default flag, no policy line
        self.assertNotIn("FLAG POLICY", _build_generation_prompt("x", "easy", None, None))

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

        from slothctf.execution import NativeBackend
        with mock.patch.object(generator, "_write_challenge_files"), \
             mock.patch.object(generator, "_verify_challenge", side_effect=verify_side_effect):
            return generate_challenge(
                "unit_test_challenge",
                prompt="make a challenge",
                caller=fake_caller,
                max_attempts=max_attempts,
                backend=NativeBackend(),   # explicit: these tests mock verify, no real exec
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

    def test_no_backend_without_docker_refuses(self):
        # No backend + Docker unavailable => explicit error, never a silent native run.
        from slothctf.execution import SandboxUnavailable
        with mock.patch("slothctf.execution.docker_available", return_value=False):
            with self.assertRaises(SandboxUnavailable):
                generate_challenge("x", target="elf", caller=lambda p, m: {}, return_report=True)

    def test_isolated_pe_without_win_image_refused(self):
        # Phase B: isolated PE needs the MinGW+Wine image. Without it => explicit
        # error, never a silent unsandboxed .exe build and never converted to ELF.
        from slothctf.execution import SandboxUnavailable
        with mock.patch("slothctf.execution.docker_available", return_value=True), \
             mock.patch("slothctf.execution.win_image_available", return_value=False):
            with self.assertRaises(SandboxUnavailable):
                generate_challenge("x", target="pe", caller=lambda p, m: {}, return_report=True)


class AdversaryBudgetTests(unittest.TestCase):
    def test_invalid_budgets_raise(self):
        for bad in [(0, 0), (-10, -5), (100, float("nan")), (5, 0), (5, float("inf"))]:
            with self.assertRaises(ValueError):
                generate_challenge("x", adversary_budget=bad, return_report=True)

    def test_default_budget_scales_with_difficulty(self):
        from slothctf.generator import _ADVERSARY_BUDGET
        self.assertLess(_ADVERSARY_BUDGET["medium"][0], _ADVERSARY_BUDGET["hard"][0])
        self.assertLess(_ADVERSARY_BUDGET["medium"][1], _ADVERSARY_BUDGET["hard"][1])

    @unittest.skipUnless(generator._find_compiler() is not None, "needs a C compiler")
    def test_budget_is_a_real_knob(self):
        # Deterministic, machine-speed-independent check that the attempt budget is
        # a real gate. The reference attacker tries each common secret first as the
        # bare word and then wrapped as sloth{word} (generator._adversary_check over
        # _COMMON_SECRETS). For a flag whose secret is the FIRST common secret:
        #   candidate #1 = bare word  -> wrong length -> "Denied"
        #   candidate #2 = sloth{word} -> the exact flag -> "Access Granted" (cracked)
        # So an attempt budget of 1 never reaches the crack; a budget of 2 does.
        # We pass an effectively infinite TIME budget so only the attempt count
        # decides — the earlier version used a 60s wall-clock budget, which made the
        # verdict depend on how fast the host spawned ~1000 real subprocesses (flaky
        # under load). The production adversary is untouched; only the test's
        # explicit budget differs, and this runs ~2 subprocesses instead of ~1800.
        import tempfile
        from slothctf.generator import _COMMON_SECRETS
        secret = _COMMON_SECRETS[0]
        flag = f"sloth{{{secret}}}"
        enc = ('#include <stdio.h>\nint main(int c,char**v){for(char*p=v[1];*p;++p)'
               'printf("0x%02x,",(unsigned char)(*p^0x37));}')
        ch = ('#include <stdio.h>\n#include <string.h>\nstatic unsigned char t[]={ '
              '/*__ENCODED_ARRAY__*/ };\nint main(void){char b[128];if(!fgets(b,sizeof b,stdin))'
              'return 1;size_t n=strlen(b);if(n&&b[n-1]==10)b[--n]=0;if(n!=sizeof t)'
              '{printf("Denied\\n");return 0;}for(size_t i=0;i<n;i++)'
              'if(((unsigned char)b[i]^0x37)!=t[i]){printf("Denied\\n");return 0;}'
              'printf("Access Granted\\n");return 0;}')
        from slothctf.execution import NativeBackend
        from slothctf.targets import host_native_target
        be, tg = NativeBackend(), host_native_target()

        NO_TIME_LIMIT = 10 ** 9  # seconds; the attempt count is the only real bound

        def mk():
            return generator._normalize_challenge(
                {"title": "B", "category": "Reverse Engineering", "difficulty": "hard",
                 "flag": flag, "challenge_kind": "check", "success_marker": "Access Granted",
                 "encoder_code": enc, "challenge_code": ch}, "hard", None, "reverse")

        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            small = generator._verify_challenge(Path(t1), mk(), be, tg, adversary=True,
                                                adversary_budget=(1, NO_TIME_LIMIT))
            large = generator._verify_challenge(Path(t2), mk(), be, tg, adversary=True,
                                                adversary_budget=(2, NO_TIME_LIMIT))
        self.assertTrue(small.ok, small.detail)      # 1 attempt: never reaches sloth{word}
        self.assertEqual(large.stage, "too-easy")    # 2 attempts: the dictionary cracks it


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
        from slothctf.execution import NativeBackend
        with TemporaryDirectory() as tmp:
            return generator._verify_log_challenge(Path(tmp) / "c", data, NativeBackend())

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
            from slothctf.execution import NativeBackend
            return generator._verify_log_analysis(Path(tmp) / "c", {"tasks": tasks}, log, NativeBackend())

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
