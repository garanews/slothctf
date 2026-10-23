"""Offline tests for the recover measurement campaign runner.

Pure-function tests need no Docker (classification, transform/shortcut heuristics,
aggregation, anonymisation, resume bookkeeping, and simulated model/infra
failures via a caller that raises before any backend is touched). A gated test
drives a real generation through Docker with a replayed response, and exercises
interruption + resume.
"""
import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from slothctf.execution import docker_available, image_available

_MOD_PATH = Path(__file__).resolve().parent.parent / "tools" / "run_recover_campaign.py"
_spec = importlib.util.spec_from_file_location("run_recover_campaign", _MOD_PATH)
camp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(camp)

_READY = docker_available() and image_available()
_SKIP = "needs Docker + the slothctf-sandbox image"

# A working XOR recover response (same mechanic proven in the recover tests).
_ENC = ("#include <stdio.h>\nint main(int c,char**v){for(char*p=v[1];*p;++p)"
        "printf(\"0x%02x,\",(unsigned char)(*p^0x5a));return 0;}\n")
_CH = ("#include <stdio.h>\n#include <string.h>\n"
       "static unsigned char data[]={ /*__ENCODED_ARRAY__*/ };\n"
       "int main(void){char b[256];if(!fgets(b,sizeof b,stdin)){printf(\"Nope\\n\");return 0;}"
       "size_t n=strlen(b);if(n&&b[n-1]=='\\n')b[--n]=0;"
       "if(n!=sizeof data){printf(\"Nope\\n\");return 0;}"
       "for(size_t i=0;i<n;i++)if(((unsigned char)b[i]^0x5a)!=data[i]){printf(\"Nope\\n\");return 0;}"
       "printf(\"Unlocked\\n\");return 0;}\n")
_SOL = ("import sys, re\nb=open(sys.argv[1],'rb').read()\n"
        "dec=bytes(x^0x5a for x in b)\nm=re.search(rb'sloth\\{[^}]{1,64}\\}',dec)\n"
        "print(m.group(0).decode() if m else '')\n")


def _good_response(flag):
    return {"title": "C", "category": "Reverse Engineering", "difficulty": "easy",
            "description": "d", "flag": flag, "challenge_kind": "recover",
            "success_marker": "Unlocked", "encoder_code": _ENC,
            "challenge_code": _CH, "solver_code": _SOL}


def _one_scenario_campaign(reps=1):
    return {"name": "t", "model": "x", "reps": reps, "max_attempts": 2,
            "scenarios": [{"id": "elf-easy", "target": "elf", "difficulty": "easy", "prompt": "p"}]}


class SpecsAndResumeTests(unittest.TestCase):
    def test_run_specs_expands_reps(self):
        camp_cfg = camp.load_campaign(
            Path(__file__).resolve().parent.parent / "tools" / "recover_campaign" / "scenarios.json")
        specs = camp.run_specs(camp_cfg)
        self.assertEqual(len(specs), 18)                      # 6 scenarios x 3 reps
        self.assertEqual(len({s["run_id"] for s in specs}), 18)

    def test_completed_run_ids_and_append(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "raw.jsonl"
            self.assertEqual(camp.completed_run_ids(p), set())
            camp.append_record(p, {"run_id": "a#rep1", "success": True})
            camp.append_record(p, {"run_id": "b#rep1", "success": False})
            self.assertEqual(camp.completed_run_ids(p), {"a#rep1", "b#rep1"})

    def test_completed_ignores_torn_last_line(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "raw.jsonl"
            p.write_text('{"run_id": "a#rep1"}\n{"run_id": "b#rep1"}\n{"run_id": "c#re',
                         encoding="utf-8")
            self.assertEqual(camp.completed_run_ids(p), {"a#rep1", "b#rep1"})


class ClassifyTests(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(camp.classify({"success": True}), "ok")
        self.assertEqual(camp.classify({"success": False, "exception": "RuntimeError"}), "infra")
        self.assertEqual(camp.classify({"success": False, "fail_stage": "model-error"}), "infra")
        self.assertEqual(camp.classify({"success": False, "fail_stage": "challenge-timeout"}), "infra")
        self.assertEqual(camp.classify({"success": False, "fail_stage": "solver-hardcoded"}), "model-reject")


class HeuristicTests(unittest.TestCase):
    def test_transform_family_tags(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "organizer").mkdir()
            (d / "organizer" / "encoder.c").write_text("x = c ^ key;", encoding="utf-8")
            (d / "organizer" / "solver.py").write_text("v = (b[i] ^ (0x3b + i))\n", encoding="utf-8")
            t = camp.analyze_transform(d)
            self.assertIn("xor", t["families"])
            self.assertIn("position_dependent", t["families"])

    def test_naive_attacker_finds_single_xor(self):
        blob = b"\x00\x01" + bytes(c ^ 0x5a for c in b"sloth{easy}") + b"\x02"
        hit = camp.naive_attacker(blob)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["flag"], "sloth{easy}")

    def test_naive_attacker_misses_position_dependent(self):
        flag = b"sloth{not_single_op}"
        blob = b"junk" + bytes(((flag[i] + 7 * (i + 1)) ^ (0x3b + i)) & 0xFF for i in range(len(flag)))
        self.assertIsNone(camp.naive_attacker(blob))

    def test_entropy_bounds(self):
        self.assertEqual(camp._shannon_entropy(b""), 0.0)
        self.assertEqual(camp._shannon_entropy(b"\x00" * 10), 0.0)
        self.assertGreater(camp._shannon_entropy(bytes(range(256))), 7.9)


class AggregateAnonymizeTests(unittest.TestCase):
    def _records(self):
        return [
            {"run_id": "elf-easy#rep1", "scenario": "elf-easy", "target": "elf", "difficulty": "easy",
             "rep": 1, "success": True, "category": "ok", "attempts_used": 1, "total_time_s": 5,
             "flag": "sloth{a}", "quality": {"family_signature": "xor", "strings_shortcut": False,
                                             "naive_attacker_shortcut": True}},
            {"run_id": "pe-hard#rep1", "scenario": "pe-hard", "target": "pe", "difficulty": "hard",
             "rep": 1, "success": True, "category": "ok", "attempts_used": 2, "total_time_s": 40,
             "flag": "sloth{b}", "quality": {"family_signature": "sbox_table+xor",
                                             "strings_shortcut": False, "naive_attacker_shortcut": False}},
            {"run_id": "elf-hard#rep1", "scenario": "elf-hard", "target": "elf", "difficulty": "hard",
             "rep": 1, "success": False, "category": "model-reject", "fail_stage": "solver-mismatch",
             "attempts_used": 2, "total_time_s": 9},
        ]

    def test_aggregate(self):
        s = camp.aggregate(self._records())
        self.assertEqual(s["total_runs"], 3)
        self.assertEqual(s["success_rate_pct"], 66.7)
        self.assertEqual(s["distinct_family_signatures"], 2)
        self.assertEqual(s["obvious_shortcuts"]["naive_single_op"], 1)
        self.assertEqual(s["category_counts"]["ok"], 2)
        self.assertEqual(s["success_by_difficulty"]["hard"], 50.0)
        md = camp.render_markdown(s)
        self.assertIn("technical success rate", md)
        self.assertIn("heuristic", md.lower())

    def test_csv_roundtrip(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.csv"
            camp.write_csv(self._records(), p)
            rows = p.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 4)  # header + 3
            self.assertIn("family_signature", rows[0])

    def test_anonymize_scrubs_without_mutating_original(self):
        recs = self._records()
        recs[0]["provenance"] = {"platform": "Windows-XYZ", "model": "qwen"}
        anon = camp.anonymize(recs)
        self.assertTrue(anon[0]["flag"].startswith("sha256:"))
        self.assertNotIn("platform", anon[0]["provenance"])
        # original untouched
        self.assertEqual(recs[0]["flag"], "sloth{a}")
        self.assertIn("platform", recs[0]["provenance"])


class ReplayAndFailureTests(unittest.TestCase):
    """These touch the driver but the replayed caller raises BEFORE any backend is
    used, so no Docker is needed."""

    def test_model_error_is_infra_category(self):
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "raw.jsonl"
            factory = camp.replay_caller_factory(
                {"elf-easy#rep1": [{"__raise__": "GenerationError", "msg": "ollama down"}]})
            recs = camp.run_campaign(_one_scenario_campaign(), tmp, jsonl,
                                     caller_factory=factory, log=lambda *a: None)
            self.assertEqual(len(recs), 1)
            self.assertFalse(recs[0]["success"])
            self.assertEqual(recs[0]["fail_stage"], "model-error")
            self.assertEqual(recs[0]["category"], "infra")

    def test_infra_exception_recorded(self):
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "raw.jsonl"
            factory = camp.replay_caller_factory(
                {"elf-easy#rep1": [{"__raise__": "RuntimeError", "msg": "boom"}]})
            recs = camp.run_campaign(_one_scenario_campaign(), tmp, jsonl,
                                     caller_factory=factory, log=lambda *a: None)
            self.assertEqual(recs[0]["category"], "infra")
            self.assertEqual(recs[0]["exception"], "RuntimeError")

    def test_resume_skips_completed(self):
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "raw.jsonl"
            factory = camp.replay_caller_factory(
                {"*": [{"__raise__": "GenerationError", "msg": "x"}]})
            camp.run_campaign(_one_scenario_campaign(), tmp, jsonl, caller_factory=factory, log=lambda *a: None)
            # second call: the single run is already recorded -> nothing new produced
            again = camp.run_campaign(_one_scenario_campaign(), tmp, jsonl, caller_factory=factory, log=lambda *a: None)
            self.assertEqual(again, [])
            self.assertEqual(len(camp.completed_run_ids(jsonl)), 1)


@unittest.skipUnless(_READY, _SKIP)
class LiveCampaignTests(unittest.TestCase):
    def test_real_generation_and_quality(self):
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "raw.jsonl"
            factory = camp.replay_caller_factory({"elf-easy#rep1": [_good_response("sloth{camp_ok}")]})
            recs = camp.run_campaign(_one_scenario_campaign(), tmp, jsonl,
                                     caller_factory=factory, log=lambda *a: None)
            self.assertEqual(len(recs), 1)
            r = recs[0]
            self.assertTrue(r["success"], r.get("fail_stage"))
            self.assertEqual(r["category"], "ok")
            self.assertIn("xor", r["quality"]["families"])
            self.assertFalse(r["quality"]["strings_shortcut"])

    def test_interruption_then_resume(self):
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "raw.jsonl"

            class _Boom(KeyboardInterrupt):
                pass

            def interrupting_factory(spec):
                if spec["run_id"].endswith("rep2"):
                    def c(prompt, model):
                        raise _Boom()
                    return c
                return camp.replay_caller_factory({"*": [_good_response("sloth{r1}")]})(spec)

            with self.assertRaises(KeyboardInterrupt):
                camp.run_campaign(_one_scenario_campaign(reps=2), tmp, jsonl,
                                  caller_factory=interrupting_factory, log=lambda *a: None)
            # rep1 recorded, rep2 interrupted (not recorded)
            self.assertEqual(camp.completed_run_ids(jsonl), {"elf-easy#rep1"})
            # resume with a good response for rep2
            good = camp.replay_caller_factory({"*": [_good_response("sloth{r2}")]})
            recs = camp.run_campaign(_one_scenario_campaign(reps=2), tmp, jsonl,
                                     caller_factory=good, log=lambda *a: None)
            self.assertEqual([r["run_id"] for r in recs], ["elf-easy#rep2"])  # only the missing one
            self.assertEqual(len(camp.completed_run_ids(jsonl)), 2)           # no duplicates


if __name__ == "__main__":
    unittest.main()
