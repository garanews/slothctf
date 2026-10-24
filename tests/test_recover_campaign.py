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
        # genuine infra = a caught backend exception
        self.assertEqual(camp.classify({"success": False, "exception": "SandboxUnavailable"}), "infra")
        # ambiguous causes are 'undetermined', NOT auto-infra
        for stg in ("solver-error", "swap-solver-error", "challenge-timeout", "model-error"):
            self.assertEqual(camp.classify({"success": False, "fail_stage": stg}), "undetermined", stg)
        # concrete content failures are attributable -> verification-failure
        for stg in ("solver-hardcoded", "solver-mismatch", "challenge-compile",
                    "recover-accepts-wrong", "too-trivial", "bad-schema"):
            self.assertEqual(camp.classify({"success": False, "fail_stage": stg}), "verification-failure", stg)


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

    def test_naive_attacker_recovers_exact_flag(self):
        blob = b"\x00\x01" + bytes(c ^ 0x5a for c in b"sloth{easy}") + b"\x02"
        hit = camp.naive_attacker(blob, "sloth{easy}")
        self.assertTrue(hit["recovered_expected"])
        self.assertEqual(hit["op"], "xor")

    def test_naive_attacker_candidate_not_expected_is_not_shortcut(self):
        # a single-XOR reveals SOME sloth{...} but NOT the expected flag -> not a
        # shortcut (false positive avoided); the candidate is recorded instead.
        blob = b"\x00" + bytes(c ^ 0x5a for c in b"sloth{other}") + b"\x00"
        hit = camp.naive_attacker(blob, "sloth{the_real_one}")
        self.assertFalse(hit["recovered_expected"])
        self.assertIsNotNone(hit["candidate"])
        self.assertEqual(hit["candidate"]["text"], "sloth{other}")

    def test_naive_attacker_misses_position_dependent(self):
        flag = b"sloth{not_single_op}"
        blob = b"junk" + bytes(((flag[i] + 7 * (i + 1)) ^ (0x3b + i)) & 0xFF for i in range(len(flag)))
        hit = camp.naive_attacker(blob, flag.decode())
        self.assertFalse(hit["recovered_expected"])

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
             "rep": 1, "success": False, "category": "verification-failure", "fail_stage": "solver-mismatch",
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
        # per-scenario and failure-by-stage distributions
        self.assertEqual(s["per_scenario"]["elf-easy"], {"ok": 1, "total": 1, "rate_pct": 100.0})
        self.assertEqual(s["per_scenario"]["elf-hard"], {"ok": 0, "total": 1, "rate_pct": 0.0})
        self.assertEqual(s["fail_stage_histogram"].get("solver-mismatch"), 1)
        self.assertEqual(s["attempts_histogram"].get("2"), 2)
        md = camp.render_markdown(s)
        self.assertIn("Technical metrics", md)
        self.assertIn("Heuristics (NOT ground truth)", md)
        self.assertIn("Success per scenario", md)
        self.assertIn("Failure distribution by stage", md)

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

    def test_anonymize_scrubs_flag_and_paths_from_diagnostics(self):
        # flag + inner secret + Windows paths planted in diagnostic fields must all
        # be scrubbed from the FULL anonymised JSONL, not just the `flag` field.
        rec = {
            "run_id": "elf-easy#rep1", "scenario": "elf-easy", "target": "elf",
            "difficulty": "easy", "rep": 1, "success": False, "category": "model-reject",
            "fail_stage": "solver-mismatch", "flag": "sloth{leaky_secret_123}",
            "detail": "solver recovered 'sloth{leaky_secret_123}' at C:\\Users\\garan\\gen\\challenge.c",
            "quality_error": "KeyError leaky_secret_123 while reading C:\\Temp\\x\\challenge.json",
            "quality": {"naive_attacker": {"candidate": {"text": "sloth{leaky_secret_123}"}}},
        }
        anon = camp.anonymize([rec])
        blob = json.dumps(anon[0])
        self.assertNotIn("leaky_secret_123", blob)          # flag + inner gone everywhere
        self.assertNotIn("sloth{leaky_secret_123}", blob)
        self.assertNotIn("C:\\\\Users", json.dumps(anon[0]))  # windows paths redacted
        self.assertNotIn("C:\\Temp", anon[0]["quality_error"])
        self.assertIn("<flag>", anon[0]["detail"])
        self.assertIn("<path>", anon[0]["detail"])
        # raw record untouched
        self.assertIn("leaky_secret_123", rec["detail"])


class ReplayAndFailureTests(unittest.TestCase):
    """These touch the driver but the replayed caller raises BEFORE any backend is
    used, so no Docker is needed."""

    def test_model_error_is_undetermined_not_infra(self):
        # a failed model call (GenerationError) is ambiguous — could be the provider
        # down OR unparseable output — so it is 'undetermined', never auto-'infra'.
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "raw.jsonl"
            factory = camp.replay_caller_factory(
                {"elf-easy#rep1": [{"__raise__": "GenerationError", "msg": "ollama down"}]})
            recs = camp.run_campaign(_one_scenario_campaign(), tmp, jsonl,
                                     caller_factory=factory, log=lambda *a: None)
            self.assertEqual(len(recs), 1)
            self.assertFalse(recs[0]["success"])
            self.assertEqual(recs[0]["fail_stage"], "model-error")
            self.assertEqual(recs[0]["category"], "undetermined")

    def test_infra_exception_recorded(self):
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "raw.jsonl"
            factory = camp.replay_caller_factory(
                {"elf-easy#rep1": [{"__raise__": "RuntimeError", "msg": "boom"}]})
            recs = camp.run_campaign(_one_scenario_campaign(), tmp, jsonl,
                                     caller_factory=factory, log=lambda *a: None)
            self.assertEqual(recs[0]["category"], "infra")
            self.assertEqual(recs[0]["exception"], "RuntimeError")

    def test_fingerprint_mismatch_refuses_resume(self):
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "raw.jsonl"
            factory = camp.replay_caller_factory({"*": [{"__raise__": "GenerationError", "msg": "x"}]})
            camp.run_campaign(_one_scenario_campaign(), tmp, jsonl, caller_factory=factory, log=lambda *a: None)
            # change the frozen prompt -> different fingerprint -> must refuse
            changed = _one_scenario_campaign()
            changed["scenarios"][0]["prompt"] = "a DIFFERENT prompt"
            with self.assertRaises(RuntimeError):
                camp.run_campaign(changed, tmp, jsonl, caller_factory=factory, log=lambda *a: None)

    def test_fingerprint_depends_on_model_digest(self):
        # Same model NAME, different DIGEST (weights re-pulled under the same tag) ->
        # different fingerprint, so the runner will not mix them in one dataset.
        campaign = _one_scenario_campaign()
        prov_a = {"model": "qwen2.5-coder:32b", "model_digest": "sha256:aaaa",
                  "inference": {"temperature": 0.7}, "tool_version": "1.0", "git_commit": "c0"}
        prov_b = dict(prov_a, model_digest="sha256:bbbb")  # only the digest changes
        self.assertNotEqual(camp.campaign_fingerprint(campaign, prov_a),
                            camp.campaign_fingerprint(campaign, prov_b))

    def test_model_digest_change_blocks_resume(self):
        # End-to-end: a dataset produced with digest A must refuse to resume once the
        # same tag resolves to digest B (the model was updated under us).
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "raw.jsonl"
            factory = camp.replay_caller_factory({"*": [{"__raise__": "GenerationError", "msg": "x"}]})
            orig = camp._ollama_model_digest
            try:
                camp._ollama_model_digest = lambda model, url: "sha256:AAAA"
                camp.run_campaign(_one_scenario_campaign(), tmp, jsonl,
                                  caller_factory=factory, online=True, log=lambda *a: None)
                camp._ollama_model_digest = lambda model, url: "sha256:BBBB"  # re-pulled
                with self.assertRaises(RuntimeError):
                    camp.run_campaign(_one_scenario_campaign(), tmp, jsonl,
                                      caller_factory=factory, online=True, log=lambda *a: None)
            finally:
                camp._ollama_model_digest = orig

    def test_online_campaign_requires_model_digest(self):
        # Online but the digest cannot be resolved -> refuse to start rather than
        # record runs that are not fully identifiable.
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "raw.jsonl"
            factory = camp.replay_caller_factory({"*": [{"__raise__": "GenerationError", "msg": "x"}]})
            orig = camp._ollama_model_digest
            try:
                camp._ollama_model_digest = lambda model, url: None  # unavailable
                with self.assertRaises(RuntimeError):
                    camp.run_campaign(_one_scenario_campaign(), tmp, jsonl,
                                      caller_factory=factory, online=True, log=lambda *a: None)
                self.assertFalse(jsonl.exists())  # nothing recorded
            finally:
                camp._ollama_model_digest = orig

    def test_torn_last_line_does_not_lose_records(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "raw.jsonl"
            p.write_text(json.dumps({"run_id": "a#rep1", "success": True}) + "\n"
                         + '{"run_id": "b#rep1", "succ', encoding="utf-8")  # torn last line
            recs = camp._read_records(p)
            self.assertEqual([r["run_id"] for r in recs], ["a#rep1"])  # valid record kept

    def test_repair_drops_torn_tail_and_ends_clean(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "raw.jsonl"
            p.write_text(json.dumps({"run_id": "a#rep1"}) + "\n"
                         + json.dumps({"run_id": "b#rep1"}) + "\n"
                         + '{"run_id": "c#re', encoding="utf-8")  # torn fragment, no newline
            kept = camp.repair_jsonl(p)
            self.assertEqual(kept, 2)
            text = p.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))          # clean to append to
            self.assertNotIn("c#re", text)                # torn fragment gone
            self.assertEqual([r["run_id"] for r in camp._read_records(p)], ["a#rep1", "b#rep1"])

    def test_repair_refuses_middle_corruption_and_leaves_file_untouched(self):
        # A corrupt line BEFORE the end is real data loss, not a crash artefact:
        # repair_jsonl must raise and leave the dataset byte-for-byte unchanged
        # (never silently drop a record from the middle).
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "raw.jsonl"
            original = (json.dumps({"run_id": "a#rep1"}) + "\n"
                        + '{"run_id": "b#re\n'                       # corrupt MIDDLE line
                        + json.dumps({"run_id": "c#rep1"}) + "\n")   # valid record after it
            p.write_text(original, encoding="utf-8")
            before = p.read_bytes()
            with self.assertRaises(ValueError):
                camp.repair_jsonl(p)
            self.assertEqual(p.read_bytes(), before)  # not modified at all

    def test_repair_refuses_complete_but_invalid_final_line(self):
        # An unparseable FINAL line that IS newline-terminated is a complete, corrupt
        # record — a crash could not have written that terminating newline — so it is
        # real corruption, not a torn fragment: raise and leave the file untouched.
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "raw.jsonl"
            original = (json.dumps({"run_id": "a#rep1"}) + "\n"
                        + '{"run_id": "b#re\n')          # invalid BUT newline-terminated
            p.write_text(original, encoding="utf-8")
            before = p.read_bytes()
            with self.assertRaises(ValueError):
                camp.repair_jsonl(p)
            self.assertEqual(p.read_bytes(), before)  # not modified at all

    def test_repair_leaves_valid_records_unchanged(self):
        # A clean, complete dataset must be returned intact: every valid record
        # kept and the bytes untouched (no spurious rewrite).
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "raw.jsonl"
            content = (json.dumps({"run_id": "a#rep1", "success": True}) + "\n"
                       + json.dumps({"run_id": "b#rep1", "success": False}) + "\n")
            p.write_text(content, encoding="utf-8")
            before = p.read_bytes()
            kept = camp.repair_jsonl(p)
            self.assertEqual(kept, 2)
            self.assertEqual(p.read_bytes(), before)  # valid records byte-for-byte intact
            self.assertEqual([r["run_id"] for r in camp._read_records(p)], ["a#rep1", "b#rep1"])

    def test_run_campaign_refuses_to_resume_middle_corrupted_dataset(self):
        # run_campaign repairs before resuming; a mid-dataset corruption must abort
        # the whole run (nothing generated, dataset left untouched) rather than
        # resume on top of silently dropped records.
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "raw.jsonl"
            original = ('{"run_id": "elf-easy#rep1", "corru\n'                     # corrupt middle
                        + json.dumps({"run_id": "elf-easy#rep2", "success": True}) + "\n")
            jsonl.write_text(original, encoding="utf-8")
            before = jsonl.read_bytes()
            factory = camp.replay_caller_factory({"*": [{"__raise__": "GenerationError", "msg": "x"}]})
            with self.assertRaises(ValueError):
                camp.run_campaign(_one_scenario_campaign(reps=2), tmp, jsonl,
                                  caller_factory=factory, log=lambda *a: None)
            self.assertEqual(jsonl.read_bytes(), before)  # untouched, no runs appended

    def test_crash_safe_resume_end_to_end(self):
        # A valid record + a truncated final line, then resume: both complete records
        # readable afterwards, the missing run is executed once, no duplicate run_id.
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "raw.jsonl"
            jsonl.write_text(json.dumps({"run_id": "elf-easy#rep1", "success": True}) + "\n"
                             + '{"run_id": "elf-easy#rep2", "succ', encoding="utf-8")  # torn tail
            factory = camp.replay_caller_factory({"*": [{"__raise__": "GenerationError", "msg": "x"}]})
            recs = camp.run_campaign(_one_scenario_campaign(reps=2), tmp, jsonl,
                                     caller_factory=factory, log=lambda *a: None)
            self.assertEqual([r["run_id"] for r in recs], ["elf-easy#rep2"])  # only the missing run
            ids = [r["run_id"] for r in camp._read_records(jsonl)]
            self.assertEqual(ids, ["elf-easy#rep1", "elf-easy#rep2"])  # both complete, readable
            self.assertEqual(len(ids), len(set(ids)))                  # no duplicates

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
