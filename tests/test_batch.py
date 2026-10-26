"""Offline tests for the Batch Campaign Runner.

The campaign-mechanics tests mock the per-run generation (`batch._generate`) and the
digest lookup, so resume / no-duplication / fingerprint / infra / incomplete / report /
artifact-isolation are exercised with NO Docker and NO Ollama. One Docker-gated class
drives the REAL generator through the batch with a replayed LLM.
"""
import contextlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from slothctf import batch
from slothctf import generator as gen
from slothctf import agentic as ag

BASE_CFG = {
    "model": "qwen2.5-coder:32b", "ollama_url": "http://tunnel-1",
    "matrix": {"genres": ["recover"], "difficulties": ["easy"], "targets": ["elf"], "workflows": ["agentic"]},
    "repeats": 3, "dest": "camp_test", "prompt": "invent a recover challenge",
    "inference": {"temperature": 0.7}, "budget": {"max_llm_calls": 24, "max_repairs_per_stage": 4},
}
FAKE_DIGEST = "deadbeef" * 8


@contextlib.contextmanager
def _cwd(tmp):
    prev = os.getcwd()
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(prev)


def _write_cfg(tmp, cfg=None):
    p = Path(tmp) / "campaign.json"
    p.write_text(json.dumps(cfg or BASE_CFG), encoding="utf-8")
    return str(p)


def make_mock(outcome_fn):
    """Return a fake _generate that writes a real ProjectState + transcript per run and
    returns a GenerationReport. `outcome_fn(spec, attempt_count)` -> outcome string."""
    counts = {}

    def mock(spec, name, config, caller, url=None):
        counts[spec["id"]] = counts.get(spec["id"], 0) + 1
        oc = outcome_fn(spec, counts[spec["id"]])
        if oc == "raise":
            raise ConnectionError("tunnel down")
        d = Path("generated") / name
        d.mkdir(parents=True, exist_ok=True)
        st = ag.ProjectState(name=name, target=spec["target"], difficulty=spec["difficulty"],
                             flag=f"sloth{{SECRET_{spec['id']}}}", path=str(d / "agentic_state.json"),
                             design="position-dependent XOR with addition")
        if oc in ("valid", "model-failure"):
            st.set_code("encoder", "enc"); st.mark_verified("encoder")
        if oc == "valid":
            st.set_code("challenge", "chal"); st.mark_verified("challenge")
            st.set_code("solver", "sol"); st.mark_verified("solver")
            (d / "README.md").write_text("# player\nRecover the flag sloth{...} from the binary.", encoding="utf-8")
            (d / "bin").mkdir(exist_ok=True)
            (d / "bin" / "challenge").write_bytes(b"\x7fELF fake player binary")
        st.budget_calls = 9 if oc == "valid" else 7
        st.budget_tokens = 10000
        st.transcript = [{"type": "llm", "tokens": 500}, {"type": "llm", "tokens": 400},
                         {"type": "revision", "applied": True, "component": "challenge"},
                         {"type": "repair-no-progress", "component": "solver"}]
        st.save()
        (d / "transcript.jsonl").write_text("\n".join(json.dumps(e) for e in st.transcript) + "\n", encoding="utf-8")
        rep = gen.GenerationReport(name=name, model=config["model"], directory=str(d), success=(oc == "valid"))
        if oc == "valid":
            rep.attempts = [gen.AttemptResult(0, True, "passed")]
            rep.telemetry = {"final_stage": "passed"}
        elif oc == "infra":
            rep.attempts = [gen.AttemptResult(0, False, "infra")]
            rep.telemetry = {"final_stage": "infra", "error": {"category": "infra"}}
        else:
            rep.attempts = [gen.AttemptResult(0, False, "challenge-compile")]
            rep.telemetry = {"final_stage": "max-repairs"}
        return rep

    mock.counts = counts
    return mock


class BatchMechanicsTests(unittest.TestCase):
    def setUp(self):
        self._orig = batch._generate

    def tearDown(self):
        batch._generate = self._orig

    def _digest(self, *_):
        return FAKE_DIGEST

    def test_full_campaign_report_and_confidential_isolation(self):
        with TemporaryDirectory() as tmp, _cwd(tmp):
            cfg = dict(BASE_CFG); cfg["repeats"] = 3
            batch._generate = make_mock(lambda spec, n: "valid" if spec["id"].endswith("_02") else "model-failure")
            camp, _ = batch.cmd_run(_write_cfg(tmp, cfg), caller_factory=lambda *a: None, digest_fn=self._digest)
            recs = batch._read_jsonl(camp / "campaign.jsonl")
            results = [r for r in recs if r.get("event") == "attempt_result"]
            self.assertEqual(len(results), 3)                         # all 3 completed
            self.assertEqual(sum(1 for r in results if r["success"]), 1)
            self.assertTrue((camp / "report.md").exists() and (camp / "report.csv").exists())
            self.assertTrue((camp / "_valid_playtest").exists())      # valid artifacts copied
            # confidential isolation: the flag lives in the state, NOT in campaign.jsonl/report
            blob = (camp / "campaign.jsonl").read_text() + (camp / "report.md").read_text() + (camp / "report.csv").read_text()
            self.assertNotIn("sloth{SECRET_", blob)
            self.assertIn("SECRET_recover_easy_elf_agentic_02", (camp / "recover_easy_elf_agentic_02" / "attempt_01" / "agentic_state.json").read_text())

    def test_infra_stops_then_resume_no_duplication(self):
        with TemporaryDirectory() as tmp, _cwd(tmp):
            # run_01 valid; run_02 infra on 1st attempt, model-failure on 2nd; run_03 model-failure
            def oc(spec, n):
                if spec["id"].endswith("_01"): return "valid"
                if spec["id"].endswith("_02"): return "infra" if n == 1 else "model-failure"
                return "model-failure"
            batch._generate = make_mock(oc)
            cfg = _write_cfg(tmp)
            camp, _ = batch.cmd_run(cfg, caller_factory=lambda *a: None, digest_fn=self._digest)
            recs = batch._read_jsonl(camp / "campaign.jsonl")
            # stopped at run_02 infra; run_03 not attempted
            self.assertTrue(any(r.get("event") == "STOP" for r in recs))
            res = [r for r in recs if r.get("event") == "attempt_result"]
            self.assertEqual({r["id"] for r in res}, {"recover_easy_elf_agentic_01", "recover_easy_elf_agentic_02"})
            self.assertEqual([r["outcome_class"] for r in res if r["id"].endswith("_02")], ["infra"])
            # resume: run_01 NOT redone; run_02 gets attempt_02 (model-failure); run_03 runs
            batch.cmd_resume(camp, caller_factory=lambda *a: None, digest_fn=self._digest)
            recs2 = batch._read_jsonl(camp / "campaign.jsonl")
            res2 = [r for r in recs2 if r.get("event") == "attempt_result"]
            r01 = [r for r in res2 if r["id"].endswith("_01")]
            r02 = [r for r in res2 if r["id"].endswith("_02")]
            self.assertEqual(len(r01), 1, "completed run must not be repeated")
            self.assertEqual([r["outcome_class"] for r in r02], ["infra", "model-failure"])  # both attempts kept
            self.assertTrue(any(r["id"].endswith("_03") for r in res2))
            # completed model generations == 3 (01 valid, 02 attempt2 fail, 03 fail); infra excluded
            completed = [r for r in res2 if r["outcome_class"] in ("valid", "model-failure")]
            self.assertEqual(len(completed), 3)

    def test_incomplete_killed_attempt_recovered_on_resume(self):
        with TemporaryDirectory() as tmp, _cwd(tmp):
            batch._generate = make_mock(lambda spec, n: "model-failure")
            # start a campaign and let it finish, then simulate a KILLED run by hand:
            cfg = dict(BASE_CFG); cfg["repeats"] = 1
            camp, _ = batch.cmd_run(_write_cfg(tmp, cfg), caller_factory=lambda *a: None, digest_fn=self._digest)
            # fabricate a killed future attempt: an attempt_start with NO result for a fresh id
            # (simulate by appending a dangling start for run_01 attempt_02 as if the process died)
            batch._append(camp / "campaign.jsonl",
                          {"event": "attempt_start", "id": "recover_easy_elf_agentic_01", "attempt": 2,
                           "name": "camp_test/recover_easy_elf_agentic_01/attempt_02", "ts": 1.0})
            # run_01 already has a model-failure attempt_01, so it is "done"; the dangling start is an orphan
            batch.cmd_resume(camp, caller_factory=lambda *a: None, digest_fn=self._digest)
            recs = batch._read_jsonl(camp / "campaign.jsonl")
            self.assertTrue(any(r.get("event") == "attempt_incomplete" and r.get("attempt") == 2 for r in recs),
                            "dangling start must be classified as incomplete")
            # it is NOT counted as a model result
            self.assertFalse(any(r.get("event") == "attempt_result" and r.get("attempt") == 2 for r in recs))

    def test_fingerprint_refuses_digest_change(self):
        with TemporaryDirectory() as tmp, _cwd(tmp):
            batch._generate = make_mock(lambda spec, n: "model-failure")
            camp, _ = batch.cmd_run(_write_cfg(tmp), caller_factory=lambda *a: None, digest_fn=self._digest)
            with self.assertRaises(batch.BatchError):
                batch.cmd_resume(camp, caller_factory=lambda *a: None, digest_fn=lambda *_: "f00dface" * 8)

    def test_url_change_allowed_on_resume(self):
        with TemporaryDirectory() as tmp, _cwd(tmp):
            # run_02 infra so there's something to resume; same digest regardless of url
            batch._generate = make_mock(lambda spec, n: ("infra" if (spec["id"].endswith("_02") and n == 1) else "model-failure"))
            camp, _ = batch.cmd_run(_write_cfg(tmp), caller_factory=lambda *a: None, digest_fn=self._digest)
            batch.cmd_resume(camp, ollama_url="http://tunnel-2-NEW", caller_factory=lambda *a: None, digest_fn=self._digest)
            recs = batch._read_jsonl(camp / "campaign.jsonl")
            self.assertTrue(any(r.get("event") == "endpoint_change" and r.get("to") == "http://tunnel-2-NEW" for r in recs))
            self.assertEqual(json.loads((camp / "manifest.json").read_text())["url_current"], "http://tunnel-2-NEW")

    def test_truncated_tail_recovered_with_diagnostic_and_resume(self):
        # Simulate a crash in the MIDDLE of a write: the final line has no newline. The
        # truncated tail is recoverable; the full damaged bytes are preserved, the tail is
        # dropped, no valid event is lost, and resume completes with no concatenation.
        with TemporaryDirectory() as tmp, _cwd(tmp):
            # run_02 infra so resume has work left to do
            batch._generate = make_mock(lambda spec, n: ("infra" if (spec["id"].endswith("_02") and n == 1) else "model-failure"))
            camp, _ = batch.cmd_run(_write_cfg(tmp), caller_factory=lambda *a: None, digest_fn=self._digest)
            jl = camp / "campaign.jsonl"
            before = batch._read_jsonl(jl)                     # clean, parses fine
            with jl.open("a", encoding="utf-8") as fh:
                fh.write('{"event":"attempt_start","id":"recover_easy_elf_agentic_03"')  # killed mid-write
            with self.assertRaises(batch.BatchError):
                batch._read_jsonl(jl)                          # strict read rejects the damaged file
            batch.cmd_resume(camp, caller_factory=lambda *a: None, digest_fn=self._digest)
            # damaged bytes preserved in a diagnostic artifact
            self.assertTrue(any(p.name.startswith("campaign.jsonl.corrupt-") for p in camp.iterdir()))
            after = batch._read_jsonl(jl)                      # now clean (no concatenation)
            # every previously-complete event survived; the truncated tail is gone
            self.assertTrue(len(after) >= len(before))
            self.assertFalse(any(r.get("id") == "recover_easy_elf_agentic_03" and r.get("event") == "attempt_start"
                                 and set(r.keys()) == {"event", "id"} for r in after))
            self.assertTrue((camp / "report.md").exists())

    def test_valid_final_line_without_newline_is_kept_not_dropped(self):
        # A complete, valid attempt_result that merely lost its trailing newline must be
        # KEPT (not treated as truncated) — otherwise the completed run would be repeated.
        with TemporaryDirectory() as tmp, _cwd(tmp):
            batch._generate = make_mock(lambda spec, n: "model-failure")
            cfg = dict(BASE_CFG); cfg["repeats"] = 1
            camp, _ = batch.cmd_run(_write_cfg(tmp, cfg), caller_factory=lambda *a: None, digest_fn=self._digest)
            jl = camp / "campaign.jsonl"
            recs = batch._read_jsonl(jl)
            keep = [r for r in recs if r.get("event") != "campaign_done"]   # last kept record is the attempt_result
            self.assertEqual(keep[-1].get("event"), "attempt_result")
            jl.write_text("\n".join(json.dumps(r) for r in keep), encoding="utf-8")   # NO trailing newline
            self.assertFalse(jl.read_text().endswith("\n"))
            batch.cmd_resume(camp, caller_factory=lambda *a: None, digest_fn=self._digest)
            after = batch._read_jsonl(jl)
            ar = [r for r in after if r.get("event") == "attempt_result" and str(r.get("id", "")).endswith("_01")]
            self.assertEqual(len(ar), 1, "completed run was repeated or its result was lost")
            self.assertFalse(any(p.name.startswith("campaign.jsonl.corrupt-") for p in camp.iterdir()),
                             "a valid final line must NOT be treated as corrupt")

    def test_midrecord_corruption_refused_with_diagnostic(self):
        # A malformed COMPLETE (newline-terminated) record in the middle is NOT auto-dropped:
        # preserved + BatchError, so no valid event vanishes silently.
        with TemporaryDirectory() as tmp, _cwd(tmp):
            batch._generate = make_mock(lambda spec, n: ("infra" if (spec["id"].endswith("_02") and n == 1) else "model-failure"))
            camp, _ = batch.cmd_run(_write_cfg(tmp), caller_factory=lambda *a: None, digest_fn=self._digest)
            jl = camp / "campaign.jsonl"
            with jl.open("a", encoding="utf-8") as fh:
                fh.write("THIS IS NOT JSON\n")                 # complete, but corrupt, record
                fh.write('{"event":"note","ok":true}\n')       # a valid record after it
            with self.assertRaises(batch.BatchError):
                batch.cmd_resume(camp, caller_factory=lambda *a: None, digest_fn=self._digest)
            self.assertTrue(any(p.name.startswith("campaign.jsonl.corrupt-") for p in camp.iterdir()))

    def test_inference_applied_in_ollama_payload(self):
        # the REAL payload sent to Ollama carries the configured temperature + num_ctx
        import urllib.request as urllib_request

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __iter__(self):
                yield b'{"response":"{}","done":false}\n'
                yield b'{"response":"","done":true,"prompt_eval_count":1,"eval_count":1}\n'
        captured = {}
        orig = urllib_request.urlopen
        try:
            urllib_request.urlopen = lambda req, timeout=None: (captured.update(body=req.data), _Resp())[1]
            ag.ollama_transport("http://x", temperature=0.33, num_ctx=2048)("prompt", "model")
        finally:
            urllib_request.urlopen = orig
        sent = json.loads(captured["body"].decode())
        self.assertEqual(sent["options"]["temperature"], 0.33)
        self.assertEqual(sent["options"]["num_ctx"], 2048)

    def test_caller_factory_wires_inference(self):
        rec = {}
        orig = ag.ollama_transport
        try:
            ag.ollama_transport = lambda url, temperature=0.7, num_ctx=4096: rec.update(t=temperature, c=num_ctx) or "C"
            c = batch._default_caller_factory("agentic", "http://e", {"temperature": 0.2, "num_ctx": 1024})
        finally:
            ag.ollama_transport = orig
        self.assertEqual((rec["t"], rec["c"], c), (0.2, 1024, "C"))
        self.assertIsNone(batch._default_caller_factory("monolithic", "http://e", {}))   # monolithic -> own transport

    def test_monolithic_passes_configured_url(self):
        captured = {}
        orig_be, orig_gen = batch._resolve_isolated_backend, gen.generate_challenge
        try:
            batch._resolve_isolated_backend = lambda target, iso: object()

            def fake_generate(name, **kw):
                captured.update(kw)
                rep = gen.GenerationReport(name=name, model=kw.get("model", "m"), directory=".", success=False)
                rep.attempts = []; rep.telemetry = {}
                return rep
            gen.generate_challenge = fake_generate
            spec = {"id": "reverse_easy_elf_monolithic_01", "genre": "reverse", "difficulty": "easy",
                    "target": "elf", "workflow": "monolithic"}
            batch._generate(spec, "n", {"model": "m", "prompt": "p", "budget": {}}, None, "http://CONFIGURED")
        finally:
            batch._resolve_isolated_backend, gen.generate_challenge = orig_be, orig_gen
        self.assertEqual(captured.get("ollama_url"), "http://CONFIGURED")   # was defaulting before

    def test_monolithic_rejects_inference_override(self):
        with TemporaryDirectory() as tmp, _cwd(tmp):
            cfg = dict(BASE_CFG)
            cfg["matrix"] = {"genres": ["reverse"], "difficulties": ["easy"], "targets": ["elf"], "workflows": ["monolithic"]}
            cfg["inference"] = {"temperature": 0.9}
            with self.assertRaises(batch.BatchError):
                batch.load_config(_write_cfg(tmp, cfg))

    def test_error_classification_infra_runner_interrupt(self):
        def one(exc_type):
            with TemporaryDirectory() as tmp, _cwd(tmp):
                batch._generate = (lambda spec, name, config, caller, url=None: (_ for _ in ()).throw(exc_type("boom")))
                camp, status = batch.cmd_run(_write_cfg(tmp), caller_factory=lambda *a: None, digest_fn=self._digest)
                res = [r for r in batch._read_jsonl(camp / "campaign.jsonl") if r.get("event") == "attempt_result"]
                return status, res[-1]["outcome_class"]
        self.assertEqual(one(ConnectionError), ("stopped-infra", "infra"))      # OSError subclass -> infra
        self.assertEqual(one(RuntimeError), ("stopped-error", "runner-error"))  # unexpected -> runner-error
        with TemporaryDirectory() as tmp, _cwd(tmp):
            batch._generate = (lambda spec, name, config, caller, url=None: (_ for _ in ()).throw(KeyboardInterrupt()))
            with self.assertRaises(KeyboardInterrupt):
                batch.cmd_run(_write_cfg(tmp), caller_factory=lambda *a: None, digest_fn=self._digest)
            res = [r for r in batch._read_jsonl(Path("generated/camp_test") / "campaign.jsonl") if r.get("event") == "attempt_result"]
            self.assertEqual(res[-1]["outcome_class"], "interrupted")           # preserved, not a model result

    def test_exit_codes_run(self):
        orig = batch._live_digest
        try:
            batch._live_digest = lambda url, model: FAKE_DIGEST
            with TemporaryDirectory() as tmp, _cwd(tmp):
                batch._generate = make_mock(lambda spec, n: "model-failure")
                self.assertEqual(batch.main(["run", "--config", _write_cfg(tmp)]), 0)   # done (with failures) -> 0
            with TemporaryDirectory() as tmp, _cwd(tmp):
                batch._generate = (lambda spec, name, config, caller, url=None: (_ for _ in ()).throw(ConnectionError("x")))
                self.assertEqual(batch.main(["run", "--config", _write_cfg(tmp)]), 3)   # infra -> 3
            with TemporaryDirectory() as tmp, _cwd(tmp):
                batch._generate = (lambda spec, name, config, caller, url=None: (_ for _ in ()).throw(RuntimeError("x")))
                self.assertEqual(batch.main(["run", "--config", _write_cfg(tmp)]), 4)   # runner-error -> 4
            with TemporaryDirectory() as tmp, _cwd(tmp):
                bad = dict(BASE_CFG); del bad["model"]
                self.assertEqual(batch.main(["run", "--config", _write_cfg(tmp, bad)]), 2)   # config error -> 2
            with TemporaryDirectory() as tmp, _cwd(tmp):
                self.assertEqual(batch.main(["report", "--dir", "generated/nope"]), 2)       # no campaign -> 2
            with TemporaryDirectory() as tmp, _cwd(tmp):
                batch._live_digest = lambda url, model: (_ for _ in ()).throw(OSError("refused"))
                self.assertEqual(batch.main(["run", "--config", _write_cfg(tmp)]), 2)         # unreachable endpoint -> 2
        finally:
            batch._live_digest = orig

    def test_no_native_fallback(self):
        with self.assertRaises(batch.BatchError):
            batch._resolve_isolated_backend("elf", "native")
        orig = gen.resolve_backend
        try:
            class _Native:  # pretend resolve returned a non-docker backend
                name = "native"
            gen.resolve_backend = lambda *a, **k: _Native()
            with self.assertRaises(batch.BatchError):
                batch._resolve_isolated_backend("elf", "isolated")
        finally:
            gen.resolve_backend = orig

    def test_run_refuses_overwrite_existing_campaign(self):
        with TemporaryDirectory() as tmp, _cwd(tmp):
            batch._generate = make_mock(lambda spec, n: "model-failure")
            cfg = _write_cfg(tmp)
            batch.cmd_run(cfg, caller_factory=lambda *a: None, digest_fn=self._digest)
            with self.assertRaises(batch.BatchError):
                batch.cmd_run(cfg, caller_factory=lambda *a: None, digest_fn=self._digest)

    def test_tokens_unavailable_not_zeroed(self):
        # a run whose transcript has no token info must report tokens=None, not 0
        with TemporaryDirectory() as tmp, _cwd(tmp):
            def mock(spec, name, config, caller, url=None):
                d = Path("generated") / name; d.mkdir(parents=True, exist_ok=True)
                st = ag.ProjectState(name=name, target="elf", difficulty="easy", flag="sloth{x}",
                                     path=str(d / "agentic_state.json"), design="xor")
                st.set_code("encoder", "e"); st.mark_verified("encoder")
                st.budget_calls = 3; st.budget_tokens = 0
                st.transcript = [{"type": "llm", "tokens": None}, {"type": "llm", "tokens": None}]
                st.save()
                (d / "transcript.jsonl").write_text("\n".join(json.dumps(e) for e in st.transcript) + "\n", encoding="utf-8")
                rep = gen.GenerationReport(name=name, model="m", directory=str(d), success=False)
                rep.attempts = [gen.AttemptResult(0, False, "challenge-compile")]; rep.telemetry = {}
                return rep
            batch._generate = mock
            cfg = dict(BASE_CFG); cfg["repeats"] = 1
            camp, _ = batch.cmd_run(_write_cfg(tmp, cfg), caller_factory=lambda *a: None, digest_fn=self._digest)
            res = [r for r in batch._read_jsonl(camp / "campaign.jsonl") if r.get("event") == "attempt_result"][0]
            self.assertIsNone(res["tokens"], "unknown tokens must be null, never 0")
            self.assertEqual(res["llm_calls"], 3)


# =========================================================================== #
# Docker-gated end-to-end: the REAL generator driven through the batch
# =========================================================================== #
from slothctf.execution import docker_available, image_available
try:
    from tests.test_agentic import Replay, ENC_GOOD, CHAL_GOOD, SOL_GOOD
except Exception:  # pragma: no cover
    Replay = None
_READY = docker_available() and image_available() and Replay is not None


@unittest.skipUnless(_READY, "needs Docker + the slothctf-sandbox image")
class BatchDockerElfTests(unittest.TestCase):
    def test_real_agentic_elf_campaign_one_valid(self):
        items = [{"design": "d"}, {"encoder_code": ENC_GOOD},
                 {"challenge_code": CHAL_GOOD, "success_marker": "Correct"}, {"solver_code": SOL_GOOD}]
        with TemporaryDirectory() as tmp, _cwd(tmp):
            cfg = dict(BASE_CFG); cfg["repeats"] = 1; cfg["dest"] = "camp_docker"
            cf = lambda wf, url, temp: Replay(list(items))   # fresh replay for the single run
            camp, _ = batch.cmd_run(_write_cfg(tmp, cfg), caller_factory=cf, digest_fn=lambda *_: FAKE_DIGEST)
            res = [r for r in batch._read_jsonl(camp / "campaign.jsonl") if r.get("event") == "attempt_result"]
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]["outcome_class"], "valid", res[0].get("final_stage"))
            self.assertTrue((camp / "_valid_playtest" / res[0]["id"] / "challenge").exists())
            # flag must not leak into the campaign log/report
            blob = (camp / "campaign.jsonl").read_text() + (camp / "report.md").read_text()
            self.assertNotIn("sloth{", blob)


if __name__ == "__main__":
    unittest.main()
