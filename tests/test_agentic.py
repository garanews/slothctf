"""Offline acceptance tests for the agentic (test-driven) recover workflow.

LLM calls are REPLAYED from pre-recorded responses (no Ollama). The Docker-gated
class drives real ELF/PE compile/run/solver through the reused primitives, with the
solver always isolated. The Docker-free class tests the state machine: dependency
invalidation, crash/resume bookkeeping, persistence, and budget limits.
"""
import contextlib
import json
import os
import sys
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

from slothctf import agentic as ag
from slothctf.execution import NativeBackend, docker_available, image_available, win_image_available

_READY = docker_available() and image_available()
_SKIP = "needs Docker + the slothctf-sandbox image"
_WIN_READY = docker_available() and win_image_available()
_WIN_SKIP = "needs Docker + the slothctf-sandbox-win image"

# --- proven-good components (XOR 0x5a; static const; marker 'Correct') ---------
ENC_GOOD = ("#include <stdio.h>\nint main(int c,char**v){for(char*p=v[1];*p;++p)"
            "printf(\"0x%02x,\",(unsigned char)(*p^0x5a));return 0;}\n")
CHAL_GOOD = ("#include <stdio.h>\n#include <string.h>\n"
             "static const unsigned char data[]={ /*__ENCODED_ARRAY__*/ };\n"
             "int main(void){char b[256];if(!fgets(b,sizeof b,stdin)){printf(\"no\\n\");return 0;}"
             "size_t n=strlen(b);if(n&&b[n-1]=='\\n')b[--n]=0;"
             "if(n!=sizeof data){printf(\"Denied\\n\");return 0;}"
             "for(size_t i=0;i<n;i++)if(((unsigned char)b[i]^0x5a)!=data[i]){printf(\"Denied\\n\");return 0;}"
             "printf(\"Correct\\n\");return 0;}\n")
SOL_GOOD = ("import sys, re\n"
            "b=open(sys.argv[1],'rb').read()\n"
            "dec=bytes(x^0x5a for x in b)\n"
            "m=re.search(rb'sloth\\{[^}]{1,64}\\}',dec)\n"
            "print(m.group(0).decode() if m else '')\n")

# deliberately broken variants
ENC_BROKEN = "#include <stdio.h>\nint main(){ this is not valid c }\n"
CHAL_BROKEN = ("#include <stdio.h>\nstatic const unsigned char d[]={ /*__ENCODED_ARRAY__*/ };\n"
               "int main(){ not valid c at all ; }\n")
SOL_WRONG = "print('sloth{definitely_not_it}')\n"          # valid python, wrong output -> mismatch

# XOR 0x5a challenge that hardcodes a wrong flag length (32) -> rejects the real flag,
# reproducing the first A100 run's defect; marker matches CHAL_FIXED below.
CHAL_BADLEN = ("#include <stdio.h>\n#include <string.h>\n"
               "static const unsigned char data[]={ /*__ENCODED_ARRAY__*/ };\n"
               "int main(void){char b[256];if(!fgets(b,sizeof b,stdin)){printf(\"no\\n\");return 0;}"
               "size_t n=strlen(b);if(n&&b[n-1]=='\\n')b[--n]=0;"
               "if(n!=32){printf(\"WRONG_INPUT\\n\");return 0;}"
               "for(size_t i=0;i<n;i++)if(((unsigned char)b[i]^0x5a)!=data[i]){printf(\"WRONG_INPUT\\n\");return 0;}"
               "printf(\"FLAG_RECOVERED\\n\");return 0;}\n")
# the same challenge with the length derived from the embedded array (the correct fix).
CHAL_FIXED = ("#include <stdio.h>\n#include <string.h>\n"
              "static const unsigned char data[]={ /*__ENCODED_ARRAY__*/ };\n"
              "int main(void){char b[256];if(!fgets(b,sizeof b,stdin)){printf(\"no\\n\");return 0;}"
              "size_t n=strlen(b);if(n&&b[n-1]=='\\n')b[--n]=0;"
              "if(n!=sizeof data){printf(\"WRONG_INPUT\\n\");return 0;}"
              "for(size_t i=0;i<n;i++)if(((unsigned char)b[i]^0x5a)!=data[i]){printf(\"WRONG_INPUT\\n\");return 0;}"
              "printf(\"FLAG_RECOVERED\\n\");return 0;}\n")


def _mk(design="d", **payloads):
    return dict(payloads)


class Replay:
    """FIFO of canned LLM responses. An item may be a dict (returned), a BaseException
    subclass (raised, to simulate a crash), or a callable(prompt, model)."""
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def __call__(self, prompt, model):
        self.calls.append(prompt)
        if not self.items:
            raise AssertionError("replay exhausted — orchestrator asked more than scripted")
        item = self.items.pop(0)
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item()
        if callable(item):
            return item(prompt, model)
        return item


# =========================================================================== #
# Docker-free: state machine, invalidation, resume bookkeeping, budget
# =========================================================================== #
class StateMachineTests(unittest.TestCase):
    def _verified_state(self):
        st = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}")
        st.set_code("encoder", ENC_GOOD); st.mark_verified("encoder")
        st.set_code("challenge", CHAL_GOOD); st.mark_verified("challenge")
        st.set_code("solver", SOL_GOOD); st.mark_verified("solver")
        return st

    def test_encoder_change_invalidates_challenge_and_solver(self):
        st = self._verified_state()
        self.assertTrue(st.is_verified("encoder") and st.is_verified("challenge") and st.is_verified("solver"))
        st.set_code("encoder", ENC_GOOD + "\n// tweak\n")   # any change to the encoder
        self.assertFalse(st.is_verified("encoder"))
        self.assertFalse(st.is_verified("challenge"), "challenge must invalidate when encoder changes")
        self.assertFalse(st.is_verified("solver"), "solver must invalidate when encoder changes")

    def test_challenge_change_invalidates_only_solver(self):
        st = self._verified_state()
        st.set_code("challenge", CHAL_GOOD + "\n")
        self.assertTrue(st.is_verified("encoder"))
        self.assertFalse(st.is_verified("challenge"))
        self.assertFalse(st.is_verified("solver"))

    def test_set_code_keeps_last_good(self):
        st = self._verified_state()
        st.set_code("solver", SOL_WRONG)
        self.assertEqual(st.solver.last_good, SOL_GOOD)   # previous good version retained

    def test_persistence_roundtrip_and_verified_survives(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "agentic_state.json"
            st = self._verified_state()
            st.path = str(p)
            st.design = "a design"
            st.save()
            st2 = ag.ProjectState.load(p)
            self.assertEqual(st2.design, "a design")
            self.assertEqual(st2.encoder.code, ENC_GOOD)
            self.assertTrue(st2.is_verified("encoder") and st2.is_verified("solver"))
            # a tampered upstream after load still invalidates downstream
            st2.set_code("encoder", "changed")
            self.assertFalse(st2.is_verified("solver"))

    def test_budget_exhaustion_terminates(self):
        # max_llm_calls=1: design is call 1; the encoder generate would be call 2 and
        # is REFUSED before it is issued (budget enforced pre-call), so exactly one
        # invocation happens. No compile is reached, so no Docker and no infinite loop.
        with TemporaryDirectory() as tmp:
            prev = __import__("os").getcwd()
            __import__("os").chdir(tmp)
            try:
                caller = Replay([{"design": "d"}, {"encoder_code": ENC_GOOD}, {"encoder_code": ENC_GOOD}])
                rep = ag.generate_recover_agentic(
                    "bud", prompt="p", target="elf", backend=NativeBackend(),
                    caller=caller, budget=ag.Budget(max_llm_calls=1), resume=False)
            finally:
                __import__("os").chdir(prev)
        self.assertFalse(rep.success)
        self.assertEqual(rep.attempts[-1].stage, "budget-exhausted")
        self.assertEqual(len(caller.calls), 1)   # design only, encoder refused pre-call

    def test_budget_never_exceeds_limit(self):
        # Regression (unit) for the pre-call budget gate: with limit N, EXACTLY N
        # invocations may be issued and the (N+1)-th is refused before it reaches the
        # transport. Driving _ask directly keeps this independent of any backend.
        for n in (0, 1, 2, 3, 5):
            with self.subTest(n=n):
                calls = []

                def caller(prompt, model):
                    calls.append(prompt)
                    return {"ok": True}

                st = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}")
                budget = ag.Budget(max_llm_calls=n)
                rep = ag.g.GenerationReport(name="x", model="m", directory=".", success=False)
                issued = 0
                with self.assertRaises(ag.BudgetExhausted):
                    for _ in range(n + 5):          # attempt more calls than the cap
                        ag._ask(caller, "m", budget, st, rep, "s", "generate", "p")
                        issued += 1
                self.assertEqual(issued, n)          # exactly N succeeded before refusal
                self.assertEqual(len(calls), n)      # exactly N reached the transport
                self.assertEqual(budget.llm_calls, n)

    def test_failed_calls_count_and_survive_resume(self):
        # Regression: an invocation that was issued but FAILED (here a connection
        # error) must count toward max_llm_calls and the count must survive resume,
        # so the real number of transport hits across runs never exceeds N.
        N = 2

        class CountingCaller:
            """1st call returns the design, every later call fails at the transport."""
            def __init__(self):
                self.n = 0

            def __call__(self, prompt, model):
                self.n += 1
                if self.n == 1:
                    return {"design": "d"}
                raise ConnectionError("tunnel down")

        caller = CountingCaller()
        with TemporaryDirectory() as tmp:
            prev = os.getcwd()
            os.chdir(tmp)
            try:
                r1 = ag.generate_recover_agentic(
                    "resfail", prompt="p", target="elf", backend=NativeBackend(),
                    caller=caller, budget=ag.Budget(max_llm_calls=N), resume=False)
                self.assertEqual(r1.attempts[-1].stage, "infra")   # design ok, encoder failed
                self.assertEqual(caller.n, N)                      # 1 ok + 1 failed == N

                # the persisted counter includes the failed invocation
                sp = Path(tmp) / "generated" / "resfail" / "agentic_state.json"
                self.assertEqual(ag.ProjectState.load(sp).budget_calls, N)

                # resume: the cap is already reached, so NO further call is issued
                before = caller.n
                r2 = ag.generate_recover_agentic(
                    "resfail", prompt="p", target="elf", backend=NativeBackend(),
                    caller=caller, budget=ag.Budget(max_llm_calls=N), resume=True)
                self.assertEqual(r2.attempts[-1].stage, "budget-exhausted")
                self.assertEqual(caller.n, before)                 # no new transport hits
            finally:
                os.chdir(prev)
        self.assertLessEqual(caller.n, N)                          # N never exceeded

    def test_max_repairs_terminates(self):
        # With a native backend the encoder gate can't compile, but the generate call
        # for the encoder already consumes a repair budget of 0 via a different path:
        # here we just assert the orchestrator stops (no infinite loop) and fails.
        with TemporaryDirectory() as tmp:
            prev = __import__("os").getcwd()
            __import__("os").chdir(tmp)
            try:
                # design, then many encoder generations; native compile fails each time
                items = [{"design": "d"}] + [{"encoder_code": ENC_BROKEN}] * 10
                caller = Replay(items)
                rep = ag.generate_recover_agentic(
                    "rep", prompt="p", target="elf", backend=NativeBackend(),
                    caller=caller, budget=ag.Budget(max_repairs_per_stage=1, max_llm_calls=20),
                    resume=False)
            finally:
                __import__("os").chdir(prev)
        self.assertFalse(rep.success)
        self.assertIn(rep.attempts[-1].stage, {"max-repairs", "budget-exhausted", "infra"})

    # -- Fix 2: persistent, integral, local transcript --------------------- #
    def test_transcript_is_integral_and_local(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "agentic_state.json"
            st = ag.ProjectState(name="x", target="elf", difficulty="easy",
                                 flag="sloth{secret}", path=str(p))
            budget = ag.Budget()
            rep = ag.g.GenerationReport(name="x", model="m", directory=tmp, success=False)
            # an LLM call is recorded VERBATIM (full prompt + response), not just sizes
            ag._record_call(st, budget, rep, "design", "generate",
                            "PROMPT-TEXT", {"design": "DESIGN-TEXT"}, 42, 0.1, "ok")
            # a code revision records the feedback that drove it AND the new source
            st.log({"type": "revision", "component": "encoder", "repairs": 1,
                    "reason": "encoder.c failed to compile", "code": ENC_GOOD})
            st.save()

            # persisted in-state and survives a save/load round-trip
            st2 = ag.ProjectState.load(p)
            llm = next(e for e in st2.transcript if e.get("type") == "llm")
            self.assertEqual(llm["prompt"], "PROMPT-TEXT")
            self.assertIn("DESIGN-TEXT", llm["response"])
            self.assertEqual(llm["tokens"], 42)
            rev = next(e for e in st2.transcript if e.get("type") == "revision")
            self.assertEqual(rev["code"], ENC_GOOD)
            self.assertIn("failed to compile", rev["reason"])

            # mirrored to a LOCAL append-only transcript.jsonl beside the state file
            jl = Path(tmp) / "transcript.jsonl"
            self.assertTrue(jl.exists())
            self.assertEqual(jl.parent, p.parent)       # organizer-local, not distributable
            lines = [json.loads(x) for x in jl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["prompt"], "PROMPT-TEXT")

    # -- Phase 2: Stage C can consult the verified encoder ----------------- #
    def test_stage_c_prompt_includes_verified_encoder(self):
        st = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}")
        st.design = "DESIGN-TEXT"
        st.set_code("encoder", ENC_GOOD); st.mark_verified("encoder")
        gen = ag._challenge_prompt(st)
        self.assertIn(ENC_GOOD.strip(), gen)                 # generate prompt carries the encoder
        rep = ag._repair_prompt("challenge", st, "recover-rejects-correct: got 'WRONG_INPUT'")
        self.assertIn(ENC_GOOD.strip(), rep)                 # repair prompt carries the encoder
        self.assertIn("DESIGN-TEXT", rep)                    # ...and the design
        self.assertIn("recover-rejects-correct", rep)        # ...and the actual error

    # -- Phase 3/4: no-progress detection, no wasted call, full telemetry --- #
    def test_no_progress_detected_no_superfluous_call_full_telemetry(self):
        # A backend whose compile always fails, so the encoder gate keeps failing; the
        # model resends identical code. We must detect no-progress, escalate, stop at the
        # repair budget WITHOUT an un-appliable extra call, and still report telemetry.
        class _R:
            ok = False
            stderr = "syntax error near 'this'"
        class _CompileFail:
            name = "fake"
            def compile_c(self, *a, **k):
                return _R()
        with TemporaryDirectory() as tmp:
            prev = os.getcwd()
            os.chdir(tmp)
            try:
                items = [{"design": "d"}] + [{"encoder_code": ENC_BROKEN}] * 8
                caller = Replay(items)
                rep = ag.generate_recover_agentic(
                    "np", prompt="p", target="elf", backend=_CompileFail(), caller=caller,
                    budget=ag.Budget(max_repairs_per_stage=2, max_llm_calls=20), resume=False)
            finally:
                os.chdir(prev)
        np = [a for a in rep.attempts if a.stage == "repair-no-progress"]
        self.assertEqual(len(np), 2, [a.stage for a in rep.attempts])     # both identical repairs flagged
        self.assertEqual(rep.telemetry.get("no_progress", {}).get("encoder"), 2)
        self.assertEqual(rep.attempts[-1].stage, "max-repairs")
        # design + encoder generate + exactly 2 repairs; the 3rd repair is NOT issued
        self.assertEqual(len(caller.calls), 4)
        # Phase 4: telemetry complete even on the max-repairs path
        self.assertEqual(rep.telemetry["budget"]["llm_calls"], 4)
        self.assertEqual(rep.telemetry["final_stage"], "max-repairs")
        self.assertIsNotNone(rep.telemetry["budget"]["tokens"])

    def test_small_change_is_progress_not_blocked(self):
        # A one-token change must count as progress (not a no-op repair).
        self.assertNotEqual(ag._normalize(ENC_GOOD), ag._normalize(ENC_GOOD.replace("0x5a", "0x5b")))
        # pure trailing-whitespace / newline differences are NOT progress
        self.assertEqual(ag._normalize(ENC_GOOD), ag._normalize(ENC_GOOD.rstrip() + "\n\n  "))

    # -- Fix 3: three disjoint error classes ------------------------------- #
    def _run_offline(self, name, items, backend):
        prev = os.getcwd()
        with TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                return ag.generate_recover_agentic(name, prompt="p", target="elf",
                                                   backend=backend, caller=Replay(items),
                                                   resume=False)
            finally:
                os.chdir(prev)

    def test_error_class_infra_connection(self):
        # a connection fault from the transport -> category 'infra' (before any compile)
        rep = self._run_offline("e_infra", [{"design": "d"}, ConnectionError], NativeBackend())
        self.assertFalse(rep.success)
        self.assertEqual(rep.attempts[-1].stage, "infra")
        self.assertEqual(rep.telemetry.get("error", {}).get("category"), "infra")

    def test_error_class_invalid_model_response(self):
        # the model returns something that is not a JSON object -> 'model-invalid'
        rep = self._run_offline("e_model", [{"design": "d"}, ["not", "a", "dict"]], NativeBackend())
        self.assertFalse(rep.success)
        self.assertEqual(rep.attempts[-1].stage, "model-invalid")
        self.assertEqual(rep.telemetry.get("error", {}).get("category"), "model-invalid")

    def test_error_class_verifier_fault(self):
        # an unexpected exception INSIDE a gate -> 'verifier-error' (not 'infra')
        class Boom:
            name = "boom"
            def compile_c(self, *a, **k):
                raise RuntimeError("compiler exploded")
        rep = self._run_offline("e_verif", [{"design": "d"}, {"encoder_code": ENC_GOOD}], Boom())
        self.assertFalse(rep.success)
        self.assertEqual(rep.attempts[-1].stage, "verifier-error")
        self.assertEqual(rep.telemetry.get("error", {}).get("category"), "verifier-error")
        self.assertIn("compiler exploded", rep.attempts[-1].detail)


class CliTests(unittest.TestCase):
    """CLI wiring for --workflow (no Docker, no Ollama)."""

    def test_default_workflow_is_monolithic(self):
        import main
        a = main._build_parser().parse_args(["generate", "--name", "x"])
        self.assertEqual(a.workflow, "monolithic")

    def test_agentic_rejects_non_recover_with_nonzero_exit_and_no_ollama(self):
        import main
        # --target pe --unsafe-native resolves a NativeBackend on this host without Docker;
        # the agentic gate must then reject genre!=recover BEFORE any model call.
        argv = ["main", "generate", "--name", "_cli_reject", "--genre", "reverse",
                "--workflow", "agentic", "--target", "pe", "--unsafe-native"]
        with mock.patch("slothctf.generator._call_ollama", side_effect=AssertionError("ollama called")), \
             mock.patch("slothctf.agentic.ollama_transport", side_effect=AssertionError("transport built")), \
             mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as cm:
                main.main()
        self.assertNotEqual(cm.exception.code, 0)   # non-zero exit
        self.assertEqual(cm.exception.code, 2)


# =========================================================================== #
# Docker-gated: full ELF/PE flows and repair paths through the real sandbox
# =========================================================================== #
@contextlib.contextmanager
def _run(name, target, backend, items, budget=None, flag=None):
    """Run an agentic generation in a throwaway cwd and YIELD (rep, caller, cdir)
    while the workspace still exists, so file-existence assertions run before the
    TemporaryDirectory is cleaned up."""
    with TemporaryDirectory() as tmp:
        prev = os.getcwd()
        os.chdir(tmp)
        try:
            caller = Replay(items)
            rep = ag.generate_recover_agentic(name, prompt="xor recover", target=target,
                                              backend=backend, caller=caller,
                                              budget=budget or ag.Budget(), flag=flag, resume=False)
            yield rep, caller, Path(tmp) / "generated" / name
        finally:
            os.chdir(prev)


@unittest.skipUnless(_READY, _SKIP)
class AgenticElfTests(unittest.TestCase):
    def _backend(self):
        from slothctf.execution import DockerLinuxBackend
        return DockerLinuxBackend()

    def test_full_valid_elf(self):
        items = [{"design": "d"}, {"encoder_code": ENC_GOOD},
                 {"challenge_code": CHAL_GOOD, "success_marker": "Correct"}, {"solver_code": SOL_GOOD}]
        with _run("ag_elf_ok", "elf", self._backend(), items) as (rep, caller, cdir):
            self.assertTrue(rep.success, [(a.stage, a.detail[:80]) for a in rep.attempts])
            self.assertEqual(rep.attempts[-1].stage, "passed")
            self.assertTrue((cdir / "bin" / "challenge").exists())
            self.assertTrue((cdir / "organizer" / "solver.py").exists())

    def test_encoder_repaired_then_valid(self):
        items = [{"design": "d"}, {"encoder_code": ENC_BROKEN}, {"encoder_code": ENC_GOOD},
                 {"challenge_code": CHAL_GOOD, "success_marker": "Correct"}, {"solver_code": SOL_GOOD}]
        with _run("ag_enc_fix", "elf", self._backend(), items) as (rep, caller, _):
            self.assertTrue(rep.success, [(a.stage, a.detail[:80]) for a in rep.attempts])
            self.assertTrue(any(a.stage == "encoder-compile" for a in rep.attempts))

    def test_challenge_compile_repaired(self):
        items = [{"design": "d"}, {"encoder_code": ENC_GOOD},
                 {"challenge_code": CHAL_BROKEN, "success_marker": "Correct"},
                 {"challenge_code": CHAL_GOOD, "success_marker": "Correct"}, {"solver_code": SOL_GOOD}]
        with _run("ag_chal_fix", "elf", self._backend(), items) as (rep, caller, _):
            self.assertTrue(rep.success, [(a.stage, a.detail[:80]) for a in rep.attempts])
            self.assertTrue(any(a.stage == "challenge-compile" for a in rep.attempts))

    def test_solver_mismatch_then_corrected(self):
        items = [{"design": "d"}, {"encoder_code": ENC_GOOD},
                 {"challenge_code": CHAL_GOOD, "success_marker": "Correct"}, {"solver_code": SOL_WRONG},
                 {"component": "solver", "code": SOL_GOOD, "reason": "fix inverse"}]
        with _run("ag_sol_fix", "elf", self._backend(), items) as (rep, caller, _):
            self.assertTrue(rep.success, [(a.stage, a.detail[:80]) for a in rep.attempts])
            self.assertTrue(any(a.stage == "solver-mismatch" for a in rep.attempts))

    def test_hardcoded_solver_rejected_by_flag_swap(self):
        flag = "sloth{fixed_demo_abc}"
        hc = f"print({flag!r})\n"  # ignores the binary; recovers only this flag
        items = [{"design": "d"}, {"encoder_code": ENC_GOOD},
                 {"challenge_code": CHAL_GOOD, "success_marker": "Correct"}, {"solver_code": hc},
                 {"component": "solver", "code": hc, "reason": "unchanged"},
                 {"component": "solver", "code": hc, "reason": "unchanged"}]
        with _run("ag_hardcode", "elf", self._backend(), items,
                  budget=ag.Budget(max_repairs_per_stage=1), flag=flag) as (rep, caller, _):
            self.assertFalse(rep.success)
            self.assertTrue(any(a.stage == "solver-hardcoded" for a in rep.attempts),
                            [(a.stage, a.detail[:60]) for a in rep.attempts])

    def test_badlen_challenge_repaired_with_diagnostic(self):
        # Phase 2 end-to-end: a challenge that rejects the correct flag (hardcoded length)
        # is repaired after the harness feeds back the concrete length diagnostic and the
        # verified encoder. The LLM still authors the fix (replayed here).
        items = [{"design": "d"}, {"encoder_code": ENC_GOOD},
                 {"challenge_code": CHAL_BADLEN, "success_marker": "FLAG_RECOVERED"},
                 {"challenge_code": CHAL_FIXED, "success_marker": "FLAG_RECOVERED"},
                 {"solver_code": SOL_GOOD}]
        with _run("ag_badlen_fix", "elf", self._backend(), items) as (rep, caller, _):
            self.assertTrue(rep.success, [(a.stage, a.detail[:80]) for a in rep.attempts])
            self.assertTrue(any(a.stage == "recover-rejects-correct" for a in rep.attempts))
            self.assertTrue(any("embedded array has" in p for p in caller.calls),
                            "concrete length diagnostic was not fed to the model")
            self.assertTrue(any(ENC_GOOD.strip() in p for p in caller.calls),
                            "verified encoder was not provided to Stage C")

    def test_crash_then_resume(self):
        import os
        from slothctf.execution import DockerLinuxBackend
        with TemporaryDirectory() as tmp:
            prev = os.getcwd()
            os.chdir(tmp)
            try:
                # run 1: design + encoder verified, then a crash on the challenge call
                c1 = Replay([{"design": "d"}, {"encoder_code": ENC_GOOD}, KeyboardInterrupt])
                with self.assertRaises(KeyboardInterrupt):
                    ag.generate_recover_agentic("ag_resume", prompt="p", target="elf",
                                                backend=DockerLinuxBackend(), caller=c1,
                                                resume=False)
                sp = Path(tmp) / "generated" / "ag_resume" / "agentic_state.json"
                self.assertTrue(sp.exists())
                st = ag.ProjectState.load(sp)
                self.assertTrue(st.is_verified("encoder"))  # encoder survived the crash
                # run 2: resume; must NOT re-ask the encoder, only challenge + solver
                c2 = Replay([{"challenge_code": CHAL_GOOD, "success_marker": "Correct"},
                             {"solver_code": SOL_GOOD}])
                rep = ag.generate_recover_agentic("ag_resume", prompt="p", target="elf",
                                                  backend=DockerLinuxBackend(), caller=c2,
                                                  resume=True)
                self.assertTrue(rep.success, [(a.stage, a.detail[:80]) for a in rep.attempts])
                self.assertEqual(len(c2.calls), 2)  # resumed: encoder not regenerated
            finally:
                os.chdir(prev)


@unittest.skipUnless(_WIN_READY and _READY, _WIN_SKIP)
class AgenticPeTests(unittest.TestCase):
    def test_full_valid_pe(self):
        from slothctf.execution import DockerWinCompatBackend
        items = [{"design": "d"}, {"encoder_code": ENC_GOOD},
                 {"challenge_code": CHAL_GOOD, "success_marker": "Correct"}, {"solver_code": SOL_GOOD}]
        with _run("ag_pe_ok", "pe", DockerWinCompatBackend(), items) as (rep, caller, cdir):
            self.assertTrue(rep.success, [(a.stage, a.detail[:80]) for a in rep.attempts])
            self.assertTrue((cdir / "bin" / "challenge.exe").exists())


if __name__ == "__main__":
    unittest.main()
