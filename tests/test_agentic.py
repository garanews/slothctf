"""Offline acceptance tests for the agentic (test-driven) recover workflow.

LLM calls are REPLAYED from pre-recorded responses (no Ollama). The Docker-gated
class drives real ELF/PE compile/run/solver through the reused primitives, with the
solver always isolated. The Docker-free class tests the state machine: dependency
invalidation, crash/resume bookkeeping, persistence, and budget limits.
"""
import contextlib
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
        # max_llm_calls=1: design is call 1, the encoder generate is call 2 -> abort
        # BEFORE any compile, so no Docker is needed and there is no infinite loop.
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
        self.assertEqual(len(caller.calls), 2)   # design + encoder, then stopped

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
