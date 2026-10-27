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

# --- Forward-Check fixtures: ONE stateful, buffer-wide transform shared verbatim by
# the encoder and the challenge (no hand-written inverse). Flag fixed at 19 bytes. ---
FC_FLAG = "sloth{abcdefghijkl}"
FC_TRANSFORM = (
    "void slothctf_transform(const unsigned char *in, size_t n, unsigned char *out) {\n"
    "    unsigned char s = 0x5a;\n"
    "    for (size_t i = 0; i < n; i++) {\n"
    "        unsigned char v = (unsigned char)(in[i] ^ s);\n"
    "        out[i] = v;\n"
    "        s = (unsigned char)(s + in[i] + (unsigned char)i);\n"
    "    }\n"
    "}\n")
FC_ENCODER_GOOD = (
    "#include <stdio.h>\n#include <string.h>\n#include <stddef.h>\n"
    "/*__TRANSFORM_FN__*/\n"
    "int main(int argc, char **argv) {\n"
    "    if (argc != 2) return 1;\n"
    "    const unsigned char *flag = (const unsigned char *)argv[1];\n"
    "    size_t n = strlen((const char *)flag);\n"
    "    unsigned char out[1024];\n"
    "    slothctf_transform(flag, n, out);\n"
    "    for (size_t i = 0; i < n; i++) { printf(\"0x%02x\", out[i]); if (i+1<n) printf(\",\"); }\n"
    "    return 0;\n"
    "}\n")
# Note: the challenge is no longer an LLM-authored fixture — in forward_check mode it is
# always ag._FC_CHALLENGE_SKELETON (harness-synthesized; see tests below that use it directly).
# independent solver: brute-forces the offset of a 19-byte window and inverts the SAME
# stateful recurrence purely from the binary's raw bytes (no encoder/challenge access).
FC_SOLVER_GOOD = (
    "import sys, re\n"
    "b = open(sys.argv[1], 'rb').read()\n"
    "L = 19\n"
    "pat = re.compile(rb'^sloth\\{[ -~]{1,40}\\}$')\n"
    "for off in range(len(b) - L + 1):\n"
    "    enc = b[off:off+L]\n"
    "    s = 0x5a\n"
    "    dec = bytearray(L)\n"
    "    for i in range(L):\n"
    "        v = enc[i] ^ s\n"
    "        dec[i] = v\n"
    "        s = (s + v + i) & 0xFF\n"
    "    if pat.match(bytes(dec)):\n"
    "        print(dec.decode()); break\n")
# same inverse math as FC_SOLVER_GOOD, but LOCATES the bytes via the NEW Stage-D guidance
# (objdump -h for section file-offset/size, then a plain file read) instead of brute-
# forcing the whole binary — proves the guided technique works end-to-end through the
# real hardened solver container, on ELF and PE, and (via Stage E's flag-swap, which
# reruns this unmodified script against a REBUILT binary) on a recompiled binary too,
# since nothing here is hardcoded — every run re-derives offsets from sys.argv[1].
FC_SOLVER_OBJDUMP = (
    "import sys, re, subprocess\n"
    "path = sys.argv[1]\n"
    "out = subprocess.run(['objdump', '-h', path], capture_output=True, text=True, timeout=10).stdout\n"
    "sections = []\n"
    "for line in out.splitlines():\n"
    "    m = re.match(r'^\\s*\\d+\\s+(\\S+)\\s+([0-9a-fA-F]+)\\s+[0-9a-fA-F]+\\s+[0-9a-fA-F]+\\s+([0-9a-fA-F]+)', line)\n"
    "    if m:\n"
    "        name, size, fileoff = m.group(1), int(m.group(2), 16), int(m.group(3), 16)\n"
    "        if name.lower() in ('.rodata', '.data', '.rdata') and size > 0:\n"
    "            sections.append((fileoff, size))\n"
    "with open(path, 'rb') as f:\n"
    "    raw = f.read()\n"
    "L = 19\n"
    "pat = re.compile(rb'^sloth\\{[ -~]{1,40}\\}$')\n"
    "for fileoff, size in sections:\n"
    "    chunk = raw[fileoff:fileoff+size]\n"
    "    for start in range(max(0, len(chunk) - L + 1)):\n"
    "        enc = chunk[start:start+L]\n"
    "        s = 0x5a\n"
    "        dec = bytearray(L)\n"
    "        for i in range(L):\n"
    "            v = enc[i] ^ s\n"
    "            dec[i] = v\n"
    "            s = (s + v + i) & 0xFF\n"
    "        if pat.match(bytes(dec)):\n"
    "            print(dec.decode()); sys.exit(0)\n")
FC_ENCODER_BROKEN = "#include <stdio.h>\n/*__TRANSFORM_FN__*/\nint main(){ not valid c ; }\n"
# a fully self-contained, independently COMPILABLE encoder that defines slothctf_transform
# inline, with NO placeholder at all — must still be REJECTED by the contract check
# (never compiled to find this out).
FC_ENCODER_AUTONOMOUS = (
    "#include <stdio.h>\n#include <string.h>\n#include <stddef.h>\n"
    + FC_TRANSFORM +
    "int main(int argc, char **argv) {\n"
    "    if (argc != 2) return 1;\n"
    "    const unsigned char *flag = (const unsigned char *)argv[1];\n"
    "    size_t n = strlen((const char *)flag);\n"
    "    unsigned char out[1024];\n"
    "    slothctf_transform(flag, n, out);\n"
    "    for (size_t i = 0; i < n; i++) { printf(\"0x%02x\", out[i]); if (i+1<n) printf(\",\"); }\n"
    "    return 0;\n"
    "}\n")


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
# raises ValueError: byte must be in range(0,256) (ELF byte 0x7f - 200 < 0) -> solver-error
SOL_RANGE_ERR = ("import sys\n"
                 "b=open(sys.argv[1],'rb').read()\n"
                 "out=bytearray()\n"
                 "for i in range(8):\n"
                 "    out.append(b[i]-200)\n"
                 "print(out.decode())\n")

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

    def test_solver_feedback_is_walled_player_equivalent(self):
        # Under inversion-first the solver feedback is player-equivalent: the solver's OWN
        # output/traceback and pass/fail only — never the flag, its length, or private evidence.
        flag = "sloth{abcdefghijkl}"
        err = ag._solver_feedback("solver-error", "", f"Traceback ... recovered {flag}", flag)
        mis = ag._solver_feedback("solver-mismatch", flag, "", flag)
        swp = ag._solver_feedback("solver-hardcoded", "", "", flag)
        for fb in (err, mis, swp):
            self.assertNotIn(flag, fb)                       # flag never leaked
            self.assertNotIn(str(len(flag)), fb)             # flag length never leaked
        self.assertIn("traceback", err.lower())              # the solver's own error is kept
        self.assertIn("mod 256", err.lower())                # generic, player-safe guidance
        self.assertIn("binary", mis.lower())

    def test_solver_independence_no_private_info_leaks(self):
        # Inversion-first: the inverse/design, encoder and challenge sources and the flag must
        # reach the challenge stage but NEVER the solver stage or its feedback.
        st = ag.ProjectState(name="demo", target="elf", difficulty="easy", flag="sloth{FLAGSECRET}")
        st.design = "FORWARD then INVERSE_SECRET_MARKER order invertibility"
        st.set_code("encoder", "ENCODER_SECRET_MARKER"); st.mark_verified("encoder")
        st.set_code("challenge", "CHALLENGE_SECRET_MARKER"); st.mark_verified("challenge")
        st.set_code("solver", "print('x')")
        secrets = ["INVERSE_SECRET_MARKER", "ENCODER_SECRET_MARKER", "CHALLENGE_SECRET_MARKER", "sloth{FLAGSECRET}"]

        # authorized stages (challenge) DO see design + encoder
        chal = ag._challenge_prompt(st)
        self.assertIn("ENCODER_SECRET_MARKER", chal)
        self.assertIn("INVERSE_SECRET_MARKER", chal)
        chal_rep = ag._repair_prompt("challenge", st, "recover-rejects-correct")
        self.assertIn("ENCODER_SECRET_MARKER", chal_rep)
        self.assertIn("INVERSE_SECRET_MARKER", chal_rep)

        # the solver sees the PUBLIC binary evidence (what a player with a disassembler sees)...
        st.binary_evidence = "EVIDENCE_PUBLIC_MARKER sections/strings/disasm"
        sol = ag._solver_prompt(st)
        sol_rep = ag._repair_prompt("solver", st, "your solver did not recover the correct flag")
        self.assertIn("EVIDENCE_PUBLIC_MARKER", sol)
        self.assertIn("EVIDENCE_PUBLIC_MARKER", sol_rep)
        # the binutils-tooling guidance (objdump/subprocess hint) rides along the same
        # evidence block and must be secret-free too
        self.assertIn("objdump", sol)
        self.assertIn("objdump", sol_rep)
        # ...but the solver generate + repair prompts must contain NONE of the private markers
        for s in secrets:
            self.assertNotIn(s, sol, f"solver generate prompt leaked {s!r}")
            self.assertNotIn(s, sol_rep, f"solver repair prompt leaked {s!r}")

        # the solver-facing feedback must not reveal the flag on ANY path:
        # generation/repair (above), flag-swap, solver-error, or Stage-E detail.
        flag = "sloth{FLAGSECRET}"
        stage_e_detail = f"the solver recovered 'x' but the flag is {flag!r}"   # what Stage E might say
        for stg, out, raw in [("solver-error", "", f"ValueError near {flag}"),
                              ("solver-mismatch", flag, stage_e_detail),
                              ("solver-hardcoded", "", stage_e_detail)]:
            fb = ag._solver_feedback(stg, out, raw, flag)
            self.assertNotIn(flag, fb, f"{stg} feedback leaked the flag")
        # design prompt now asks for forward + inverse + order + invertibility
        dp = ag._design_prompt(st).lower()
        for kw in ("forward", "inverse", "order", "invertib"):
            self.assertIn(kw, dp)

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

    # -- Forward-Check: mode isolation, resolution, persistence --------------- #
    def test_mode_defaults_to_inverse_and_resolve_is_noop(self):
        st = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}")
        self.assertEqual(st.mode, "inverse")  # unchanged default, old behaviour preserved
        raw = "RAW" + ag.TRANSFORM_PLACEHOLDER + "CODE"
        code, err, tc = ag._resolve_fc("challenge", st, raw, {})
        self.assertEqual((code, err, tc), (raw, None, None))  # no-op in inverse mode

    def test_resolve_fc_splices_transform_exactly_once(self):
        st = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}",
                             mode="forward_check")
        raw_enc = "A" + ag.TRANSFORM_PLACEHOLDER + "B"
        code, err, tc = ag._resolve_fc("encoder", st, raw_enc, {"transform_code": "FNBODY"})
        self.assertIsNone(err)
        self.assertEqual(code, "AFNBODYB")
        self.assertEqual(tc, "FNBODY")
        self.assertEqual(st.transform_code, "", "PURE function: must NOT mutate state itself")
        st.transform_code = "FNBODY"   # simulate the caller committing it (apply_fix's job)
        raw_chal = "X" + ag.TRANSFORM_PLACEHOLDER + "Y"
        code2, err2, tc2 = ag._resolve_fc("challenge", st, raw_chal, {})
        self.assertIsNone(err2)
        self.assertEqual(code2, "XFNBODYY")
        self.assertIsNone(tc2)   # challenge never produces a new transform

    def test_resolve_fc_rejects_missing_or_empty_transform_code(self):
        st = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}", mode="forward_check")
        raw = "A" + ag.TRANSFORM_PLACEHOLDER + "B"
        code, err, tc = ag._resolve_fc("encoder", st, raw, {})                      # key absent
        self.assertIsNotNone(err); self.assertEqual(code, raw); self.assertIsNone(tc)
        code, err, tc = ag._resolve_fc("encoder", st, raw, {"transform_code": "   "})  # empty/whitespace
        self.assertIsNotNone(err)

    def test_resolve_fc_rejects_missing_or_duplicated_placeholder(self):
        st = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}", mode="forward_check")
        # encoder: zero occurrences (model wrote F inline, fully compilable on its own) -> rejected
        code, err, tc = ag._resolve_fc("encoder", st, "no marker at all", {"transform_code": "FN"})
        self.assertIsNotNone(err); self.assertIn("exactly once (found 0)", err); self.assertIsNone(tc)
        # encoder: duplicated marker -> rejected
        dup = ag.TRANSFORM_PLACEHOLDER + "x" + ag.TRANSFORM_PLACEHOLDER
        code, err, tc = ag._resolve_fc("encoder", st, dup, {"transform_code": "FN"})
        self.assertIsNotNone(err); self.assertIn("found 2", err)
        # challenge: never relies on the compiler — checked the same way, before any compile
        st.transform_code = "FN"
        code, err, tc = ag._resolve_fc("challenge", st, "no marker here either", {})
        self.assertIsNotNone(err); self.assertIn("exactly once (found 0)", err)

    def test_display_code_fc_restores_placeholder_for_repair_prompt(self):
        st = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}", mode="forward_check")
        st.transform_code = "FNBODY"
        st.challenge.code = "X" + "FNBODY" + "Y"   # as actually stored (already spliced)
        shown = ag._display_code_fc("challenge", st)
        self.assertEqual(shown, "X" + ag.TRANSFORM_PLACEHOLDER + "Y")
        self.assertNotIn("FNBODY", shown)   # model is shown the marker, not the inlined body

    def test_rejected_encoder_repair_never_desyncs_transform_code(self):
        # requirement: a repair that is proposed but NOT applied (e.g. repair budget already
        # exhausted at apply time) must leave state.transform_code untouched.
        st = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}", mode="forward_check")
        st.transform_code = "OLD_FN"
        st.encoder.code = "A" + "OLD_FN" + "B"
        st.encoder.repairs = 999   # simulate budget already exhausted
        code, err, cand_tc = ag._resolve_fc("encoder", st, "A" + ag.TRANSFORM_PLACEHOLDER + "B",
                                            {"transform_code": "NEW_FN"})
        self.assertIsNone(err); self.assertEqual(cand_tc, "NEW_FN")
        self.assertEqual(st.transform_code, "OLD_FN", "resolving a candidate must not commit it")
        # only a real apply_fix-style commit (here simulated) should change it:
        # (apply_fix itself refuses when repairs >= budget — covered by the Docker test below)

    def test_route_responsible_sends_challenge_failures_to_encoder_in_forward_check(self):
        st_fc = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}", mode="forward_check")
        st_inv = ag.ProjectState(name="y", target="elf", difficulty="easy", flag="sloth{a}")  # inverse (default)
        # Stage C is harness-synthesized under forward_check: only F (encoder) can be fixed.
        self.assertEqual(ag._route_responsible("recover-rejects-correct", "challenge", st_fc), "encoder")
        self.assertEqual(ag._route_responsible("challenge-crash", "challenge", st_fc), "encoder")
        self.assertEqual(ag._route_responsible("fc-contract-challenge", "challenge", st_fc), "encoder")
        # encoder/solver failures are unaffected by the reroute
        self.assertEqual(ag._route_responsible("encoder-compile", "encoder", st_fc), "encoder")
        self.assertEqual(ag._route_responsible("solver-mismatch", "solver", st_fc), "solver")
        # inverse mode (default, unchanged): challenge failures stay on "challenge"
        self.assertEqual(ag._route_responsible("recover-rejects-correct", "challenge", st_inv), "challenge")

    def test_challenge_skeleton_resync_uses_current_transform_not_stale(self):
        # The exact property run_gate() relies on: re-resolving the FIXED skeleton always
        # reflects state.transform_code AS IT IS NOW, so an accepted encoder repair (which
        # changes F) can never leave the challenge embedding a stale, pre-repair F.
        st = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}", mode="forward_check")
        st.transform_code = "OLD_FN_BODY"
        stale, err, _ = ag._resolve_fc("challenge", st, ag._FC_CHALLENGE_SKELETON, {})
        self.assertIsNone(err)
        self.assertIn("OLD_FN_BODY", stale)
        st.transform_code = "NEW_FN_BODY"   # simulates an accepted encoder repair changing F
        fresh, err2, _ = ag._resolve_fc("challenge", st, ag._FC_CHALLENGE_SKELETON, {})
        self.assertIsNone(err2)
        self.assertIn("NEW_FN_BODY", fresh)
        self.assertNotIn("OLD_FN_BODY", fresh)

    def test_gate_challenge_hint_is_mode_aware(self):
        # The leftover "invert the encoder transform byte-for-byte" hint must not appear
        # for forward_check (there is no inverse to write there).
        import inspect
        src = inspect.getsource(ag._gate_challenge)
        self.assertIn("forward_check", src)
        self.assertIn("there is no inverse to write", src)

    def test_transform_only_change_is_progress_not_no_progress(self):
        # a repair that changes ONLY transform_code (same encoder_code wrapper text) must
        # still be recognised as a real change on the RESOLVED (post-splice) text.
        a = ag._normalize("WRAP" + "FN_A" + "WRAP")
        b = ag._normalize("WRAP" + "FN_B" + "WRAP")
        self.assertNotEqual(a, b)

    def test_encoder_repair_changing_f_invalidates_challenge_and_solver(self):
        st = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}", mode="forward_check")
        st.transform_code = "FN_OLD"
        st.set_code("encoder", "WRAP" + "FN_OLD" + "WRAP"); st.mark_verified("encoder")
        st.set_code("challenge", "CHAL" + "FN_OLD" + "CHAL"); st.mark_verified("challenge")
        st.set_code("solver", "sol"); st.mark_verified("solver")
        self.assertTrue(st.is_verified("challenge") and st.is_verified("solver"))
        # the atomic commit pattern apply_fix uses when an encoder repair changes F:
        st.set_code("encoder", "WRAP" + "FN_NEW" + "WRAP")
        st.transform_code = "FN_NEW"
        self.assertFalse(st.is_verified("encoder"))
        self.assertFalse(st.is_verified("challenge"), "a changed F must invalidate the challenge")
        self.assertFalse(st.is_verified("solver"))

    def test_mode_and_transform_code_persist_across_save_load(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "agentic_state.json"
            st = ag.ProjectState(name="x", target="elf", difficulty="easy", flag="sloth{a}",
                                 mode="forward_check", transform_code="void f(){}", path=str(p))
            st.save()
            st2 = ag.ProjectState.load(p)
            self.assertEqual(st2.mode, "forward_check")
            self.assertEqual(st2.transform_code, "void f(){}")


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
def _run(name, target, backend, items, budget=None, flag=None, mode="inverse"):
    """Run an agentic generation in a throwaway cwd and YIELD (rep, caller, cdir)
    while the workspace still exists, so file-existence assertions run before the
    TemporaryDirectory is cleaned up."""
    with TemporaryDirectory() as tmp:
        prev = os.getcwd()
        os.chdir(tmp)
        try:
            caller = Replay(items)
            rep = ag.generate_recover_agentic(name, prompt="xor recover", target=target,
                                              backend=backend, caller=caller, mode=mode,
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
                 {"solver_code": SOL_GOOD}]
        with _run("ag_sol_fix", "elf", self._backend(), items) as (rep, caller, _):
            self.assertTrue(rep.success, [(a.stage, a.detail[:80]) for a in rep.attempts])
            self.assertTrue(any(a.stage == "solver-mismatch" for a in rep.attempts))

    def test_solver_error_then_repaired_to_valid(self):
        # Inversion-first: a crashing solver gets WALLED, player-equivalent feedback (its own
        # traceback + generic guidance), with NO private info (no design/encoder/challenge/flag),
        # and the next LLM repair produces a working solver so Stage E passes.
        items = [{"design": "d"}, {"encoder_code": ENC_GOOD},
                 {"challenge_code": CHAL_GOOD, "success_marker": "Correct"},
                 {"solver_code": SOL_RANGE_ERR},
                 {"solver_code": SOL_GOOD}]
        with _run("ag_solver_err", "elf", self._backend(), items) as (rep, caller, _):
            self.assertTrue(rep.success, [(a.stage, a.detail[:80]) for a in rep.attempts])
            self.assertTrue(any(a.stage == "solver-error" for a in rep.attempts))
            # the solver repair prompt carries the solver's own traceback + mod-256 guidance...
            self.assertTrue(any("traceback" in p.lower() and "mod 256" in p.lower() for p in caller.calls),
                            "walled solver-error feedback was not fed to the model")
            # ...but NEVER the encoder, the challenge source, or the design (solver independence)
            self.assertFalse(any(ENC_GOOD.strip() in p or CHAL_GOOD.strip() in p
                                 for p in caller.calls if "solver" in p.lower() and "python" in p.lower()))

    def test_solver_no_progress_then_effective_repair(self):
        # Phase 2/3 on the ambiguous solver-mismatch path: an identical diagnose reply is
        # detected as no-progress (NOT recompiled), the feedback escalates, and the next
        # effective repair lets the solver recover the flag -> Stage E passes.
        items = [{"design": "d"}, {"encoder_code": ENC_GOOD},
                 {"challenge_code": CHAL_GOOD, "success_marker": "Correct"},
                 {"solver_code": SOL_WRONG},
                 {"solver_code": SOL_WRONG},   # no-progress (identical)
                 {"solver_code": SOL_GOOD}]  # effective
        with _run("ag_solver_np", "elf", self._backend(), items) as (rep, caller, _):
            self.assertTrue(rep.success, [(a.stage, a.detail[:80]) for a in rep.attempts])
            self.assertTrue(any(a.stage == "repair-no-progress" for a in rep.attempts))
            self.assertEqual(rep.telemetry.get("no_progress", {}).get("solver"), 1)
            # the identical reply did NOT trigger a third solver gate run (no wasted recompile)
            gate_runs = [a for a in rep.attempts if a.stage in ("solver-mismatch", "solver-ok")]
            self.assertEqual(len(gate_runs), 2, [a.stage for a in rep.attempts])
            self.assertTrue(any("IDENTICAL" in p for p in caller.calls))   # escalated feedback sent

    def test_hardcoded_solver_rejected_by_flag_swap(self):
        flag = "sloth{fixed_demo_abc}"
        hc = f"print({flag!r})\n"  # ignores the binary; recovers only this flag
        items = [{"design": "d"}, {"encoder_code": ENC_GOOD},
                 {"challenge_code": CHAL_GOOD, "success_marker": "Correct"}, {"solver_code": hc},
                 {"solver_code": hc},
                 {"solver_code": hc}]
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


@unittest.skipUnless(_READY, _SKIP)
class AgenticForwardCheckElfTests(unittest.TestCase):
    """Forward-Check mode: ONE shared transform (stateful, inter-byte dependency) is
    spliced verbatim into encoder AND the harness-synthesized challenge — no hand-written
    inverse, and (since the 15-run campaign showed the model's own I/O boilerplate was the
    actual bug) no hand-written length/exit-code glue either. Old 'inverse' mode
    (AgenticElfTests above) is untouched and used as the reference. Stage C now never asks
    the LLM anything, so these Replay lists have NO challenge-stage item."""

    def _backend(self):
        from slothctf.execution import DockerLinuxBackend
        return DockerLinuxBackend()

    def test_full_valid_elf_shared_transform_harness_synthesized_challenge(self):
        items = [{"design": "d"}, {"transform_code": FC_TRANSFORM, "encoder_code": FC_ENCODER_GOOD},
                 {"solver_code": FC_SOLVER_GOOD}]
        with _run("ag_fc_elf_ok", "elf", self._backend(), items,
                  flag=FC_FLAG, mode="forward_check") as (rep, caller, cdir):
            self.assertTrue(rep.success, [(a.stage, a.detail[:120]) for a in rep.attempts])
            self.assertTrue((cdir / "bin" / "challenge").exists())
            st = ag.ProjectState.load(cdir / "agentic_state.json")
            self.assertEqual(st.success_marker, ag.FC_SUCCESS_MARKER)
            # the SAME literal transform (incl. the stateful recurrence line) is baked into
            # BOTH built sources — proof of single shared source, not two divergent ones.
            needle = "s = (unsigned char)(s + in[i]"
            self.assertIn(needle, st.encoder.code)
            self.assertIn(needle, st.challenge.code)
            self.assertNotIn(ag.TRANSFORM_PLACEHOLDER, st.encoder.code)
            self.assertNotIn(ag.TRANSFORM_PLACEHOLDER, st.challenge.code)
            # Stage C never asked the LLM anything: exactly 3 calls (design, encoder, solver)
            self.assertEqual(len(caller.calls), 3, caller.calls)

    def test_broken_encoder_still_fixed_by_ordinary_compile_error_path(self):
        # Non-contract compile errors (e.g. a syntax error) still go through the ordinary
        # encoder-compile repair path, unaffected by the new contract pre-check.
        items = [{"design": "d"}, {"transform_code": FC_TRANSFORM, "encoder_code": FC_ENCODER_BROKEN},
                 {"transform_code": FC_TRANSFORM, "encoder_code": FC_ENCODER_GOOD},
                 {"solver_code": FC_SOLVER_GOOD}]
        with _run("ag_fc_fix", "elf", self._backend(), items,
                  flag=FC_FLAG, mode="forward_check") as (rep, caller, _):
            self.assertTrue(rep.success, [(a.stage, a.detail[:120]) for a in rep.attempts])
            self.assertTrue(any(a.stage == "encoder-compile" for a in rep.attempts))

    def test_autonomous_f_without_placeholder_rejected_without_compiling(self):
        # A fully self-contained, independently compilable encoder (slothctf_transform
        # written inline, no marker at all) must be REJECTED by the contract check — proving
        # the harness does NOT rely on the compiler to catch a missing placeholder.
        items = [{"design": "d"},
                 {"transform_code": FC_TRANSFORM, "encoder_code": FC_ENCODER_AUTONOMOUS},  # contract violation
                 {"transform_code": FC_TRANSFORM, "encoder_code": FC_ENCODER_GOOD},        # correct repair
                 {"solver_code": FC_SOLVER_GOOD}]
        with _run("ag_fc_autonomous", "elf", self._backend(), items,
                  flag=FC_FLAG, mode="forward_check") as (rep, caller, cdir):
            self.assertTrue(rep.success, [(a.stage, a.detail[:120]) for a in rep.attempts])
            stages = [a.stage for a in rep.attempts]
            self.assertIn("fc-contract-encoder", stages, [(a.stage, a.detail[:80]) for a in rep.attempts])
            # caught BEFORE any compile attempt — never relied on the compiler to detect it
            self.assertLess(stages.index("fc-contract-encoder"), stages.index("encoder-ok"))
            st = ag.ProjectState.load(cdir / "agentic_state.json")
            self.assertEqual(st.encoder.repairs, 1)   # one accepted repair (the contract
                                                       # violation itself cost no repair slot)

    def test_hardcoded_solver_still_rejected_by_flag_swap(self):
        # Stage D/E are untouched by Forward-Check: the anti-hardcode swap still fires.
        hc = f"print({FC_FLAG!r})\n"
        items = [{"design": "d"}, {"transform_code": FC_TRANSFORM, "encoder_code": FC_ENCODER_GOOD},
                 {"solver_code": hc}, {"solver_code": hc}, {"solver_code": hc}]
        with _run("ag_fc_hardcode", "elf", self._backend(), items,
                  budget=ag.Budget(max_repairs_per_stage=1), flag=FC_FLAG,
                  mode="forward_check") as (rep, caller, _):
            self.assertFalse(rep.success)
            self.assertTrue(any(a.stage == "solver-hardcoded" for a in rep.attempts))

    def test_objdump_guided_solver_passes_including_flag_swap_on_rebuilt_binary(self):
        # End-to-end proof of the new Stage-D guidance: a solver that locates bytes via
        # `objdump -h` + a plain file read (not brute-forcing the whole binary) must pass
        # all the way through Stage E — which reruns this UNCHANGED script against a
        # REBUILT binary with a different flag (flag-swap). Nothing here is hardcoded, so
        # passing both runs proves it works on the original AND a recompiled binary.
        # This demonstrates the MECHANISM only — not a claim that Qwen writes this.
        items = [{"design": "d"}, {"transform_code": FC_TRANSFORM, "encoder_code": FC_ENCODER_GOOD},
                 {"solver_code": FC_SOLVER_OBJDUMP}]
        with _run("ag_fc_objdump_solver", "elf", self._backend(), items,
                  flag=FC_FLAG, mode="forward_check") as (rep, caller, cdir):
            self.assertTrue(rep.success, [(a.stage, a.detail[:120]) for a in rep.attempts])
            self.assertEqual(rep.attempts[-1].stage, "passed")

    def test_wrong_length_input_rejected_safely(self):
        # Direct binary-level check on the harness-synthesized challenge: short, empty,
        # too-long, no-trailing-newline, and embedded-NUL inputs must all be rejected
        # cleanly (exit 0, no crash) — the embedded-NUL case is the fix under test: a
        # strlen()-based read would silently accept "<flag>\0<garbage>" as the bare flag.
        import slothctf.generator as gmod
        items = [{"design": "d"}, {"transform_code": FC_TRANSFORM, "encoder_code": FC_ENCODER_GOOD},
                 {"solver_code": FC_SOLVER_GOOD}]
        with _run("ag_fc_shortlen", "elf", self._backend(), items,
                  flag=FC_FLAG, mode="forward_check") as (rep, caller, cdir):
            self.assertTrue(rep.success, [(a.stage, a.detail[:120]) for a in rep.attempts])
            tgt = gmod.get_target("elf")
            be = self._backend()
            for label, stdin in [("short", "short\n"), ("empty", "\n"),
                                 ("too-long", "definitely_wrong_" + FC_FLAG + "\n"),
                                 ("no-newline-exact-len", FC_FLAG),
                                 ("embedded-nul-then-garbage", FC_FLAG + "\x00GARBAGE_AFTER_NUL\n"),
                                 ("embedded-nul-at-eof", FC_FLAG + "\x00")]:
                r = be.run_binary(cdir, "bin/challenge", tgt, stdin_text=stdin)
                self.assertTrue(r.ok and not r.timed_out, f"{label}: crashed/timed out: {r.stderr}")
                self.assertEqual(r.exit_code, 0, f"{label}: must exit 0")
                if label == "no-newline-exact-len":
                    self.assertIn(ag.FC_SUCCESS_MARKER, r.stdout, label)
                else:
                    self.assertIn("WRONG_INPUT", r.stdout, label)
            # the correct flag (with newline) must still succeed
            good = be.run_binary(cdir, "bin/challenge", tgt, stdin_text=FC_FLAG + "\n")
            self.assertIn(ag.FC_SUCCESS_MARKER, good.stdout)


@unittest.skipUnless(_WIN_READY and _READY, _WIN_SKIP)
class AgenticPeTests(unittest.TestCase):
    def test_full_valid_pe(self):
        from slothctf.execution import DockerWinCompatBackend
        items = [{"design": "d"}, {"encoder_code": ENC_GOOD},
                 {"challenge_code": CHAL_GOOD, "success_marker": "Correct"}, {"solver_code": SOL_GOOD}]
        with _run("ag_pe_ok", "pe", DockerWinCompatBackend(), items) as (rep, caller, cdir):
            self.assertTrue(rep.success, [(a.stage, a.detail[:80]) for a in rep.attempts])
            self.assertTrue((cdir / "bin" / "challenge.exe").exists())

    def test_full_valid_pe_forward_check_shared_transform(self):
        from slothctf.execution import DockerWinCompatBackend
        items = [{"design": "d"}, {"transform_code": FC_TRANSFORM, "encoder_code": FC_ENCODER_GOOD},
                 {"solver_code": FC_SOLVER_GOOD}]
        with _run("ag_fc_pe_ok", "pe", DockerWinCompatBackend(), items,
                  flag=FC_FLAG, mode="forward_check") as (rep, caller, cdir):
            self.assertTrue(rep.success, [(a.stage, a.detail[:120]) for a in rep.attempts])
            self.assertTrue((cdir / "bin" / "challenge.exe").exists())
            st = ag.ProjectState.load(cdir / "agentic_state.json")
            self.assertIn("s = (unsigned char)(s + in[i]", st.encoder.code)
            self.assertIn("s = (unsigned char)(s + in[i]", st.challenge.code)
            # same embedded-NUL regression as ELF, on the real .exe (byte-counting read
            # must behave identically regardless of target — PE and ELF are equal footing)
            import slothctf.generator as gmod
            be = DockerWinCompatBackend()
            tgt = gmod.get_target("pe")
            for label, stdin in [("embedded-nul-then-garbage", FC_FLAG + "\x00GARBAGE_AFTER_NUL\n"),
                                 ("embedded-nul-at-eof", FC_FLAG + "\x00")]:
                r = be.run_binary(cdir, "bin/challenge.exe", tgt, stdin_text=stdin)
                self.assertTrue(r.ok and not r.timed_out, f"{label}: crashed/timed out: {r.stderr}")
                self.assertEqual(r.exit_code, 0, f"{label}: must exit 0")
                self.assertIn("WRONG_INPUT", r.stdout, label)
            good = be.run_binary(cdir, "bin/challenge.exe", tgt, stdin_text=FC_FLAG + "\n")
            self.assertIn(ag.FC_SUCCESS_MARKER, good.stdout)

    def test_objdump_guided_solver_passes_pe_including_flag_swap_on_rebuilt_binary(self):
        # Same end-to-end proof as ELF, on PE. The solver itself always runs in the LINUX
        # sandbox (per solver_backend resolution) and calls plain `objdump` (no mingw
        # prefix) — confirming it reads the PE file correctly from there too.
        from slothctf.execution import DockerWinCompatBackend
        items = [{"design": "d"}, {"transform_code": FC_TRANSFORM, "encoder_code": FC_ENCODER_GOOD},
                 {"solver_code": FC_SOLVER_OBJDUMP}]
        with _run("ag_fc_objdump_solver_pe", "pe", DockerWinCompatBackend(), items,
                  flag=FC_FLAG, mode="forward_check") as (rep, caller, cdir):
            self.assertTrue(rep.success, [(a.stage, a.detail[:120]) for a in rep.attempts])
            self.assertEqual(rep.attempts[-1].stage, "passed")


@unittest.skipUnless(_READY, _SKIP)
class ForwardCheckCollisionRiskTests(unittest.TestCase):
    """Standalone (no agentic orchestrator, no LLM): documents a KNOWN, un-fixed risk of
    Forward-Check — a non-injective transform lets the challenge accept a WRONG input
    that happens to collide under F. This is not claimed to be solved; it is a recorded
    limitation to weigh against the mode's benefit (no hand-written-inverse divergence)."""

    def test_noninjective_transform_accepts_a_colliding_wrong_input(self):
        from slothctf.execution import DockerLinuxBackend
        import slothctf.generator as g
        backend = DockerLinuxBackend()
        tgt = g.get_target("elf")
        noninjective = (  # case-folding loses information -> distinct ASCII inputs collide
            "void slothctf_transform(const unsigned char *in, size_t n, unsigned char *out) {\n"
            "    for (size_t i = 0; i < n; i++) {\n"
            "        unsigned char c = in[i];\n"
            "        out[i] = (c >= 'a' && c <= 'z') ? (unsigned char)(c - 32) : c;\n"
            "    }\n"
            "}\n")
        real, colliding = "ABCD", "abcd"   # both fold to "ABCD" -> same encoded bytes; plain ASCII, no encoding pitfalls
        body = ",".join(f"0x{ord(c):02x}" for c in real)
        # use the REAL, shipped skeleton (not a test-only fixture) — the fix under test
        chal_src = ag._FC_CHALLENGE_SKELETON.replace(ag.TRANSFORM_PLACEHOLDER, noninjective) \
                                            .replace(g.ENCODED_PLACEHOLDER, body)
        with TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "src").mkdir(); (ws / "bin").mkdir()
            (ws / "src" / "challenge.c").write_text(chal_src, encoding="utf-8")
            r = backend.compile_c(ws, "src/challenge.c", "bin/challenge", tgt)
            self.assertTrue(r.ok, r.stderr)
            good = backend.run_binary(ws, "bin/challenge", tgt, stdin_text=real + "\n")
            bad = backend.run_binary(ws, "bin/challenge", tgt, stdin_text=colliding + "\n")
        self.assertIn(ag.FC_SUCCESS_MARKER, good.stdout)
        self.assertIn(ag.FC_SUCCESS_MARKER, bad.stdout, "documents the known risk: a non-injective F "
                      "lets a colliding WRONG input pass — Forward-Check does not prove F is injective.")


if __name__ == "__main__":
    unittest.main()
