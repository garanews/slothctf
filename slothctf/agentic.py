"""Experimental AGENTIC / test-driven generation for the `recover` genre.

This is an OPT-IN alternative to the monolithic `generator.generate_challenge`
(`--workflow agentic`). It keeps SlothCTF fully LLM-driven — Qwen invents the
mechanic and writes the C encoder, the C challenge and the Python solver — but
builds them INCREMENTALLY: each component is verified with the real sandbox before
the next is written, and the real deterministic error is fed back so the model can
fix its own work during a single generation. There are NO deterministic templates
and NO fixed algorithm list: the harness only runs checks, it never decides which
obfuscation is allowed.

Design: docs/agentic-workflow-design.md. The authoritative acceptance gate is the
UNCHANGED `generator._verify_recover_challenge` (Stage E); the staged gates reuse the
same primitives one component at a time to produce targeted feedback.

Stages:  A design → B encoder → C challenge → D solver → E final verification.

Repairs are produced by the LLM; the harness only verifies. Failures route to the
responsible component (the model diagnoses ambiguous ones); changing a component
invalidates everything downstream (via a content signature) and re-runs its gates.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import slothctf.generator as g
from slothctf.execution import SandboxUnavailable

DEFAULT_MODEL = g.DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# Budget + telemetry
# --------------------------------------------------------------------------- #
class BudgetExhausted(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Budget:
    """Hard caps so an agentic run can never loop forever or be 'valid' only
    because we spent more. Counters live here; wall time is per process (reset on
    resume — noted in the ledger)."""
    max_llm_calls: int = 24
    max_repairs_per_stage: int = 4
    max_wall_seconds: float = 1800.0
    max_tokens: Optional[int] = None
    # live counters (persisted across resume except the wall clock)
    llm_calls: int = 0
    tokens: int = 0
    _t0: float = field(default_factory=time.monotonic)

    def charge(self, toks: Optional[int]) -> None:
        self.llm_calls += 1
        if toks:
            self.tokens += int(toks)
        self.check()

    def check(self) -> None:
        if self.llm_calls > self.max_llm_calls:
            raise BudgetExhausted(f"max_llm_calls ({self.max_llm_calls})")
        if self.max_tokens is not None and self.tokens > self.max_tokens:
            raise BudgetExhausted(f"max_tokens ({self.max_tokens})")
        if (time.monotonic() - self._t0) > self.max_wall_seconds:
            raise BudgetExhausted(f"max_wall_seconds ({self.max_wall_seconds})")


# --------------------------------------------------------------------------- #
# Project state (persistent, atomic, resumable)
# --------------------------------------------------------------------------- #
@dataclass
class Component:
    code: str = ""
    last_good: Optional[str] = None     # last version that passed its gate
    status: str = "pending"             # pending | verified | failed
    verified_sig: Optional[str] = None  # signature the gate passed against
    gate_detail: str = ""
    repairs: int = 0
    artifacts: dict = field(default_factory=dict)  # e.g. encoder want_hex, bin name


_DEPS = {"encoder": [], "challenge": ["encoder"], "solver": ["challenge", "encoder"]}


@dataclass
class ProjectState:
    name: str
    target: str
    difficulty: str
    flag: str
    model: str = DEFAULT_MODEL
    prompt: str = ""
    success_marker: str = ""
    design: str = ""
    encoder: Component = field(default_factory=Component)
    challenge: Component = field(default_factory=Component)
    solver: Component = field(default_factory=Component)
    transcript: list = field(default_factory=list)
    budget_calls: int = 0
    budget_tokens: int = 0
    path: Optional[str] = None

    # -- component access / signatures ------------------------------------- #
    def comp(self, which: str) -> Component:
        return getattr(self, which)

    def sig(self, which: str) -> str:
        """Content signature of a component INCLUDING its dependencies, so changing
        an upstream component automatically invalidates the 'verified' status of
        everything downstream (no manual dirty-propagation to get wrong)."""
        h = hashlib.sha256()
        h.update(self.comp(which).code.encode("utf-8"))
        for dep in _DEPS[which]:
            h.update(b"|")
            h.update(self.comp(dep).code.encode("utf-8"))
        return h.hexdigest()[:16]

    def is_verified(self, which: str) -> bool:
        c = self.comp(which)
        return c.status == "verified" and c.verified_sig == self.sig(which)

    def set_code(self, which: str, code: str) -> None:
        c = self.comp(which)
        if c.status == "verified":
            c.last_good = c.code           # keep the last good version around
        c.code = code
        c.status = "pending"
        c.verified_sig = None

    def mark_verified(self, which: str, detail: str = "") -> None:
        c = self.comp(which)
        c.status = "verified"
        c.verified_sig = self.sig(which)
        c.gate_detail = detail

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> dict:
        def cd(c: Component) -> dict:
            return {"code": c.code, "last_good": c.last_good, "status": c.status,
                    "verified_sig": c.verified_sig, "gate_detail": c.gate_detail,
                    "repairs": c.repairs, "artifacts": c.artifacts}
        return {
            "name": self.name, "target": self.target, "difficulty": self.difficulty,
            "flag": self.flag, "model": self.model, "prompt": self.prompt,
            "success_marker": self.success_marker, "design": self.design,
            "encoder": cd(self.encoder), "challenge": cd(self.challenge),
            "solver": cd(self.solver), "transcript": self.transcript,
            "budget_calls": self.budget_calls, "budget_tokens": self.budget_tokens,
        }

    def save(self) -> None:
        if not self.path:
            return
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        with tmp.open("r+", encoding="utf-8") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)

    @classmethod
    def load(cls, path: str | Path) -> "ProjectState":
        d = json.loads(Path(path).read_text(encoding="utf-8"))

        def mkc(x: dict) -> Component:
            return Component(code=x.get("code", ""), last_good=x.get("last_good"),
                             status=x.get("status", "pending"),
                             verified_sig=x.get("verified_sig"),
                             gate_detail=x.get("gate_detail", ""),
                             repairs=int(x.get("repairs", 0)),
                             artifacts=x.get("artifacts", {}))
        st = cls(name=d["name"], target=d["target"], difficulty=d["difficulty"],
                 flag=d["flag"], model=d.get("model", DEFAULT_MODEL),
                 prompt=d.get("prompt", ""), success_marker=d.get("success_marker", ""),
                 design=d.get("design", ""), encoder=mkc(d.get("encoder", {})),
                 challenge=mkc(d.get("challenge", {})), solver=mkc(d.get("solver", {})),
                 transcript=d.get("transcript", []),
                 budget_calls=int(d.get("budget_calls", 0)),
                 budget_tokens=int(d.get("budget_tokens", 0)), path=str(path))
        return st


# --------------------------------------------------------------------------- #
# LLM transport (token-aware) + test hook
# --------------------------------------------------------------------------- #
def _invoke(caller: Callable, prompt: str, model: str) -> tuple[dict, Optional[int]]:
    """Call the LLM. `caller(prompt, model)` may return a dict (tests / offline) or
    a (dict, token_count) tuple (token-aware real transport)."""
    out = caller(prompt, model)
    if isinstance(out, tuple):
        data, toks = out
        return dict(data), toks
    return dict(out), None


def ollama_transport(base_url: str, temperature: float = 0.7):
    """Real token-aware transport around Ollama, reusing generator's JSON coercion.
    Returns a caller(prompt, model) -> (dict, total_tokens)."""
    from urllib import request

    endpoint = g._generate_endpoint(base_url)

    def caller(prompt: str, model: str):
        payload = json.dumps({"model": model, "prompt": prompt, "stream": True,
                              "format": "json", "keep_alive": "10m",
                              "options": {"temperature": temperature, "num_ctx": 4096}}).encode()
        req = request.Request(endpoint, data=payload,
                              headers={"Content-Type": "application/json"}, method="POST")
        chunks: list[str] = []
        toks = 0
        with request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                raw = raw.strip()
                if not raw:
                    continue
                piece = json.loads(raw.decode("utf-8"))
                chunks.append(piece.get("response", ""))
                if piece.get("done"):
                    toks = int(piece.get("prompt_eval_count", 0)) + int(piece.get("eval_count", 0))
                    break
        return g._coerce_json("".join(chunks).strip()), (toks or None)
    return caller


# --------------------------------------------------------------------------- #
# Stage prompts (free-form; correctness guidance only, NO algorithm list)
# --------------------------------------------------------------------------- #
_FREEDOM = ("You are free to invent ANY reverse-engineering mechanic and obfuscation "
            "(composed transforms, position-dependent checks, state machines, custom "
            "algorithms, control-flow obfuscation, original ideas). Do not restrict "
            "yourself to a fixed list like plain XOR/ADD unless you want to.")


def _platform_line(state: ProjectState) -> str:
    return ("Linux ELF (x86-64, gcc)" if state.target == "elf"
            else "Windows PE (x86-64, MinGW gcc)")


def _design_prompt(state: ProjectState) -> str:
    return (
        f"{_FREEDOM}\n\nDesign a {state.difficulty} 'recover' reverse-engineering "
        f"challenge for {_platform_line(state)}. The player must RECOVER a hidden flag by "
        "ANALYSING the compiled binary; running it must not reveal the flag. Describe the "
        "mechanic, the transform and its inverse, the program structure, the expected "
        "behaviour, the obfuscation, and why it fits the difficulty. "
        f"User request: {state.prompt}\n"
        'Return ONE JSON object: {"design": "<your technical design as text>"}.')


def _encoder_prompt(state: ProjectState) -> str:
    return (
        f"Your design:\n{state.design}\n\n"
        "Write encoder_code: a COMPLETE C program (organizer tool). It reads the flag from "
        "argv[1], applies your invertible transform, and prints the result as comma-separated "
        "single-byte 0xNN tokens and NOTHING else. Process the whole flag; no fixed length. "
        f"It must compile and run on {_platform_line(state)}. "
        'Return ONE JSON object: {"encoder_code": "<C source>"}.')


def _challenge_prompt(state: ProjectState) -> str:
    return (
        f"Your design:\n{state.design}\n\n"
        f"The encoder output bytes will be substituted at the token {g.ENCODED_PLACEHOLDER}. "
        "Write challenge_code: a COMPLETE C program given to the player. Declare the embedded "
        f"array as a file-scope `static const unsigned char NAME[] = {{ {g.ENCODED_PLACEHOLDER} }};` "
        "and actually use it. Read one line from stdin, strip the newline, print EXACTLY the "
        "success marker iff the input equals the flag, otherwise a different message; never print "
        "the flag, and keep no plaintext flag literal. Size buffers correctly (memcmp with a known "
        "length, or NUL-terminate before strcmp). Use only headers available on "
        f"{_platform_line(state)}. "
        'Return ONE JSON object: {"challenge_code": "<C source>", "success_marker": "<exact text>"}.')


def _solver_prompt(state: ProjectState) -> str:
    return (
        f"Your design:\n{state.design}\n\nThe challenge C source:\n{state.challenge.code}\n\n"
        "Write solver_code: a COMPLETE Python 3 program run as `python3 -I solver.py <binary>`. "
        "It reads ONLY the binary's raw bytes from argv[1] and recovers the flag by inverting your "
        "transform, then prints EXACTLY the flag and nothing else. It must NOT hardcode the flag, "
        "read any other file, or execute the binary. Valid Python 3 only (use # comments). "
        'Return ONE JSON object: {"solver_code": "<python source>"}.')


_REPAIR_PROMPTS = {
    "encoder": "Fix encoder_code. The harness reported:\n{err}\nCurrent encoder_code:\n{code}\n"
               'Return ONE JSON object: {{"encoder_code": "<fixed C source>"}}.',
    "challenge": "Fix challenge_code. The harness reported:\n{err}\nCurrent challenge_code:\n{code}\n"
                 'Return ONE JSON object: {{"challenge_code": "<fixed C source>", "success_marker": "<exact text>"}}.',
    "solver": "Fix solver_code. The harness reported:\n{err}\nCurrent solver_code:\n{code}\n"
              'Return ONE JSON object: {{"solver_code": "<fixed python source>"}}.',
}


def _repair_prompt(which: str, state: ProjectState, err: str) -> str:
    return _REPAIR_PROMPTS[which].format(err=err[:600], code=state.comp(which).code)


def _diagnose_prompt(state: ProjectState, err: str) -> str:
    return (
        "A verification step failed and the cause is ambiguous. Decide which component is "
        "responsible and fix it. The harness reported:\n" + err[:600] + "\n\n"
        f"design:\n{state.design}\n\nencoder_code:\n{state.encoder.code}\n\n"
        f"challenge_code:\n{state.challenge.code}\n\nsolver_code:\n{state.solver.code}\n\n"
        'Return ONE JSON object: {"component": "encoder|challenge|solver", '
        '"code": "<fixed source for that component>", "reason": "<why>"}.')


# --------------------------------------------------------------------------- #
# Stage gates (reuse generator primitives; one component at a time)
# --------------------------------------------------------------------------- #
def _enc_rel(tgt) -> str:
    return f"bin/{tgt.artifact('encoder')}"


def _chal_rel(tgt) -> str:
    return f"bin/{tgt.artifact('challenge')}"


def _gate_encoder(state, base_dir, backend, tgt, solver_backend) -> tuple[bool, str, str]:
    (base_dir / "organizer").mkdir(parents=True, exist_ok=True)
    (base_dir / "bin").mkdir(parents=True, exist_ok=True)
    (base_dir / "organizer" / "encoder.c").write_text(state.encoder.code, encoding="utf-8")
    r = backend.compile_c(base_dir, "organizer/encoder.c", _enc_rel(tgt), tgt)
    if not r.ok:
        return False, "encoder-compile", "encoder.c failed to compile:\n" + "\n".join(r.stderr.splitlines()[:20])
    run = backend.run_binary(base_dir, _enc_rel(tgt), tgt, args=[state.flag])
    if run.timed_out or not run.ok:
        return False, "encoder-crash", f"encoder did not run: {run.stderr[:200]}"
    body = g._parse_encoder_output(run.stdout)
    if not body:
        return False, "encoder-output", f"encoder must print comma-separated 0xNN bytes only; got {run.stdout[:120]!r}"
    state.encoder.artifacts = {"body": body, "want_len": len(g._tokens_to_bytes(body))}
    return True, "encoder-ok", ""


def _gate_challenge(state, base_dir, backend, tgt, solver_backend) -> tuple[bool, str, str]:
    body = state.encoder.artifacts.get("body")
    if not body:
        return False, "encoder-output", "encoder bytes unavailable (re-verify encoder)."
    code = state.challenge.code
    if g.ENCODED_PLACEHOLDER not in code:
        return False, "missing-placeholder", f"challenge_code must contain the token {g.ENCODED_PLACEHOLDER}."
    (base_dir / "src").mkdir(parents=True, exist_ok=True)
    (base_dir / "src" / "challenge.c").write_text(code.replace(g.ENCODED_PLACEHOLDER, body), encoding="utf-8")
    r = backend.compile_c(base_dir, "src/challenge.c", _chal_rel(tgt), tgt)
    if not r.ok:
        return False, "challenge-compile", "challenge.c failed to compile:\n" + "\n".join(r.stderr.splitlines()[:20])
    cbin = base_dir / _chal_rel(tgt)
    try:
        if g._flag_leaks_recover(cbin, state.flag):
            return False, "flag-leak", "the plaintext flag is readable in the binary (UTF-8/UTF-16LE)."
    except OSError as exc:
        return False, "leak-check-failed", f"could not read the binary: {exc}"
    marker = (state.success_marker or "").strip()
    if not marker:
        return False, "no-marker", "a recover challenge needs a non-empty success_marker."
    if marker in state.flag or state.flag in marker:
        return False, "bad-marker", "success_marker must not contain the flag."
    good = backend.run_binary(base_dir, _chal_rel(tgt), tgt, stdin_text=state.flag + "\n")
    bad = backend.run_binary(base_dir, _chal_rel(tgt), tgt, stdin_text="definitely_wrong_" + state.flag + "\n")
    prob = g._recover_execution_problem(good, bad, marker, state.flag)
    if prob is not None:
        return False, prob[0], prob[1]
    # contiguity stays a non-fatal diagnostic (P1)
    want = g._tokens_to_bytes(body)
    state.challenge.artifacts = {"encoded_bytes_contiguous": bool(len(want) >= 4 and want in cbin.read_bytes())}
    return True, "challenge-ok", ""


def _gate_solver(state, base_dir, backend, tgt, solver_backend) -> tuple[bool, str, str]:
    cbin = base_dir / _chal_rel(tgt)
    if not cbin.exists():
        return False, "challenge-compile", "challenge binary missing (re-verify challenge)."
    ok, out, detail = g._run_solver_isolated(state.solver.code, cbin, solver_backend)
    if not ok:
        return False, "solver-error", detail
    if out.strip() != state.flag:
        return False, "solver-mismatch", (f"the solver recovered {out.strip()[:80]!r} but the flag is "
                                          f"{state.flag!r}. Its inverse must match the encoder's transform.")
    # mini anti-hardcode swap DURING construction (reuses the authoritative helper)
    swap = g._recover_flag_swap(base_dir, state.challenge.code, state.encoder.code, state.solver.code,
                                backend, tgt, solver_backend, original_bin=cbin, original_flag=state.flag)
    if swap is not None:
        return False, swap.stage, swap.detail
    return True, "solver-ok", ""


_GATES = {"encoder": _gate_encoder, "challenge": _gate_challenge, "solver": _gate_solver}


def _stage_e(state, base_dir, backend, tgt, solver_backend) -> g.AttemptResult:
    """Authoritative, model-independent final gate: the UNCHANGED recover verifier."""
    data = {"flag": state.flag, "success_marker": state.success_marker,
            "challenge_kind": "recover", "difficulty": state.difficulty,
            "encoder_code": state.encoder.code, "challenge_code": state.challenge.code,
            "solver_code": state.solver.code}
    return g._verify_recover_challenge(base_dir, data, backend, tgt, solver_backend=solver_backend)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
# Which component an unambiguous stage failure belongs to. Ambiguous stages (the
# solver recovering the wrong flag, or the flag-swap failing) are diagnosed by the
# model, which may backtrack to the encoder, challenge or solver.
_UNAMBIGUOUS = {
    "encoder-compile": "encoder", "encoder-crash": "encoder", "encoder-output": "encoder",
    "challenge-compile": "challenge", "missing-placeholder": "challenge", "flag-leak": "challenge",
    "no-marker": "challenge", "bad-marker": "challenge", "recover-rejects-correct": "challenge",
    "recover-accepts-wrong": "challenge", "challenge-crash": "challenge",
    "challenge-bad-exit": "challenge", "too-trivial": "challenge", "output-truncated": "challenge",
    "solver-error": "solver",
}
_AMBIGUOUS = {"solver-mismatch", "solver-hardcoded", "swap-solver-error", "swap-encoder",
              "swap-compile", "swap-identical"}


def _record_call(state, budget, report, stage, kind, prompt, payload, toks, wall_s, outcome):
    resp = json.dumps(payload) if isinstance(payload, dict) else str(payload)
    state.transcript.append({"stage": stage, "kind": kind, "prompt_chars": len(prompt),
                             "resp_chars": len(resp), "tokens": toks, "wall_s": round(wall_s, 2),
                             "outcome": outcome, "ts": time.time()})
    state.budget_calls = budget.llm_calls
    state.budget_tokens = budget.tokens


def _ask(caller, model, budget, state, report, stage, kind, prompt) -> dict:
    t0 = time.monotonic()
    payload, toks = _invoke(caller, prompt, model)
    budget.charge(toks)
    _record_call(state, budget, report, stage, kind, prompt, payload, toks, time.monotonic() - t0, "ok")
    return payload


def generate_recover_agentic(
    name: str, prompt: str = "", difficulty: str = "easy", target: str = "elf", *,
    backend=None, solver_backend=None, caller: Optional[Callable] = None,
    model: str = DEFAULT_MODEL, ollama_url: str = g.DEFAULT_OLLAMA_URL,
    budget: Optional[Budget] = None, flag: Optional[str] = None, resume: bool = True,
) -> g.GenerationReport:
    """Agentic test-driven generation of a `recover` challenge. Returns a
    GenerationReport; a challenge is successful ONLY if the unchanged Stage-E gate
    passes. Never raises for a model/verification failure; infra errors are caught
    and recorded as category 'infra'. `caller` is the LLM hook (tests inject replay)."""
    base_dir = Path("generated") / name
    base_dir.mkdir(parents=True, exist_ok=True)
    tgt = g.get_target(target)
    if backend is None:
        backend = g.resolve_backend(tgt, "isolated")
    if solver_backend is None:
        solver_backend = (g.resolve_backend("elf", "isolated", image=g.DEFAULT_IMAGE)
                          if backend.name == "docker-wincompat" else backend)
    budget = budget or Budget()
    caller = caller or ollama_transport(ollama_url)

    statepath = base_dir / "agentic_state.json"
    if resume and statepath.exists():
        state = ProjectState.load(statepath)
        state.path = str(statepath)
        budget.llm_calls = state.budget_calls
        budget.tokens = state.budget_tokens
    else:
        state = ProjectState(name=name, target=tgt.name, difficulty=difficulty,
                             flag=flag or g._random_flag(), model=model, prompt=prompt,
                             path=str(statepath))
    state.save()

    report = g.GenerationReport(name=name, model=model, directory=str(base_dir), success=False)
    report.telemetry = {"workflow": "agentic", "stages": [], "repairs": {}}  # type: ignore[attr-defined]

    def apply_fix(which: str, code: str, reason: str) -> bool:
        c = state.comp(which)
        if c.repairs >= budget.max_repairs_per_stage:
            report.attempts.append(g.AttemptResult(0, False, "max-repairs", f"{which}: {reason}"[:300]))
            return False
        state.set_code(which, code)
        c.repairs += 1
        c.gate_detail = reason
        report.telemetry["repairs"][which] = c.repairs  # type: ignore[index]
        state.save()
        return True

    try:
        # Stage A — design
        if not state.design:
            payload = _ask(caller, model, budget, state, report, "design", "generate", _design_prompt(state))
            state.design = str(payload.get("design", "")).strip()
            state.save()

        STAGES = ["encoder", "challenge", "solver"]
        while True:
            budget.check()
            which = next((s for s in STAGES if not state.is_verified(s)), None)
            if which is None:
                break  # B, C, D all verified against current artifacts
            comp = state.comp(which)
            if not comp.code:  # first time we need this component
                key = f"{which}_code"
                payload = _ask(caller, model, budget, state, report, which, "generate",
                               {"encoder": _encoder_prompt, "challenge": _challenge_prompt,
                                "solver": _solver_prompt}[which](state))
                comp.code = str(payload.get(key, ""))
                if which == "challenge" and payload.get("success_marker"):
                    state.success_marker = str(payload["success_marker"]).strip()
                comp.status = "pending"
                state.save()
            ok, stage_name, detail = _GATES[which](state, base_dir, backend, tgt, solver_backend)
            report.attempts.append(g.AttemptResult(0, ok, stage_name, detail[:300]))
            report.telemetry["stages"].append({which: stage_name})  # type: ignore[index]
            if ok:
                state.mark_verified(which, stage_name)
                state.save()
                continue
            # failed -> repair (LLM-authored), possibly backtracking
            if stage_name in _AMBIGUOUS:
                payload = _ask(caller, model, budget, state, report, which, "diagnose", _diagnose_prompt(state, detail))
                responsible = str(payload.get("component", which))
                if responsible not in STAGES:
                    responsible = which
                newcode = str(payload.get("code", state.comp(responsible).code))
                reason = str(payload.get("reason", detail))[:300]
            else:
                responsible = _UNAMBIGUOUS.get(stage_name, which)
                key = f"{responsible}_code"
                payload = _ask(caller, model, budget, state, report, responsible, "repair",
                               _repair_prompt(responsible, state, detail))
                newcode = str(payload.get(key, state.comp(responsible).code))
                if responsible == "challenge" and payload.get("success_marker"):
                    state.success_marker = str(payload["success_marker"]).strip()
                reason = detail[:300]
            if not apply_fix(responsible, newcode, reason):
                state.save()
                return report  # repair budget exhausted for a component

        # Stage E — authoritative, unchanged verifier
        e_repairs = 0
        while True:
            budget.check()
            e = _stage_e(state, base_dir, backend, tgt, solver_backend)
            e.index = 0
            report.attempts.append(e)
            report.telemetry["stages"].append({"final": e.stage})  # type: ignore[index]
            if e.ok:
                _finalize(state, base_dir, tgt, model)
                report.success = True
                break
            # E failed -> diagnose + fix responsible, invalidate, re-verify B/C/D, retry E
            if e_repairs >= budget.max_repairs_per_stage:
                report.attempts.append(g.AttemptResult(0, False, "max-repairs", f"final: {e.stage}"))
                break
            e_repairs += 1
            if e.stage in _AMBIGUOUS or e.stage.startswith("swap") or e.stage == "solver-mismatch":
                payload = _ask(caller, model, budget, state, report, "final", "diagnose", _diagnose_prompt(state, e.detail))
                responsible = str(payload.get("component", "solver"))
                responsible = responsible if responsible in STAGES else "solver"
                newcode = str(payload.get("code", state.comp(responsible).code))
                reason = str(payload.get("reason", e.detail))[:300]
            else:
                responsible = _UNAMBIGUOUS.get(e.stage, "challenge")
                key = f"{responsible}_code"
                payload = _ask(caller, model, budget, state, report, responsible, "repair",
                               _repair_prompt(responsible, state, e.detail))
                newcode = str(payload.get(key, state.comp(responsible).code))
                reason = e.detail[:300]
            if not apply_fix(responsible, newcode, reason):
                break
            # re-verify any now-invalidated upstream stages before the next E
            while True:
                budget.check()
                which = next((s for s in STAGES if not state.is_verified(s)), None)
                if which is None:
                    break
                ok, sn, det = _GATES[which](state, base_dir, backend, tgt, solver_backend)
                report.attempts.append(g.AttemptResult(0, ok, sn, det[:300]))
                if ok:
                    state.mark_verified(which, sn)
                    state.save()
                else:
                    # fold back into the same repair machinery via a fresh fix
                    resp = _UNAMBIGUOUS.get(sn, which)
                    if sn in _AMBIGUOUS:
                        pl = _ask(caller, model, budget, state, report, which, "diagnose", _diagnose_prompt(state, det))
                        resp = str(pl.get("component", which)); resp = resp if resp in STAGES else which
                        nc = str(pl.get("code", state.comp(resp).code)); rs = str(pl.get("reason", det))[:300]
                    else:
                        k = f"{resp}_code"
                        pl = _ask(caller, model, budget, state, report, resp, "repair", _repair_prompt(resp, state, det))
                        nc = str(pl.get(k, state.comp(resp).code))
                        if resp == "challenge" and pl.get("success_marker"):
                            state.success_marker = str(pl["success_marker"]).strip()
                        rs = det[:300]
                    if not apply_fix(resp, nc, rs):
                        state.save()
                        return report
    except BudgetExhausted as be:
        report.attempts.append(g.AttemptResult(0, False, "budget-exhausted", be.reason))
    except (KeyboardInterrupt, SystemExit):
        state.save()
        raise
    except SandboxUnavailable as su:
        report.attempts.append(g.AttemptResult(0, False, "infra", str(su)[:300]))
    except Exception as exc:  # any other infra failure, recorded not raised
        report.attempts.append(g.AttemptResult(0, False, "infra", f"{type(exc).__name__}: {exc}"[:300]))

    state.budget_calls = budget.llm_calls
    state.budget_tokens = budget.tokens
    report.telemetry["budget"] = {"llm_calls": budget.llm_calls, "tokens": budget.tokens}  # type: ignore[index]
    state.save()
    return report


def _finalize(state: ProjectState, base_dir: Path, tgt, model: str) -> None:
    """On a Stage-E pass, write the challenge files exactly like the monolithic
    success path (README/solution rendered locally, manifest written)."""
    data = {"title": state.name, "category": "Reverse Engineering",
            "difficulty": state.difficulty, "description": "Recover the flag from the binary.",
            "flag": state.flag, "challenge_kind": "recover", "success_marker": state.success_marker,
            "encoder_code": state.encoder.code, "challenge_code": state.challenge.code,
            "solver_code": state.solver.code, "target": tgt.name}
    try:
        data["readme"] = g._render_readme(data)
        data["solution"] = g._render_solution(data)
    except Exception:
        pass
    data["generation"] = {"workflow": "agentic", "model": model, "genre": "recover",
                          "difficulty": state.difficulty, "target": tgt.name,
                          "llm_calls": state.budget_calls, "tokens": state.budget_tokens}
    g._write_challenge_files(base_dir, data)
