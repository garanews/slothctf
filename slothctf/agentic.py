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


# --------------------------------------------------------------------------- #
# Error taxonomy — three disjoint failure classes, reported as distinct
# categories so a run's outcome says WHERE it broke (env vs model vs our checks).
# --------------------------------------------------------------------------- #
class InfraError(Exception):
    """Environment/transport failure outside the model and outside our checks:
    Ollama HTTP, a connection refused/timeout/DNS error, a malformed Ollama
    stream, or the sandbox being unavailable. Reported as category 'infra'."""


class InvalidModelResponse(Exception):
    """The model replied with something unusable: not JSON, not a JSON object,
    or (on a first generation) missing the required field. This is the model's
    fault, not the environment's. Reported as category 'model-invalid'."""


class VerifierError(Exception):
    """An UNEXPECTED exception raised *inside* a verification gate or the Stage-E
    verifier — a harness/checker fault, not a legitimate 'challenge rejected'
    (gates signal rejection by returning, never by raising). Reported as
    category 'verifier-error'. SandboxUnavailable is infra, not this."""


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

    def count_failed(self) -> None:
        """Count an invocation that WAS issued to the transport but failed (timeout,
        connection error, or invalid model reply). It consumed an attempt against the
        model, so it counts toward max_llm_calls exactly like a successful one. Its
        token usage is unknown and is deliberately NOT recorded as zero. Never raises:
        precheck() ran before the call, so this increment cannot exceed the cap."""
        self.llm_calls += 1

    def precheck(self) -> None:
        """Enforce caps BEFORE a new LLM call is issued, so a limit of N permits at
        most N invocations — the (N+1)-th is refused rather than issued-then-rejected.
        This is the authoritative budget gate; `check()` remains a post-condition."""
        if self.llm_calls >= self.max_llm_calls:
            raise BudgetExhausted(f"max_llm_calls ({self.max_llm_calls})")
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            raise BudgetExhausted(f"max_tokens ({self.max_tokens})")
        if (time.monotonic() - self._t0) > self.max_wall_seconds:
            raise BudgetExhausted(f"max_wall_seconds ({self.max_wall_seconds})")

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


def _jsonl_append(path: Optional[Path], obj: dict) -> None:
    """Append one JSON record to a LOCAL append-only transcript. Best-effort: the
    transcript must never fail a run, so a write error is swallowed. Local only —
    nothing here is ever transmitted, so it may hold flags/organizer code."""
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass


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

    def transcript_path(self) -> Optional[Path]:
        return Path(self.path).with_name("transcript.jsonl") if self.path else None

    def log(self, entry: dict) -> dict:
        """Append an INTEGRAL record to the persistent transcript. The full text of
        prompts, responses, code revisions, feedback, stages and telemetry is kept —
        in `self.transcript` (saved atomically inside agentic_state.json, so it
        survives resume) and mirrored to a local append-only transcript.jsonl (which
        survives even a crash between saves). Both are organizer-local; neither is
        ever published."""
        entry = dict(entry)
        entry.setdefault("ts", time.time())
        entry["seq"] = len(self.transcript)
        self.transcript.append(entry)
        _jsonl_append(self.transcript_path(), entry)
        return entry

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
    """Call the LLM and classify any failure. `caller(prompt, model)` may return a
    dict (tests / offline) or a (dict, token_count) tuple (token-aware real transport).
    Connection/HTTP/OS faults surface as InfraError; a non-JSON / non-object reply
    surfaces as InvalidModelResponse. Control-flow exceptions pass through."""
    try:
        out = caller(prompt, model)
    except (InfraError, InvalidModelResponse, BudgetExhausted, KeyboardInterrupt, SystemExit):
        raise
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise InfraError(f"{type(exc).__name__}: {exc}") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidModelResponse(f"model reply was not valid JSON: {exc}") from exc
    data, toks = out if isinstance(out, tuple) else (out, None)
    if not isinstance(data, dict):
        raise InvalidModelResponse(f"model did not return a JSON object (got {type(data).__name__})")
    return dict(data), toks


def ollama_transport(base_url: str, temperature: float = 0.7, num_ctx: int = 4096):
    """Real token-aware transport around Ollama, reusing generator's JSON coercion.
    Returns a caller(prompt, model) -> (dict, total_tokens). `num_ctx` defaults to the
    historical 4096 so existing behaviour is unchanged when no override is given."""
    from urllib import request
    from urllib.error import URLError

    endpoint = g._generate_endpoint(base_url)

    def caller(prompt: str, model: str):
        payload = json.dumps({"model": model, "prompt": prompt, "stream": True,
                              "format": "json", "keep_alive": "10m",
                              "options": {"temperature": temperature, "num_ctx": num_ctx}}).encode()
        req = request.Request(endpoint, data=payload,
                              headers={"Content-Type": "application/json"}, method="POST")
        chunks: list[str] = []
        toks = 0
        # Transport/protocol faults -> InfraError (Ollama/HTTP/connection layer).
        try:
            with request.urlopen(req, timeout=600) as resp:
                for raw in resp:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        piece = json.loads(raw.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise InfraError(f"malformed Ollama stream chunk: {exc}") from exc
                    if piece.get("error"):
                        raise InfraError(f"Ollama reported an error: {piece['error']}")
                    chunks.append(piece.get("response", ""))
                    if piece.get("done"):
                        toks = int(piece.get("prompt_eval_count", 0)) + int(piece.get("eval_count", 0))
                        break
        except URLError as exc:  # HTTPError, connection refused, DNS, read timeout
            raise InfraError(f"Ollama transport failed ({endpoint}): {exc}") from exc
        except OSError as exc:
            raise InfraError(f"Ollama connection failed ({endpoint}): {exc}") from exc
        # Content validity is the MODEL's responsibility -> InvalidModelResponse.
        text = "".join(chunks).strip()
        try:
            data = g._coerce_json(text)
        except Exception as exc:
            raise InvalidModelResponse(f"model did not return valid JSON: {exc}; head={text[:160]!r}") from exc
        if not isinstance(data, dict):
            raise InvalidModelResponse(f"model JSON was not an object (got {type(data).__name__})")
        return data, (toks or None)
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
        "ANALYSING the compiled binary; running it must not reveal the flag.\n"
        "Inversion-first: before anything else, pin down the MATH so the forward and inverse "
        "transforms are provably consistent. In the design, state EXPLICITLY:\n"
        "  1. the FORWARD transform (what the encoder does to each byte), as precise step-by-step ops;\n"
        "  2. the exact INVERSE transform (each forward op undone);\n"
        "  3. the ORDER of operations and the ORDER the inverse must undo them (reverse order);\n"
        "  4. the INVERTIBILITY conditions (e.g. all byte arithmetic is mod 256; any multiply is by an "
        "odd constant; any per-index/position or state term is reproducible; no information is lost).\n"
        "Composed transforms, permutations, state machines and inter-byte dependencies are all allowed "
        "— but only if you can write their exact inverse. Also describe the program structure, expected "
        "behaviour, the obfuscation, and why it fits the difficulty.\n"
        f"User request: {state.prompt}\n"
        'Return ONE JSON object: {"design": "<your technical design, including forward, inverse, order '
        'and invertibility conditions, as text>"}.')


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
        "The VERIFIED encoder (organizer tool) that produced the embedded bytes — your challenge "
        f"MUST invert exactly this transform, byte for byte:\n{state.encoder.code}\n\n"
        f"The encoder output bytes will be substituted at the token {g.ENCODED_PLACEHOLDER}. "
        "Write challenge_code: a COMPLETE C program given to the player. Declare the embedded "
        f"array as a file-scope `static const unsigned char NAME[] = {{ {g.ENCODED_PLACEHOLDER} }};` "
        "and actually use it. Derive the number of bytes from the array itself (e.g. sizeof(NAME)); "
        "do NOT hardcode a flag length. Read one line from stdin, strip the newline, print EXACTLY the "
        "success marker iff the input equals the flag, otherwise a different message; never print "
        "the flag, and keep no plaintext flag literal. Size buffers correctly (memcmp with a known "
        "length, or NUL-terminate before strcmp). Use only headers available on "
        f"{_platform_line(state)}. "
        "Mathematical consistency is mandatory: the check must invert the encoder's transform EXACTLY — "
        "undo each operation, in reverse order, with the same mod-256 / position / state semantics. Do NOT "
        "add, drop or reorder any operation that is not in the encoder's transform; introduce no step the "
        "encoder did not perform. "
        'Return ONE JSON object: {"challenge_code": "<C source>", "success_marker": "<exact text>"}.')


def _binary_evidence_block(state: ProjectState) -> str:
    ev = getattr(state, "binary_evidence", "") or ""
    if not ev:
        return ""
    return ("\n\nReverse-engineering EVIDENCE from the DISTRIBUTED binary (standard tools, static "
            "analysis only — this is exactly what a player with a disassembler sees):\n" + ev + "\n")


def _solver_prompt(state: ProjectState) -> str:
    # INDEPENDENT solver: the generating model gets ONLY what a player has — the public
    # description and reverse-engineering evidence derived from the distributed binary. It must
    # NOT receive the flag, the encoder, the challenge source, or the private inverse. This keeps
    # a passing solver genuine proof that the challenge is solvable from the binary alone.
    return (
        "You are an INDEPENDENT solver with only what a normal CTF player receives. The challenge is a "
        f"{state.difficulty} '{state.name}' reverse-engineering binary for {_platform_line(state)}; the "
        "goal is to recover the hidden flag (format sloth{...}) by analysing the DISTRIBUTED BINARY "
        "alone. You are NOT given the flag, the encoder, the challenge source, or any description of the "
        "transform — infer it from the binary evidence below and the raw bytes."
        + _binary_evidence_block(state) +
        "\nWrite solver_code: a COMPLETE Python 3 program run as `python3 -I solver.py <binary>`. It reads "
        "ONLY the binary's raw bytes from argv[1], recovers the flag, and prints EXACTLY the flag and "
        "nothing else. It must NOT hardcode the flag, read any other file, or execute the binary. Do all "
        "byte arithmetic mod 256. Valid Python 3 only (use # comments). "
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
    base = _REPAIR_PROMPTS[which].format(err=err[:1500], code=state.comp(which).code)
    if which == "challenge":
        # Stage C repairs are organizer-side: the verified encoder + design are provided so the
        # challenge can be made mathematically consistent with the encoder transform.
        base += ("\n\nThe VERIFIED encoder whose transform you MUST invert exactly (derive the length "
                 f"from the embedded array, never a fixed constant):\n{state.encoder.code}\n"
                 f"Your design:\n{state.design}\n"
                 "First determine the SINGLE concrete cause of the divergence — compare your decode "
                 "length, indices, casts, order and mod-256 arithmetic against the encoder and the "
                 "reported error — then return challenge_code that changes exactly that. Add no "
                 "operation the encoder does not perform.")
    elif which == "solver":
        # solver repairs stay player-only: the binary RE evidence is public, but NEVER the
        # encoder, challenge source, design or flag.
        base += _binary_evidence_block(state)
    return base


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
        # Phase 2: attach a concrete, transform-agnostic diagnostic from the probe — the
        # real lengths (data we already hold) and what the program actually printed — so
        # the repair has something to act on besides "the correct flag was rejected".
        want_len = state.encoder.artifacts.get("want_len")
        first = ((good.stdout or "").strip().splitlines() or [""])[0][:48]
        diag = (f" | diagnostic from feeding the correct flag: the flag is {len(state.flag)} bytes and the "
                f"embedded array has {want_len} bytes (one encoded byte per flag byte); the program printed "
                f"{first!r} instead of the marker {marker!r}. Derive the number of bytes from the embedded "
                f"array itself (e.g. sizeof), never a fixed constant, and invert the encoder transform "
                f"byte-for-byte.")
        return False, prob[0], prob[1] + diag
    # contiguity stays a non-fatal diagnostic (P1)
    want = g._tokens_to_bytes(body)
    state.challenge.artifacts = {"encoded_bytes_contiguous": bool(len(want) >= 4 and want in cbin.read_bytes())}
    return True, "challenge-ok", ""


def _scrub(text: str, flag: str) -> str:
    """Remove the secret flag from any solver-facing text (solver independence)."""
    if not text:
        return text or ""
    return text.replace(flag, "<redacted-flag>") if flag else text


def _solver_feedback(stage: str, out: str, raw: str, flag: str) -> str:
    """Player-EQUIVALENT feedback for the independent solver: only what a player running
    their own solver would see — its own output or its own traceback, plus pass/fail. It
    reveals NO flag, NO flag length, NO encoded-byte count/contiguity, and nothing from the
    encoder, challenge source or private inverse. Keeps the solver genuinely independent."""
    if stage == "solver-error":
        return ("your solver exited with an error. Recover the flag ONLY from the binary's raw bytes, and do "
                "all byte arithmetic mod 256.\n--- traceback ---\n" + _scrub(raw, flag)[:800])
    if stage == "solver-mismatch":
        return (f"your solver ran but did not recover the correct flag (it printed {_scrub(out, flag)[:80]!r}). "
                "Reverse-engineer the transform from the binary's bytes and invert it; print exactly the flag.")
    # flag-swap / hardcode failures: the solver relied on something outside the binary
    return ("your solver does not recover the flag from the binary alone — it failed when the challenge was "
            "rebuilt with a different secret. Do not hardcode or assume the flag; derive it from the bytes.")


def _gate_solver(state, base_dir, backend, tgt, solver_backend) -> tuple[bool, str, str]:
    cbin = base_dir / _chal_rel(tgt)
    if not cbin.exists():
        return False, "challenge-compile", "challenge binary missing (re-verify challenge)."
    ok, out, detail = g._run_solver_isolated(state.solver.code, cbin, solver_backend)
    if not ok:
        return False, "solver-error", _solver_feedback("solver-error", "", detail, state.flag)
    if out.strip() != state.flag:
        return False, "solver-mismatch", _solver_feedback("solver-mismatch", out.strip(), "", state.flag)
    # mini anti-hardcode swap DURING construction (reuses the authoritative helper)
    swap = g._recover_flag_swap(base_dir, state.challenge.code, state.encoder.code, state.solver.code,
                                backend, tgt, solver_backend, original_bin=cbin, original_flag=state.flag)
    if swap is not None:
        # a solver-culprit swap failure gets WALLED feedback; an upstream (encoder/challenge)
        # swap failure gets the scrubbed organizer detail (it is repaired organizer-side).
        if swap.stage in ("solver-hardcoded", "swap-solver-error"):
            return False, swap.stage, _solver_feedback(swap.stage, "", "", state.flag)
        return False, swap.stage, _scrub(swap.detail, state.flag)
    return True, "solver-ok", ""


def _ensure_binary_evidence(state, base_dir, backend, tgt) -> None:
    """Compute + cache player-safe STATIC reverse-engineering evidence for the current
    challenge binary, so the INDEPENDENT solver prompts can include it. Cached per challenge
    signature (the binary only changes when the challenge does), so solver repairs reuse it
    without re-running tools. Never raises — on any failure the solver just gets no evidence."""
    try:
        sig = state.sig("challenge")
    except Exception:
        sig = None
    if getattr(state, "_evidence_sig", None) == sig and getattr(state, "binary_evidence", ""):
        return
    cbin = base_dir / _chal_rel(tgt)
    if not cbin.exists():
        return
    try:
        state.binary_evidence = backend.static_analysis(base_dir, _chal_rel(tgt), tgt)
    except Exception as exc:
        state.binary_evidence = f"(static analysis unavailable: {type(exc).__name__})"
    state._evidence_sig = sig


_GATES = {"encoder": _gate_encoder, "challenge": _gate_challenge, "solver": _gate_solver}


def _guard_verifier(fn, *args):
    """Run a verification primitive (a stage gate or Stage E). A legitimate reject is
    RETURNED by the gate as (False, stage, detail) / an AttemptResult; only an
    UNEXPECTED exception raised inside the checker is reclassified as VerifierError.
    Sandbox-unavailability and control-flow exceptions pass through unchanged."""
    try:
        return fn(*args)
    except (SandboxUnavailable, InfraError, InvalidModelResponse, BudgetExhausted,
            KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise VerifierError(f"{type(exc).__name__}: {exc}") from exc


def _classify(report, category: str, detail: str) -> None:
    """Record a terminal, classified failure: as the final attempt and in
    telemetry['error'], so the failure class is visible without scanning attempts."""
    report.attempts.append(g.AttemptResult(0, False, category, detail[:300]))
    try:
        report.telemetry["error"] = {"category": category, "detail": detail[:300]}  # type: ignore[index]
    except Exception:
        pass


def _normalize(code: str) -> str:
    """Whitespace-insensitive view of source, for no-progress detection. Trailing
    whitespace and a trailing newline do not count as a change, but ANY token change
    does — so a legitimate one-character fix is still treated as progress."""
    return "\n".join(line.rstrip() for line in (code or "").strip().splitlines())


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
# Every stage failure maps to the component responsible for it. By the time the solver
# stage runs, the challenge is already gate-verified, so a solver failure (wrong flag,
# crash, or a hardcoded solver caught by the swap) is the solver's own problem and is
# repaired with the WALLED, player-only solver prompt — never the leaking diagnose prompt
# (removed). Only swap failures that implicate the encoder/challenge route upstream.
_RESPONSIBLE = {
    "encoder-compile": "encoder", "encoder-crash": "encoder", "encoder-output": "encoder",
    "challenge-compile": "challenge", "missing-placeholder": "challenge", "flag-leak": "challenge",
    "no-marker": "challenge", "bad-marker": "challenge", "recover-rejects-correct": "challenge",
    "recover-accepts-wrong": "challenge", "challenge-crash": "challenge",
    "challenge-bad-exit": "challenge", "too-trivial": "challenge", "output-truncated": "challenge",
    "solver-error": "solver", "solver-mismatch": "solver", "solver-hardcoded": "solver",
    "swap-solver-error": "solver", "swap-encoder": "encoder", "swap-identical": "encoder",
    "swap-compile": "challenge",
}


def _record_call(state, budget, report, stage, kind, prompt, payload, toks, wall_s, outcome):
    resp = json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else str(payload)
    state.log({"type": "llm", "stage": stage, "kind": kind,
               "prompt": prompt, "response": resp,            # integral, verbatim
               "prompt_chars": len(prompt), "resp_chars": len(resp),
               "tokens": toks, "wall_s": round(wall_s, 2), "outcome": outcome,
               "llm_calls": budget.llm_calls, "budget_tokens": budget.tokens})
    state.budget_calls = budget.llm_calls
    state.budget_tokens = budget.tokens


def _ask(caller, model, budget, state, report, stage, kind, prompt) -> dict:
    budget.precheck()   # enforce the LLM-call cap BEFORE issuing the call
    t0 = time.monotonic()
    try:
        payload, toks = _invoke(caller, prompt, model)
    except (InfraError, InvalidModelResponse) as exc:
        # the call was ISSUED, so it counts toward the budget even though it failed;
        # tokens are unknown (None), not zero. Persist so the count survives resume.
        budget.count_failed()
        state.budget_calls = budget.llm_calls
        state.budget_tokens = budget.tokens
        state.log({"type": "llm", "stage": stage, "kind": kind, "prompt": prompt,
                   "response": "", "tokens": None, "wall_s": round(time.monotonic() - t0, 2),
                   "outcome": f"error:{type(exc).__name__}", "error": str(exc)[:600],
                   "llm_calls": budget.llm_calls})
        state.save()
        raise
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
    passes. Never raises for a model/verification failure. Terminal failures are
    recorded under a classified category: 'budget-exhausted', 'infra' (Ollama/HTTP/
    connection/sandbox), 'model-invalid' (unusable model reply), or 'verifier-error'
    (an unexpected fault inside a check). `caller` is the LLM hook (tests inject
    replay). Everything is logged to a local, never-published transcript."""
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
            state.log({"type": "revision", "component": which, "applied": False,
                       "note": "max-repairs", "reason": str(reason)[:600], "repairs": c.repairs})
            return False
        state.set_code(which, code)
        c.repairs += 1
        c.gate_detail = reason
        report.telemetry["repairs"][which] = c.repairs  # type: ignore[index]
        # integral: record the feedback that drove this revision AND the new code
        state.log({"type": "revision", "component": which, "applied": True,
                   "reason": str(reason)[:2000], "repairs": c.repairs,
                   "sig": state.sig(which), "code": code})
        state.save()
        return True

    def run_gate(which: str):
        ok, sn, det = _guard_verifier(_GATES[which], state, base_dir, backend, tgt, solver_backend)
        state.log({"type": "gate", "component": which, "stage": sn, "ok": ok, "detail": det[:600]})
        return ok, sn, det

    def run_stage_e():
        e = _guard_verifier(_stage_e, state, base_dir, backend, tgt, solver_backend)
        state.log({"type": "gate", "component": "final", "stage": e.stage,
                   "ok": e.ok, "detail": (e.detail or "")[:600]})
        return e

    escalate: dict[str, str] = {}   # per-component extra feedback after a no-progress repair

    def fetch_fix(responsible: str, detail: str):
        """Obtain a CHANGED fix for a known-responsible component (Phase 2/3). Pre-checks
        the repair budget so a call that could no longer be applied is never issued; when
        the model resends identical code it records a `repair-no-progress` event, escalates
        the feedback and retries WITHOUT recompiling, all within budget. Returns
        (newcode, marker) once the code actually changes, or None when the component's
        repair budget is exhausted. Applying the returned code is the caller's job."""
        if responsible == "solver":        # refresh the public binary evidence for the walled prompt
            _ensure_binary_evidence(state, base_dir, backend, tgt)
        while True:
            c = state.comp(responsible)
            if c.repairs >= budget.max_repairs_per_stage:           # never issue an un-appliable call
                report.attempts.append(g.AttemptResult(0, False, "max-repairs", f"{responsible}: {detail[:160]}"[:300]))
                state.log({"type": "max-repairs", "component": responsible})
                return None
            payload = _ask(caller, model, budget, state, report, responsible, "repair",
                           _repair_prompt(responsible, state, detail + escalate.get(responsible, "")))
            newcode = str(payload.get(f"{responsible}_code", c.code))
            marker = (str(payload["success_marker"]).strip()
                      if responsible == "challenge" and payload.get("success_marker") else None)
            if _normalize(newcode) != _normalize(c.code):          # real change -> progress
                escalate.pop(responsible, None)
                return newcode, marker
            # no progress: record, escalate the feedback, bound via the repair counter, retry
            c.repairs += 1
            np = report.telemetry.setdefault("no_progress", {})     # type: ignore[union-attr]
            np[responsible] = np.get(responsible, 0) + 1
            report.attempts.append(g.AttemptResult(0, False, "repair-no-progress",
                                                   f"{responsible}: identical reply; escalating feedback"[:300]))
            state.log({"type": "repair-no-progress", "component": responsible,
                       "repairs": c.repairs, "detail": detail[:600]})
            escalate[responsible] = ("\n\nNOTE: your previous reply was byte-for-byte IDENTICAL and fixed "
                                     "nothing — do not resend the same code. Name the ONE concrete cause of the "
                                     "failure and change exactly that part.")
            state.save()

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
            if which == "solver":   # independent solver sees public binary evidence, not sources
                _ensure_binary_evidence(state, base_dir, backend, tgt)
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
            ok, stage_name, detail = run_gate(which)
            report.attempts.append(g.AttemptResult(0, ok, stage_name, detail[:300]))
            report.telemetry["stages"].append({which: stage_name})  # type: ignore[index]
            if ok:
                state.mark_verified(which, stage_name)
                state.save()
                continue
            # failed -> repair. Every stage maps to a responsible component; the solver's
            # repair prompt + feedback are walled (player-only), so no private info leaks to it.
            responsible = _RESPONSIBLE.get(stage_name, which)
            got = fetch_fix(responsible, detail)
            if got is None:
                state.save()
                return report  # repair budget exhausted for this component
            newcode, marker = got
            if marker:
                state.success_marker = marker
            if not apply_fix(responsible, newcode, detail[:300]):
                state.save()
                return report

        # Stage E — authoritative, unchanged verifier
        e_repairs = 0
        while True:
            budget.check()
            e = run_stage_e()
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
            responsible = _RESPONSIBLE.get(e.stage, "challenge")
            # Stage E's authoritative detail may reveal the flag on a solver failure; when the
            # solver is the culprit, hand it only WALLED public feedback (never e.detail directly).
            fb = _solver_feedback(e.stage, "", e.detail or "", state.flag) if responsible == "solver" else e.detail
            got = fetch_fix(responsible, fb)
            if got is None:
                break
            newcode, marker = got
            if marker:
                state.success_marker = marker
            if not apply_fix(responsible, newcode, fb[:300]):
                break
            # re-verify any now-invalidated upstream stages before the next E
            while True:
                budget.check()
                which = next((s for s in STAGES if not state.is_verified(s)), None)
                if which is None:
                    break
                ok, sn, det = run_gate(which)
                report.attempts.append(g.AttemptResult(0, ok, sn, det[:300]))
                if ok:
                    state.mark_verified(which, sn)
                    state.save()
                else:
                    # fold back into the same repair machinery via a fresh fix
                    resp = _RESPONSIBLE.get(sn, which)
                    got = fetch_fix(resp, det)
                    if got is None:
                        state.save()
                        return report
                    nc, marker = got
                    if marker:
                        state.success_marker = marker
                    if not apply_fix(resp, nc, det[:300]):
                        state.save()
                        return report
    except BudgetExhausted as be:
        _classify(report, "budget-exhausted", be.reason)
    except (KeyboardInterrupt, SystemExit):
        raise
    except InvalidModelResponse as exc:            # the model's reply was unusable
        _classify(report, "model-invalid", str(exc)[:300])
    except InfraError as exc:                       # Ollama / HTTP / connection
        _classify(report, "infra", str(exc)[:300])
    except SandboxUnavailable as su:                # sandbox missing == infra
        _classify(report, "infra", str(su)[:300])
    except VerifierError as exc:                    # unexpected fault inside a check
        _classify(report, "verifier-error", str(exc)[:300])
    except Exception as exc:                         # unknown -> infra, recorded not raised
        _classify(report, "infra", f"{type(exc).__name__}: {exc}"[:300])
    finally:
        # Phase 4: telemetry (llm_calls, tokens, final stage) is populated on EVERY exit
        # path — success, early max-repairs return, classified error, or interrupt.
        state.budget_calls = budget.llm_calls
        state.budget_tokens = budget.tokens
        report.telemetry["budget"] = {"llm_calls": budget.llm_calls, "tokens": budget.tokens}  # type: ignore[index]
        report.telemetry["final_stage"] = report.attempts[-1].stage if report.attempts else None  # type: ignore[index]
        state.log({"type": "telemetry", "success": report.success, "telemetry": report.telemetry})
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
