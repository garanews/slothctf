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
    base = _REPAIR_PROMPTS[which].format(err=err[:1500], code=state.comp(which).code)
    if which == "challenge":
        # Stage C repairs need the verified encoder + design, and must diagnose, not guess.
        base += ("\n\nThe VERIFIED encoder whose transform you MUST invert exactly (derive the length "
                 f"from the embedded array, never a fixed constant):\n{state.encoder.code}\n"
                 f"Your design:\n{state.design}\n"
                 "First determine the SINGLE concrete cause of the divergence — compare your decode "
                 "length, indices, casts and types against the encoder and the reported error — then "
                 "return challenge_code that changes exactly that.")
    elif which == "solver":
        base += (f"\n\nThe challenge C source you must invert:\n{state.challenge.code}\n"
                 f"The verified encoder:\n{state.encoder.code}\n")
    return base


def _diagnose_prompt(state: ProjectState, err: str) -> str:
    return (
        "A verification step failed and the cause is ambiguous. Decide which component is "
        "responsible and fix it. The harness reported:\n" + err[:1500] + "\n\n"
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


def _solver_evidence(state, cbin) -> tuple:
    """Observable, organizer-side evidence shared by both solver diagnostics: the
    binary size, the expected encoded-byte count, and whether those bytes appear
    contiguously. Never reveals how to locate or invert them."""
    want_len = state.encoder.artifacts.get("want_len")
    try:
        bin_sz = cbin.stat().st_size
    except OSError:
        bin_sz = None
    contig = state.challenge.artifacts.get("encoded_bytes_contiguous")
    return want_len, bin_sz, contig


def _solver_mismatch_detail(state, cbin, got: str) -> str:
    """Build a DIAGNOSTIC (not a solution) for a solver that produced the WRONG flag.
    It separates OBSERVED facts from HYPOTHESES and from what is NOT DETERMINABLE
    without instrumenting the solver. In particular a correct output LENGTH does NOT
    prove the bytes were extracted correctly. It never reveals how to parse the binary
    or invert the transform, and does not change any acceptance condition."""
    want_len, bin_sz, contig = _solver_evidence(state, cbin)
    obs = (f"OBSERVED: the solver printed {len(got)} characters; the flag is {len(state.flag)}; the binary is "
           f"{bin_sz} bytes and {want_len} encoded bytes are expected"
           f"{', present contiguously' if contig else ''}.")
    if not got:
        hyp = ("POSSIBLE CAUSE (hypothesis): the solver did not locate/extract the data (it produced nothing); "
               "re-examine how it finds the bytes in the binary.")
    elif len(got) != len(state.flag):
        hyp = "POSSIBLE CAUSE (hypothesis): the extracted region or length is wrong."
    else:
        hyp = ("POSSIBLE CAUSES (hypotheses, not certainties): a correct output LENGTH does NOT prove correct "
               "extraction — the region read may still be wrong, OR the inverse transform may be wrong (check "
               "8-bit masking / operation order). Weigh both.")
    undet = ("NOT DETERMINABLE from outside the solver: the exact offset you read and the bytes you extracted — "
             "the harness cannot see inside your solver, so state your assumption explicitly.")
    return (f"the solver recovered {got[:80]!r} but the flag is {state.flag!r}. Its inverse must match the "
            f"encoder's transform. | {obs} {hyp} {undet}")


def _solver_error_detail(state, cbin, raw: str) -> str:
    """Build a DIAGNOSTIC for a solver that RAISED (exited non-zero). Preserves the
    exception/traceback within a safe limit and adds the same observed evidence, framed
    as facts vs hypotheses. An out-of-range byte is explicitly NOT attributed to the
    inverse algorithm alone — it may equally come from reading the wrong region. Never
    provides an implementation; asks the model to diagnose before fixing."""
    want_len, bin_sz, contig = _solver_evidence(state, cbin)
    raw = raw or ""
    low = raw.lower()
    obs = (f"OBSERVED: the solver raised an exception (traceback below). The binary is {bin_sz} bytes and "
           f"{want_len} encoded bytes are expected"
           f"{', present contiguously' if contig else ''}.")
    if "range(0, 256)" in raw or "byte must be in range" in low or "out of range" in low:
        hyp = ("POSSIBLE CAUSES (hypotheses, not certainties): an out-of-range byte can come EITHER from reading "
               "the WRONG region of the binary (the values are then not your encoded bytes) OR from 8-bit "
               "arithmetic that is not normalised (a subtract/rotate without mod-256 / &0xFF). Check BOTH the "
               "extraction region and the 8-bit handling; do not assume the inverse algorithm alone is at fault.")
    else:
        hyp = ("POSSIBLE CAUSES (hypotheses): the region you read, the extracted length, the inverse arithmetic, "
               "or a plain Python mistake — decide which from the traceback.")
    undet = ("NOT DETERMINABLE from outside the solver: the exact offset you read and the bytes you extracted. "
             "First state the single concrete cause, then return a corrected solver.")
    return f"solver exited non-zero. | {obs} {hyp} {undet}\n--- traceback (truncated) ---\n{raw[:800]}"


def _gate_solver(state, base_dir, backend, tgt, solver_backend) -> tuple[bool, str, str]:
    cbin = base_dir / _chal_rel(tgt)
    if not cbin.exists():
        return False, "challenge-compile", "challenge binary missing (re-verify challenge)."
    ok, out, detail = g._run_solver_isolated(state.solver.code, cbin, solver_backend)
    if not ok:
        return False, "solver-error", _solver_error_detail(state, cbin, detail)
    if out.strip() != state.flag:
        return False, "solver-mismatch", _solver_mismatch_detail(state, cbin, out.strip())
    # mini anti-hardcode swap DURING construction (reuses the authoritative helper)
    swap = g._recover_flag_swap(base_dir, state.challenge.code, state.encoder.code, state.solver.code,
                                backend, tgt, solver_backend, original_bin=cbin, original_flag=state.flag)
    if swap is not None:
        return False, swap.stage, swap.detail
    return True, "solver-ok", ""


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
            escalate[responsible] = ("\n\nNOTE: your previous reply was byte-for-byte identical and did NOT fix "
                                     "the defect — do not resend the same code. Name the ONE concrete cause of the "
                                     "divergence (check the decode length/indices/casts against the encoder) and "
                                     "change exactly that.")
            state.save()

    def fetch_diagnose_fix(stage_label: str, detail: str, default_resp: str):
        """No-progress-aware repair for the AMBIGUOUS path (solver-mismatch, flag-swap,
        ambiguous Stage-E). The model still CHOOSES which component to change. A diagnose
        call is never issued once no component has repair budget left; an identical reply
        is recorded as `repair-no-progress`, escalated and retried WITHOUT recompiling,
        bounded by max_repairs_per_stage no-progress rounds; a changed fix for an already
        -exhausted component is refused rather than applied. Returns (responsible, newcode,
        reason) or None when nothing applicable remains."""
        extra = ""
        np_rounds = 0
        while True:
            if all(state.comp(s).repairs >= budget.max_repairs_per_stage for s in STAGES):
                report.attempts.append(g.AttemptResult(0, False, "max-repairs", f"diagnose: {detail[:150]}"[:300]))
                state.log({"type": "max-repairs", "component": "diagnose"})
                return None
            payload = _ask(caller, model, budget, state, report, stage_label, "diagnose",
                           _diagnose_prompt(state, detail + extra))
            resp = str(payload.get("component", default_resp))
            resp = resp if resp in STAGES else default_resp
            newcode = str(payload.get("code", state.comp(resp).code))
            reason = str(payload.get("reason", detail))[:300]
            if _normalize(newcode) != _normalize(state.comp(resp).code):       # real change
                if state.comp(resp).repairs >= budget.max_repairs_per_stage:   # but no budget to apply it
                    report.attempts.append(g.AttemptResult(0, False, "max-repairs", f"{resp}: diagnose"[:300]))
                    state.log({"type": "max-repairs", "component": resp})
                    return None
                return resp, newcode, reason
            # no progress: record against the chosen component, escalate, bound, retry
            np_rounds += 1
            np = report.telemetry.setdefault("no_progress", {})     # type: ignore[union-attr]
            np[resp] = np.get(resp, 0) + 1
            report.attempts.append(g.AttemptResult(0, False, "repair-no-progress",
                                                   f"{resp}: identical diagnosis; escalating feedback"[:300]))
            state.log({"type": "repair-no-progress", "component": resp, "detail": detail[:600]})
            extra = ("\n\nNOTE: your previous diagnosis returned code IDENTICAL to the current version of that "
                     "component and fixed nothing — do not resend the same code. Identify the REAL culprit "
                     "component and change exactly the part that is wrong.")
            state.save()
            if np_rounds >= budget.max_repairs_per_stage:
                report.attempts.append(g.AttemptResult(0, False, "max-repairs", f"diagnose no-progress: {resp}"[:300]))
                return None

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
            ok, stage_name, detail = run_gate(which)
            report.attempts.append(g.AttemptResult(0, ok, stage_name, detail[:300]))
            report.telemetry["stages"].append({which: stage_name})  # type: ignore[index]
            if ok:
                state.mark_verified(which, stage_name)
                state.save()
                continue
            # failed -> repair (LLM-authored), possibly backtracking
            if stage_name in _AMBIGUOUS:
                # ambiguous (solver-mismatch / swap): the model picks the responsible component.
                got = fetch_diagnose_fix(which, detail, which)
                if got is None:
                    state.save()
                    return report
                responsible, newcode, reason = got
                if not apply_fix(responsible, newcode, reason):
                    state.save()
                    return report
            else:
                # unambiguous: known responsible component; fetch a CHANGED fix within budget.
                responsible = _UNAMBIGUOUS.get(stage_name, which)
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
            if e.stage in _AMBIGUOUS or e.stage.startswith("swap") or e.stage == "solver-mismatch":
                got = fetch_diagnose_fix("final", e.detail, "solver")
                if got is None:
                    break
                responsible, newcode, reason = got
                if not apply_fix(responsible, newcode, reason):
                    break
            else:
                responsible = _UNAMBIGUOUS.get(e.stage, "challenge")
                got = fetch_fix(responsible, e.detail)
                if got is None:
                    break
                newcode, marker = got
                if marker:
                    state.success_marker = marker
                if not apply_fix(responsible, newcode, e.detail[:300]):
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
                    if sn in _AMBIGUOUS:
                        got = fetch_diagnose_fix(which, det, which)
                        if got is None:
                            state.save()
                            return report
                        resp, nc, rs = got
                        if not apply_fix(resp, nc, rs):
                            state.save()
                            return report
                    else:
                        resp = _UNAMBIGUOUS.get(sn, which)
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
