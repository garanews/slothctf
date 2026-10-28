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

Two Stage-B/C modes (ProjectState.mode, default "inverse" — unchanged behaviour):
  "inverse"       — the model writes the challenge's inverse by hand (original).
  "forward_check" — EXPERIMENTAL. The model writes ONE transform function F (any buffer-
                     wide, stateful, composed algorithm); the harness splices that SAME
                     verified C source into both the encoder and the challenge (which
                     re-applies F to the input and compares), so no hand-written inverse
                     is needed. Stage D/E are identical in both modes.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
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
    mode: str = "inverse"       # "inverse" (default, unchanged) | "forward_check" (experimental)
    transform_code: str = ""   # forward_check: the ONE shared F implementation (encoder+challenge)
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
            "mode": self.mode, "transform_code": self.transform_code,
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
                 design=d.get("design", ""), mode=d.get("mode", "inverse"),
                 transform_code=d.get("transform_code", ""), encoder=mkc(d.get("encoder", {})),
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


# --------------------------------------------------------------------------- #
# Forward-Check mode (experimental): ONE shared transform F, reused verbatim by
# encoder AND challenge — no hand-written inverse. Splice point is a plain C
# comment token, same mechanism as g.ENCODED_PLACEHOLDER (text substitution, no
# parsing). F may be stateful / buffer-wide (permutations, state machines,
# inter-byte dependencies all remain allowed) — it just must be a pure, callable
# function of (in, n).
# --------------------------------------------------------------------------- #
TRANSFORM_PLACEHOLDER = "/*__TRANSFORM_FN__*/"
TRANSFORM_FN_SIG = "void slothctf_transform(const unsigned char *in, size_t n, unsigned char *out)"


def _encoder_prompt_fc(state: ProjectState) -> str:
    return (
        f"Your design:\n{state.design}\n\n"
        "FORWARD-CHECK mode: you never write an inverse. Your transform will be compiled "
        "VERBATIM into both the organizer encoder and the player challenge; the challenge just "
        "re-applies it and compares bytes. Write TWO fields:\n"
        f"1. transform_code: a SELF-CONTAINED C function, exactly this signature:\n   {TRANSFORM_FN_SIG};\n"
        "   It may use permutations, a running state, inter-byte dependencies or any composed "
        "operation over the WHOLE buffer — not restricted to independent per-byte ops. It must be "
        "a PURE function of (in, n): no I/O, no mutable state across calls, no randomness.\n"
        f"2. encoder_code: a COMPLETE C program (organizer tool) containing the literal token "
        f"{TRANSFORM_PLACEHOLDER} near the TOP of the file (right after your #includes, before any "
        "other code, so the function is declared before main() calls it) — the harness inserts your "
        "transform_code there verbatim. main() reads the flag from argv[1], calls "
        "slothctf_transform(flag, strlen(flag), out), and prints out as comma-separated single-byte "
        f"0xNN tokens and NOTHING else. Process the whole flag; no fixed length. Must compile on "
        f"{_platform_line(state)}. "
        'Return ONE JSON object: {"transform_code": "<C function source>", '
        f'"encoder_code": "<C source containing {TRANSFORM_PLACEHOLDER}>"}}.')


# Forward-check Stage C is NOT LLM-authored: the 15-run campaign showed 7/7 challenge
# failures were the SAME structural bug class — the model's own hand-written fgets/
# strlen/sizeof(NAME) buffer arithmetic around calling the shared F (off-by-one buffer
# sizes, a bogus "exclude the null terminator" subtraction, even a genuine 1-byte stack
# overflow from `fgets(buf, sizeof(buf)+1, ...)`), NOT a transform/inverse problem (zero
# fc-contract or F-divergence failures occurred). This boilerplate has NO algorithmic
# content — reading a line, checking its length against a known array size, calling F,
# comparing, printing a fixed marker — so it is generated deterministically, exactly
# once, correctness-reviewed here instead of re-derived (and re-broken) by the model on
# every run. F itself remains 100% LLM-invented; only this I/O glue is fixed.
FC_SUCCESS_MARKER = "FLAG_RECOVERED"
_FC_CHALLENGE_SKELETON = (
    "#include <stdio.h>\n#include <string.h>\n#include <stddef.h>\n\n"
    f"{TRANSFORM_PLACEHOLDER}\n\n"
    f"static const unsigned char SLOTHCTF_ENCODED[] = {{ {g.ENCODED_PLACEHOLDER} }};\n"
    "#define SLOTHCTF_FLAG_LEN (sizeof(SLOTHCTF_ENCODED))\n\n"
    "int main(void) {\n"
    "    /* Read RAW bytes one at a time and count them ourselves — never strlen() on a\n"
    "     * fgets() buffer, which stops at the FIRST embedded NUL and would silently accept\n"
    "     * \"<flag>\\0<garbage>\" as if it were just <flag>. A line is exactly the bytes up\n"
    "     * to '\\n' or EOF, NUL bytes included; n is always the TRUE count received. */\n"
    "    unsigned char buf[SLOTHCTF_FLAG_LEN];\n"
    "    size_t n = 0;\n"
    "    int c;\n"
    "    while ((c = fgetc(stdin)) != EOF && c != '\\n') {\n"
    "        if (n < SLOTHCTF_FLAG_LEN) {\n"
    "            buf[n] = (unsigned char)c;\n"
    "        }\n"
    "        n++;   /* keep counting past the buffer so a too-long line is still rejected */\n"
    "    }\n"
    "    if (n != SLOTHCTF_FLAG_LEN) {\n"
    "        printf(\"WRONG_INPUT\\n\");\n"
    "        return 0;\n"
    "    }\n"
    "    unsigned char computed[SLOTHCTF_FLAG_LEN];\n"
    "    slothctf_transform(buf, SLOTHCTF_FLAG_LEN, computed);\n"
    "    if (memcmp(computed, SLOTHCTF_ENCODED, SLOTHCTF_FLAG_LEN) == 0) {\n"
    f"        printf(\"{FC_SUCCESS_MARKER}\\n\");\n"
    "    } else {\n"
    "        printf(\"WRONG_INPUT\\n\");\n"
    "    }\n"
    "    return 0;\n"
    "}\n")


# Verified (4/7 Stage-D failures in the 15-run campaign were hand-rolled ELF/PE parsing
# bugs: struct.error, "Could not determine .rodata boundaries", "Could not find encoded
# flag start", a ZeroDivisionError from a failed key extraction). Reliable section/offset
# lookup has no RE content — it is a solved, mechanical problem — so the solver is told it
# can call the SAME standard binutils already used to produce the evidence above, on its
# own argv[1], instead of re-deriving ELF/PE section tables from scratch in pure Python.
# This is guidance only: no new tool, no new container, no change to the solver's contract
# (still one self-contained script reading only argv[1]) — `objdump`/`strings` are already
# on PATH in the exact sandbox the solver runs in (confirmed: objdump auto-detects BOTH ELF
# and PE; `readelf` is ELF-only, so prefer objdump for target-agnostic behaviour).
_SOLVER_TOOLING_HINT = (
    "\nYou MAY call standard binutils yourself at runtime via subprocess, against "
    "sys.argv[1] (the file you are actually given — do this EVERY run; never hardcode an "
    "offset you saw once, since the binary may be rebuilt with a different flag later): "
    "`objdump` and `strings` are on PATH and work on BOTH ELF and PE (objdump auto-detects "
    "the format; `readelf` is ELF-only, so prefer objdump). This only helps you locate "
    "bytes reliably — it does NOT find the flag or invert anything for you; that stays your "
    "own analysis. Example: get each section's exact file offset/size from `objdump -h`, "
    "then read those bytes directly from the file (no need to parse the harder -s/-d "
    "output):\n"
    "    out = subprocess.run(['objdump', '-h', path], capture_output=True, text=True).stdout\n"
    "    for line in out.splitlines():\n"
    "        m = re.match(r'^\\s*\\d+\\s+(\\S+)\\s+([0-9a-fA-F]+)\\s+[0-9a-fA-F]+\\s+"
    "[0-9a-fA-F]+\\s+([0-9a-fA-F]+)', line)\n"
    "        if m:\n"
    "            name, size, fileoff = m.group(1), int(m.group(2), 16), int(m.group(3), 16)\n"
    "            # name is e.g. .rodata/.data (ELF) or .rdata/.data (PE); read bytes with\n"
    "            # open(path,'rb').seek(fileoff) + .read(size) — a plain file read, not a\n"
    "            # hand-rolled ELF/PE header walk. A missing section just yields no match.\n"
    "This runs instantly and shares your script's own time limit; no setup needed.\n")


def _binary_evidence_block(state: ProjectState) -> str:
    ev = getattr(state, "binary_evidence", "") or ""
    if not ev:
        return ""
    return ("\n\nReverse-engineering EVIDENCE from the DISTRIBUTED binary (standard tools, static "
            "analysis only — this is exactly what a player with a disassembler sees):\n" + ev + "\n"
            + _SOLVER_TOOLING_HINT)


_DISASM_LINE_RE = re.compile(r"^\s*([0-9a-fA-F]+):\t(.*)$")


def _parse_disasm_lines(binary_evidence: str) -> dict[str, str]:
    """Extract {normalized hex address: full disassembly line} from the '=== disassembly'
    section of the SAME static-analysis text already shown to the solver model — never
    re-runs objdump, so the provenance check is against exactly what the model actually
    saw (including whatever truncation applied to it, which is only fair)."""
    out: dict[str, str] = {}
    if not binary_evidence:
        return out
    idx = binary_evidence.find("=== disassembly")
    if idx == -1:
        return out
    for line in binary_evidence[idx:].splitlines():
        m = _DISASM_LINE_RE.match(line)
        if m:
            out[_normalize_addr(m.group(1))] = line.strip()
    return out


def _normalize_addr(addr: str) -> str:
    a = addr.strip().lower()
    if a.startswith("0x"):
        a = a[2:]
    return a.lstrip("0") or "0"


def _normalize_ws(text: str) -> str:
    """Collapse whitespace runs to a single space — objdump pads operand columns with
    a variable number of spaces/tabs depending on the instruction's encoded byte count;
    that is a FORMATTING difference, never a content one, so it must not cause a real
    citation to be rejected as fabricated. Used for human-readable error messages."""
    return re.sub(r"\s+", " ", text).strip()


def _insn_key(text: str) -> str:
    """Comparison key for instruction-text EQUALITY: strips ALL whitespace (not just
    collapsing runs), so the mnemonic/operand gap width and comma-spacing conventions
    ("eax, 0x3b" vs "eax,0x3b") never cause a real, complete citation to be rejected —
    only the actual sequence of non-whitespace characters (mnemonic + every operand,
    in order) has to match. Content, never formatting."""
    return re.sub(r"\s+", "", text)


def _insn_only(line: str) -> str:
    """Extract just the mnemonic+operands field of one parsed disassembly line (drop
    the address/raw-bytes columns and any trailing objdump '# ...' relocation/symbol
    comment — annotation, not part of the instruction), normalized for comparison.
    A PLT-style target suffix like '<memcmp@plt>' is part of the operand text itself
    (no '#'), so it is preserved — PLT references stay supported."""
    field = line.split("\t")[-1] if "\t" in line else line
    field = field.split("#", 1)[0]
    return _normalize_ws(field).lower()


def _check_solver_evidence(payload: dict, state: ProjectState) -> Optional[str]:
    """Reverse-Engineering-First provenance check: every (addr, insn) pair the model cites
    in payload["analysis"]["evidence"] must match the WHOLE instruction (mnemonic AND
    operands, not an arbitrary substring of it) at that address in the disassembly the
    model was ACTUALLY given (state.binary_evidence) — modulo whitespace/formatting only.
    A fabricated address, a fabricated instruction, or a bare/partial fragment (e.g. just
    the mnemonic) that does not name the full instruction is rejected. This checks ONLY
    that the citation is real and complete; it never judges whether the model's
    interpretation (the "claim") of that instruction is mathematically correct — that
    stays the model's own analysis, same as before. An honest "insufficient_evidence"
    with no citations is accepted, not punished — but is never treated as a verified,
    grounded reconstruction (see _evidence_grounded)."""
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        return 'your reply is missing an "analysis" object with your reverse-engineering evidence.'
    items = analysis.get("evidence") or []
    if not isinstance(items, list):
        return '"analysis.evidence" must be a list of {"addr": ..., "insn": ...} objects.'
    if not items:
        if analysis.get("insufficient_evidence"):
            return None
        return ('"analysis.evidence" is empty — cite at least one real (addr, insn) pair copied from '
                'the disassembly you were given, or set "insufficient_evidence": true and explain why '
                'in "notes".')
    lines = _parse_disasm_lines(getattr(state, "binary_evidence", "") or "")
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "addr" not in item or "insn" not in item:
            return f'analysis.evidence[{i}] must have both "addr" and "insn".'
        addr = str(item["addr"])
        insn = _normalize_ws(str(item["insn"]).split("#", 1)[0]).lower()
        if not insn:
            return (f'analysis.evidence[{i}] has an empty "insn" — cite the FULL instruction '
                    "(mnemonic and operands) at that address, not just the address.")
        real = lines.get(_normalize_addr(addr))
        if real is None:
            return (f"analysis.evidence[{i}] cites address {addr!r}, which does not appear in the "
                    "disassembly you were given — do not invent addresses.")
        if _insn_key(insn) != _insn_key(_insn_only(real)):
            return (f"analysis.evidence[{i}] cites instruction {item['insn']!r} at {addr!r}, but the "
                    f"disassembly you were given actually shows {_insn_only(real)!r} there — cite the "
                    "FULL instruction exactly as shown (mnemonic and operands), not a fragment, "
                    "paraphrase or invention.")
    return None


def _evidence_grounded(payload: dict) -> bool:
    """True only if the solver reply cited at least one evidence item that PASSED
    _check_solver_evidence (i.e. this is only meaningful to call once that check has
    already returned None). An honest "insufficient_evidence" reply is accepted but is
    NOT grounded — diagnostics must be able to tell the two apart, never conflating a
    verified reconstruction with an admitted absence of one."""
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        return False
    items = analysis.get("evidence") or []
    return isinstance(items, list) and len(items) > 0


def _log_solver_evidence(state: ProjectState, report, payload: dict, *, where: str) -> None:
    """Record, for DIAGNOSTICS ONLY, whether an ACCEPTED solver reply's analysis was a
    verified, grounded reconstruction (real cited evidence — _evidence_grounded) or an
    honest admission of insufficient evidence — these must never be conflated in
    reporting. Purely additive: never affects which stage passes or fails. Stage E
    remains the only end-to-end validity gate, completely unchanged by this."""
    grounded = _evidence_grounded(payload)
    analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
    state.log({"type": "solver-analysis", "where": where, "grounded": grounded,
               "insufficient_evidence": bool(analysis.get("insufficient_evidence")),
               "evidence_count": len(analysis.get("evidence") or [])})
    hist = report.telemetry.setdefault("solver_evidence", [])  # type: ignore[union-attr]
    hist.append({"where": where, "grounded": grounded})


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
        "\n\nBEFORE writing any code, reverse-engineer the binary from the disassembly above. Identify, "
        "each with REAL evidence: where the flag comparison happens; how the data length is determined, "
        "if it can be; what operations the transform actually performs and in what order. Do NOT assume "
        "the transform kept its source name or is even a separate function — the binary may be stripped "
        "and/or the function inlined into main; find it by behaviour, not by name or symbol. Every "
        "evidence item must be an address and instruction text COPIED from the disassembly you were "
        "given, never paraphrased or invented — a fabricated citation is rejected outright. If the "
        "disassembly given to you is not enough to determine something, say so (insufficient_evidence) "
        "instead of inventing it.\n"
        "Then write solver_code: a COMPLETE Python 3 program run as `python3 -I solver.py <binary>`, "
        "implementing ONLY the operations you found evidence for, inverted in order. It reads ONLY the "
        "binary's raw bytes from argv[1] (the file it is ACTUALLY given — recompute everything at "
        "runtime, never hardcode a value you only saw once, since the binary may be rebuilt with a "
        "different flag later), recovers the flag, and prints EXACTLY the flag and nothing else. It must "
        "NOT hardcode the flag, read any other file, or execute the binary. Do all byte arithmetic mod "
        "256. Valid Python 3 only (use # comments).\n"
        'Return ONE JSON object: {"analysis": {"evidence": [{"addr": "<hex address, copied>", '
        '"insn": "<instruction text at that address, copied>", "claim": "<what this shows>"}, ...], '
        '"insufficient_evidence": <true|false>, "notes": "<free text>"}, '
        '"solver_code": "<python source>"}.')


_REPAIR_PROMPTS = {
    "encoder": "Fix encoder_code. The harness reported:\n{err}\nCurrent encoder_code:\n{code}\n"
               'Return ONE JSON object: {{"encoder_code": "<fixed C source>"}}.',
    "challenge": "Fix challenge_code. The harness reported:\n{err}\nCurrent challenge_code:\n{code}\n"
                 'Return ONE JSON object: {{"challenge_code": "<fixed C source>", "success_marker": "<exact text>"}}.',
    "solver": "Fix your analysis and/or solver_code. The harness reported:\n{err}\n"
              "Current solver_code:\n{code}\n"
              "Evidence items must be addresses/instructions COPIED from the disassembly below, never "
              "invented. Return ONE JSON object: "
              '{{"analysis": {{"evidence": [{{"addr": "...", "insn": "...", "claim": "..."}}], '
              '"insufficient_evidence": <true|false>, "notes": "..."}}, '
              '"solver_code": "<fixed python source>"}}.',
}


def _display_code_fc(which: str, state: ProjectState) -> str:
    """Forward-check only: the STORED component code already has F's body spliced in
    (Stage E needs it that way). For display in a repair prompt, restore the single
    placeholder marker in its place — so the model edits a version that still shows the
    marker, and naturally keeps/returns it, instead of seeing (and risking copying or
    tweaking) F's inlined body. Falls back to the raw stored code if the exact transform
    text can't be found verbatim (defensive; should not happen once resolved)."""
    code = state.comp(which).code
    tc = state.transform_code
    if tc and tc in code:
        return code.replace(tc, TRANSFORM_PLACEHOLDER)
    return code


def _repair_prompt(which: str, state: ProjectState, err: str) -> str:
    if which == "encoder" and state.mode == "forward_check":
        # Forward-check: always re-ask for BOTH fields (transform_code may need the fix, or
        # just encoder_code's I/O glue) — a single shared-source repair, not two divergent ones.
        return (f"Fix the transform/encoder. The harness reported:\n{err[:1500]}\n"
                f"Current transform_code:\n{state.transform_code}\n"
                f"Current encoder_code (marker shown where your transform is spliced in):\n"
                f"{_display_code_fc('encoder', state)}\n"
                "FORWARD-CHECK mode: this transform_code is compiled VERBATIM into both the "
                "encoder and the player challenge — fix whichever field caused the error. Keep "
                f"the literal token {TRANSFORM_PLACEHOLDER} near the top of encoder_code (exactly "
                f"once), and keep the signature {TRANSFORM_FN_SIG}. "
                'Return ONE JSON object: {"transform_code": "<C function source>", "encoder_code": "<C source>"}.')
    # NOTE: "challenge" repairs under forward_check never happen — Stage C is harness-
    # synthesized there (see _FC_CHALLENGE_SKELETON) and failures route to "encoder"
    # instead (only F can be at fault); see _route_responsible.
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


def _resolve_fc(which: str, state: ProjectState, raw_code: str, payload: dict) -> tuple[str, Optional[str], Optional[str]]:
    """Forward-check mode only. PURE function — never mutates `state` — so a rejected,
    identical or ultimately-unapplied reply can never desync state.transform_code from
    the actually-stored state.encoder.code; the caller commits the returned candidate
    transform ONLY when it accepts the code (atomically with state.set_code).

    Returns (code, error, candidate_transform):
      - on success: (code with F spliced in exactly once, None, candidate_transform_or_None)
      - on a CONTRACT violation (missing/empty transform_code, or the placeholder absent or
        duplicated): (raw_code unchanged, <error detail>, None) — the caller must treat this
        as a repairable failure WITHOUT compiling (never relies on the compiler to catch it).
    `candidate_transform` is only meaningful for which == "encoder" (the new F the caller
    should commit to state.transform_code once this exact code is accepted)."""
    if state.mode != "forward_check":
        return raw_code, None, None
    if which == "encoder":
        tc = str(payload.get("transform_code", "")).strip()
        if not tc:
            return raw_code, "transform_code is missing or empty; return a non-empty C function.", None
        n = raw_code.count(TRANSFORM_PLACEHOLDER)
        if n != 1:
            return raw_code, (
                f"encoder_code must contain the token {TRANSFORM_PLACEHOLDER} exactly once (found {n}). "
                "Do not write slothctf_transform's body directly in encoder_code — put it ONLY in "
                f"transform_code, and mark its single splice point in encoder_code with {TRANSFORM_PLACEHOLDER}."
            ), None
        return raw_code.replace(TRANSFORM_PLACEHOLDER, tc), None, tc
    if which == "challenge":
        n = raw_code.count(TRANSFORM_PLACEHOLDER)
        if n != 1:
            return raw_code, (
                f"challenge_code must contain the token {TRANSFORM_PLACEHOLDER} exactly once (found {n}). "
                "Do not rewrite or inline slothctf_transform — the harness inserts the verified one at "
                "that single marker; restore it if your previous reply removed or duplicated it."
            ), None
        if not state.transform_code.strip():
            return raw_code, "no verified transform is available yet (re-verify the encoder first).", None
        return raw_code.replace(TRANSFORM_PLACEHOLDER, state.transform_code), None, None
    return raw_code, None, None


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
    same_length_bad = [
        (label, backend.run_binary(base_dir, _chal_rel(tgt), tgt, stdin_text=cand + "\n"))
        for label, cand in g._recover_same_length_mutations(state.flag)
    ]
    prob = g._recover_execution_problem(good, bad, marker, state.flag, same_length_bad=same_length_bad)
    if prob is not None:
        # Phase 2: attach a concrete, transform-agnostic diagnostic from the probe — the
        # real lengths (data we already hold) and what the program actually printed — so
        # the repair has something to act on besides "the correct flag was rejected".
        want_len = state.encoder.artifacts.get("want_len")
        first = ((good.stdout or "").strip().splitlines() or [""])[0][:48]
        # mode-aware: forward_check never asks for an inverse (there is none); a mismatch there
        # means F itself behaved differently than when the encoder ran it, not a derivation bug.
        hint = ("Your transform_code must be a PURE function of (in, n) with no state carried "
                "between calls — if it produced different output here than when the encoder ran "
                "it, that is the bug (fix transform_code; there is no inverse to write)."
               ) if state.mode == "forward_check" else (
                "Derive the number of bytes from the embedded array itself (e.g. sizeof), never a "
                "fixed constant, and invert the encoder transform byte-for-byte.")
        diag = (f" | diagnostic from feeding the correct flag: the flag is {len(state.flag)} bytes and the "
                f"embedded array has {want_len} bytes (one encoded byte per flag byte); the program printed "
                f"{first!r} instead of the marker {marker!r}. {hint}")
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
    "recover-accepts-wrong": "challenge", "recover-degenerate-transform": "challenge",
    "challenge-crash": "challenge",
    "challenge-bad-exit": "challenge", "too-trivial": "challenge", "output-truncated": "challenge",
    "solver-error": "solver", "solver-mismatch": "solver", "solver-hardcoded": "solver",
    "solver-evidence-fabricated": "solver",
    "swap-solver-error": "solver", "swap-encoder": "encoder", "swap-identical": "encoder",
    "swap-compile": "challenge",
    # forward_check-only contract violations (checked WITHOUT compiling — see _resolve_fc)
    "fc-contract-encoder": "encoder", "fc-contract-challenge": "challenge",
}


def _route_responsible(stage_name: str, which: str, state) -> str:
    """Resolve the component responsible for a failure. In forward_check mode, Stage C
    is harness-synthesized (see _FC_CHALLENGE_SKELETON) — the model never authors
    challenge_code there, so it can never be the fix target. Any challenge-stage
    failure routes to "encoder" instead: F is the only LLM-controlled input baked into
    the deterministic challenge, so a repair there is the only one that can be acted on."""
    resp = _RESPONSIBLE.get(stage_name, which)
    if state.mode == "forward_check" and resp == "challenge":
        return "encoder"
    return resp


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
    mode: str = "inverse",
) -> g.GenerationReport:
    """Agentic test-driven generation of a `recover` challenge. Returns a
    GenerationReport; a challenge is successful ONLY if the unchanged Stage-E gate
    passes. Never raises for a model/verification failure. Terminal failures are
    recorded under a classified category: 'budget-exhausted', 'infra' (Ollama/HTTP/
    connection/sandbox), 'model-invalid' (unusable model reply), or 'verifier-error'
    (an unexpected fault inside a check). `caller` is the LLM hook (tests inject
    replay). Everything is logged to a local, never-published transcript.

    `mode`: "inverse" (default; unchanged) or "forward_check" (experimental — see module
    docstring). Only applies to a FRESH run; resuming an existing run keeps its saved mode."""
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
                             mode=mode, path=str(statepath))
    state.save()

    report = g.GenerationReport(name=name, model=model, directory=str(base_dir), success=False)
    report.telemetry = {"workflow": "agentic", "stages": [], "repairs": {}}  # type: ignore[attr-defined]

    def apply_fix(which: str, code: str, reason: str, transform_code: Optional[str] = None) -> bool:
        c = state.comp(which)
        if c.repairs >= budget.max_repairs_per_stage:
            report.attempts.append(g.AttemptResult(0, False, "max-repairs", f"{which}: {reason}"[:300]))
            state.log({"type": "revision", "component": which, "applied": False,
                       "note": "max-repairs", "reason": str(reason)[:600], "repairs": c.repairs})
            return False
        state.set_code(which, code)
        # forward_check: commit the new F ATOMICALLY with the code that embeds it, and only
        # on this accepted path — never on a rejected/no-progress/exhausted candidate, so
        # transform_code can never drift out of sync with the actually-stored encoder.code.
        if transform_code is not None:
            state.transform_code = transform_code
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
        if which == "challenge" and state.mode == "forward_check":
            # Harness-synthesized every time this stage is (re)visited, so it always
            # embeds the CURRENT state.transform_code — never stale after an F repair.
            resolved, fc_err, _ = _resolve_fc("challenge", state, _FC_CHALLENGE_SKELETON, {})
            if fc_err is not None:   # defensive: the fixed skeleton should never violate the contract
                state.log({"type": "gate", "component": "challenge", "stage": "fc-contract-challenge",
                           "ok": False, "detail": fc_err[:600]})
                return False, "fc-contract-challenge", fc_err
            if _normalize(resolved) != _normalize(state.challenge.code):
                state.set_code("challenge", resolved)
            state.success_marker = FC_SUCCESS_MARKER
            state.save()
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
        """Obtain a CHANGED, CONTRACT-VALID fix for a known-responsible component. Pre-checks
        the repair budget so a call that could no longer be applied is never issued. An
        identical reply records `repair-no-progress`; a forward_check contract violation
        (missing/duplicated placeholder, empty transform_code — checked WITHOUT compiling)
        records `repair-contract-violation`; both escalate feedback and retry WITHOUT
        recompiling, bounded by the same repair counter. Returns (newcode, marker,
        transform_code) once a valid, changed code is produced, or None when the
        component's repair budget is exhausted. Applying the returned code (and committing
        transform_code) is the caller's job — fetch_fix never mutates state.transform_code."""
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
            raw = str(payload.get(f"{responsible}_code", c.code))
            newcode, fc_err, cand_tc = _resolve_fc(responsible, state, raw, payload)
            # Reverse-Engineering-First: a solver repair must ALSO re-ground its evidence —
            # independent of _resolve_fc (a no-op for "solver"), never gated by forward_check.
            if fc_err is None and responsible == "solver":
                fc_err = _check_solver_evidence(payload, state)
            marker = (str(payload["success_marker"]).strip()
                      if responsible == "challenge" and payload.get("success_marker") else None)
            if fc_err is None and _normalize(newcode) != _normalize(c.code):   # valid + real change
                escalate.pop(responsible, None)
                if responsible == "solver":
                    _log_solver_evidence(state, report, payload, where="repair")
                return newcode, marker, cand_tc
            # either a contract/evidence violation or an identical reply: record, escalate,
            # retry — bounded by the SAME repair counter, never recompiled.
            c.repairs += 1
            if fc_err is not None:
                stage = "repair-evidence-fabricated" if responsible == "solver" else "repair-contract-violation"
                report.attempts.append(g.AttemptResult(0, False, stage,
                                                       f"{responsible}: {fc_err}"[:300]))
                state.log({"type": stage, "component": responsible,
                           "repairs": c.repairs, "detail": fc_err[:600]})
                escalate[responsible] = "\n\nNOTE: " + fc_err
            else:
                np = report.telemetry.setdefault("no_progress", {})     # type: ignore[union-attr]
                np[responsible] = np.get(responsible, 0) + 1
                report.attempts.append(g.AttemptResult(0, False, "repair-no-progress",
                                                       f"{responsible}: identical reply; escalating feedback"[:300]))
                state.log({"type": "repair-no-progress", "component": responsible,
                           "repairs": c.repairs, "detail": detail[:600]})
                escalate[responsible] = ("\n\nNOTE: your previous reply was byte-for-byte IDENTICAL and fixed "
                                         "nothing — do not resend the same code. Name the ONE concrete cause of "
                                         "the failure and change exactly that part.")
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
            fc_gen_err: Optional[str] = None   # set below if the FIRST reply violates the contract
            evidence_err: Optional[str] = None  # set below if a FIRST solver reply fabricates evidence
            fc = state.mode == "forward_check"
            if which == "challenge" and fc:
                pass   # no LLM call: run_gate() deterministically (re)synthesizes it from F
            elif not comp.code:  # first time we need this component
                key = f"{which}_code"
                gen_prompt = ({"encoder": _encoder_prompt_fc if fc else _encoder_prompt,
                              "challenge": _challenge_prompt,   # only reached in 'inverse' mode
                              "solver": _solver_prompt}[which])(state)
                payload = _ask(caller, model, budget, state, report, which, "generate", gen_prompt)
                resolved, fc_gen_err, cand_tc = _resolve_fc(which, state, str(payload.get(key, "")), payload)
                if fc_gen_err is None and which == "solver":
                    # Reverse-Engineering-First: reject the FIRST solver reply too if its cited
                    # evidence is fabricated — independent of forward_check, never gated by it.
                    evidence_err = _check_solver_evidence(payload, state)
                if fc_gen_err is None and evidence_err is None:
                    comp.code = resolved
                    if which == "encoder" and cand_tc is not None:
                        state.transform_code = cand_tc        # first-ever F: nothing to desync yet
                    if which == "challenge" and payload.get("success_marker"):
                        state.success_marker = str(payload["success_marker"]).strip()
                    comp.status = "pending"
                    if which == "solver":
                        _log_solver_evidence(state, report, payload, where="generate")
                state.save()
            if fc_gen_err is not None:
                # Contract violation on the FIRST reply: never compiled — go straight to the
                # SAME repair machinery a gate failure would use (never trust the compiler
                # to catch a missing/duplicated placeholder).
                ok, stage_name, detail = False, f"fc-contract-{which}", fc_gen_err
            elif evidence_err is not None:
                ok, stage_name, detail = False, "solver-evidence-fabricated", evidence_err
            else:
                ok, stage_name, detail = run_gate(which)
            report.attempts.append(g.AttemptResult(0, ok, stage_name, detail[:300]))
            report.telemetry["stages"].append({which: stage_name})  # type: ignore[index]
            if ok:
                state.mark_verified(which, stage_name)
                state.save()
                continue
            # failed -> repair. Every stage maps to a responsible component; the solver's
            # repair prompt + feedback are walled (player-only), so no private info leaks to it.
            responsible = _route_responsible(stage_name, which, state)
            got = fetch_fix(responsible, detail)
            if got is None:
                state.save()
                return report  # repair budget exhausted for this component
            newcode, marker, cand_tc = got
            if marker:
                state.success_marker = marker
            if not apply_fix(responsible, newcode, detail[:300], transform_code=cand_tc):
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
            responsible = _route_responsible(e.stage, "challenge", state)
            # Stage E's authoritative detail may reveal the flag on a solver failure; when the
            # solver is the culprit, hand it only WALLED public feedback (never e.detail directly).
            fb = _solver_feedback(e.stage, "", e.detail or "", state.flag) if responsible == "solver" else e.detail
            got = fetch_fix(responsible, fb)
            if got is None:
                break
            newcode, marker, cand_tc = got
            if marker:
                state.success_marker = marker
            if not apply_fix(responsible, newcode, fb[:300], transform_code=cand_tc):
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
                    resp = _route_responsible(sn, which, state)
                    got = fetch_fix(resp, det)
                    if got is None:
                        state.save()
                        return report
                    nc, marker, cand_tc = got
                    if marker:
                        state.success_marker = marker
                    if not apply_fix(resp, nc, det[:300], transform_code=cand_tc):
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
