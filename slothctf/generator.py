from __future__ import annotations

import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib import request

# Base URL of the Ollama server. Defaults to localhost but can point at a
# remote machine (e.g. a beefy workstation reachable over WireGuard) via the
# SLOTHCTF_OLLAMA_URL env var or the --ollama-url CLI flag.
DEFAULT_OLLAMA_URL = os.environ.get("SLOTHCTF_OLLAMA_URL", "http://localhost:11434")


def _generate_endpoint(base_url: str) -> str:
    """Turn a base URL (or a full endpoint) into the /api/generate endpoint."""
    base = (base_url or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    if base.endswith("/api/generate"):
        return base
    return base + "/api/generate"


OLLAMA_ENDPOINT = _generate_endpoint(DEFAULT_OLLAMA_URL)
DEFAULT_PROMPT = "Create an easy reverse engineering challenge with XOR encryption"
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_MAX_ATTEMPTS = 5

# The JSON schema we require the model to return. The whole project lives or
# dies on the model filling these fields with *consistent* content: the flag
# declared in `flag` must be the one the C program hides and the one
# `solver_code` recovers. We deliberately do NOT ask the model for the long
# prose (README / solution) here — generating it on every repair round wastes
# tokens and time. It is rendered locally once the challenge validates.
REQUIRED_FIELDS = (
    "title",
    "category",
    "difficulty",
    "description",
    "flag",
    "encoder_code",
    "challenge_code",
)

# Token the model leaves in challenge_code where the machine-computed encoded
# byte array is injected. Keeps the model out of the byte arithmetic entirely.
ENCODED_PLACEHOLDER = "/*__ENCODED_ARRAY__*/"

SURPRISE_CATEGORIES = (
    "reverse engineering with a custom byte transformation",
    "reverse engineering with a password check hidden from `strings`",
    "cryptography with a small home-made cipher",
    "reverse engineering with a short bytecode interpreter",
    "reverse engineering with arithmetic constraints on the input",
)


class GenerationError(RuntimeError):
    """Raised when no valid challenge could be produced."""


@dataclass
class AttemptResult:
    index: int
    ok: bool
    stage: str  # which check failed (or "passed")
    detail: str = ""


@dataclass
class GenerationReport:
    name: str
    model: str
    directory: str
    success: bool
    attempts: list[AttemptResult] = field(default_factory=list)

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)

    def as_text(self) -> str:
        lines = [
            f"Challenge: {self.name}",
            f"Model: {self.model}",
            f"Directory: {self.directory}",
            f"Result: {'VALID' if self.success else 'FAILED'} "
            f"(after {self.attempts_used} attempt(s))",
            "",
        ]
        for att in self.attempts:
            mark = "ok" if att.ok else "fail"
            line = f"  attempt {att.index}: [{mark}] {att.stage}"
            if att.detail:
                line += f" — {att.detail}"
            lines.append(line)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Toolchain helpers
# --------------------------------------------------------------------------- #
def _find_compiler() -> str | None:
    for candidate in ("gcc", "clang", "x86_64-w64-mingw32-gcc", "gcc.exe"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


_COMPILE_TIMEOUT = 60.0  # model-written C must compile fast; a hang is a failure, not a wait


def _compile_binary(source_path: Path, binary_path: Path, compiler: str) -> tuple[bool, str, Path]:
    """Compile `source_path`. Returns (ok, stderr, actual_binary_path).

    MinGW gcc appends `.exe` when the output name has no extension, so we probe
    for the real artifact after a successful build instead of assuming the name.
    """
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [compiler, "-O2", "-s", "-o", str(binary_path), str(source_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_COMPILE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return (False,
                f"compilation timed out after {_COMPILE_TIMEOUT:.0f}s "
                "(possible pathological source from the model)",
                binary_path)
    produced = binary_path
    if not produced.exists():
        exe = binary_path.with_suffix(".exe")
        if exe.exists():
            produced = exe
    return result.returncode == 0, result.stderr, produced


def _run_binary(
    binary_path: Path,
    args: list[str] | None = None,
    stdin_text: str | None = None,
    timeout: float = 15.0,
) -> tuple[bool, str, str]:
    """Run a compiled binary and capture its stdout/stderr, optionally feeding
    `stdin_text` on standard input (for input-validation challenges)."""
    result = subprocess.run(
        [str(binary_path), *(args or [])],
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    # A challenge binary may print arbitrary bytes (especially a non-invertible
    # transform on a random round-trip input). Decode them with replacement so a
    # wrong decode is detected as a mismatch, never crashes the run.
    return result.returncode == 0, result.stdout or "", result.stderr.strip()


# --------------------------------------------------------------------------- #
# Flag handling
# --------------------------------------------------------------------------- #
def _normalize_flag(flag: str) -> str:
    cleaned = (flag or "").strip().strip("\"'`")
    if cleaned.startswith("sloth{") and cleaned.endswith("}"):
        return cleaned
    inner = re.sub(r"[^a-zA-Z0-9_\- ]+", "", cleaned).strip()
    inner = re.sub(r"\s+", "_", inner).lower()
    if not inner:
        return "sloth{slow_and_steady}"
    return f"sloth{{{inner}}}"


def _flag_leaks_in_binary(binary_path: Path, flag: str) -> bool:
    """True if the plaintext flag (or its inner secret) is trivially visible in
    the compiled binary — i.e. a `strings` run would give the answer away.

    Raises OSError if the binary can't be read: a security check that could not
    run must never be reported as passed (fail closed)."""
    blob = binary_path.read_bytes()
    candidates = {flag}
    m = re.match(r"sloth\{(.*)\}$", flag)
    if m and m.group(1):
        candidates.add(m.group(1))
    for needle in candidates:
        if len(needle) >= 4 and needle.encode("utf-8") in blob:
            return True
    return False


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
def _coerce_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response that may wrap it in prose or
    fences."""
    text = text.strip()
    if text.startswith("```"):
        inner = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if inner:
            text = inner.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def _call_ollama(
    prompt: str,
    model: str,
    timeout: float = 600.0,
    retries: int = 2,
    base_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Call a local Ollama model and return the parsed JSON challenge.

    Uses streaming so a slow CPU keeps the socket alive, asks Ollama to
    constrain the output to JSON so Gemma does not wrap it in prose, and keeps
    the model warm (`keep_alive`) so repair rounds do not pay the reload cost.
    Transient failures (e.g. a socket timeout while the model reloads) are
    retried in place so they do not burn an attempt from the repair budget.
    """
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "format": "json",
            "keep_alive": "10m",
            # Our prompt + expected output fit comfortably in 4k tokens; a
            # smaller context means a smaller KV cache and faster first token.
            "options": {"temperature": temperature, "num_ctx": 4096},
        }
    ).encode("utf-8")

    endpoint = _generate_endpoint(base_url)
    last_exc: Exception | None = None
    for _ in range(retries + 1):
        req = request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks: list[str] = []
        try:
            with request.urlopen(req, timeout=timeout) as response:
                for raw in response:
                    raw = raw.strip()
                    if not raw:
                        continue
                    piece = json.loads(raw.decode("utf-8"))
                    chunks.append(piece.get("response", ""))
                    if piece.get("done"):
                        break
        except Exception as exc:  # network / model errors — retry
            last_exc = exc
            continue

        text = "".join(chunks).strip()
        if not text:
            last_exc = GenerationError("Ollama returned an empty response.")
            continue

        try:
            return _coerce_json(text)
        except Exception:
            last_exc = GenerationError("Could not parse a JSON challenge from the model output.")
            continue

    raise GenerationError(f"Ollama unavailable: {last_exc}")


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
_SCHEMA_RULES = """\
You are SlothCTF, a generator of small, self-contained Capture-The-Flag reverse
engineering challenges written in C. You invent EVERYTHING — the secret, the
scheme, and all the C code. There are no templates.

You never compute obfuscated/encrypted bytes in your head (you get the
arithmetic wrong). Instead you write an ENCODER in C and the machine runs it on
the secret to get the exact bytes, then injects them into the challenge. This
guarantees consistency.

Return ONE JSON object (no prose, no markdown fences) with EXACTLY these keys:
  "title"          : short challenge name
  "category"       : e.g. "Reverse Engineering" or "Cryptography"
  "difficulty"     : "easy", "medium" or "hard"
  "description"    : 1-2 sentences the player sees (do NOT reveal the secret or
                     the method)
  "flag"           : the SECRET, formatted as sloth{...}. For a "decode"
                     challenge it is what the binary prints; for a "check"
                     challenge it is the exact input the player must type to win.
  "challenge_kind" : "decode" or "check" (see below)
  "success_marker" : (check challenges only) the exact text the program prints
                     when the input is correct, e.g. "Access Granted"
  "encoder_code"   : a COMPLETE C program (organizer tool, players never see it)
  "challenge_code" : a COMPLETE C program given to the player

CHALLENGE KINDS:
  - "decode": the challenge decodes an embedded array at runtime and PRINTS the
    flag to stdout.
  - "check": the challenge READS one line from stdin and prints success_marker
    only if the input equals the secret (the flag), otherwise prints something
    else. Use this for password / input-validation challenges (state machines,
    hash checks, ciphers compared to a target, anti-debug gates, etc.).

HARD REQUIREMENTS (breaking any is rejected and sent back to you to fix):

1. encoder_code:
   - compiles with: gcc -O2 -s -o encoder encoder.c
   - reads the secret (the flag) from argv[1];
   - applies YOUR chosen transform (xor, add, sbox, hash, cipher, …) and prints
     the resulting bytes to stdout as a comma-separated list of SINGLE-BYTE 0xNN
     hex tokens (each 0x00..0xff) and NOTHING else, e.g.:  0x31,0x2e,0x2d,
   - IMPORTANT: print ONE token PER BYTE. If your transform yields integers
     (e.g. a 32-bit FNV hash, or TEA/RC4 32-bit words), split each integer into
     its individual bytes and print them as separate 0xNN tokens (e.g. a 32-bit
     value = 4 tokens). Never print a wide value like 0x1a2b3c4d as one token.
   - process the WHOLE secret; do NOT require a fixed input length. If your
     scheme uses fixed-size blocks (e.g. TEA's 8-byte block), pad the secret to
     a multiple of the block size and process every block.
   - these are the exact bytes the challenge must embed (the decoded flag for a
     "decode" challenge, or the comparison target for a "check" challenge).

2. challenge_code:
   - compiles with: gcc -O2 -s -o challenge challenge.c
   - contains the EXACT token %(ph)s where the embedded byte array is
     initialised, e.g.:  static unsigned char data[] = { %(ph)s };
     (the machine substitutes the encoder's output there — put NO numbers there
     yourself);
   - "decode": applies the INVERSE transform to `data` and prints EXACTLY the
     flag to stdout;
   - "check": reads one line from stdin, STRIPS the trailing newline, applies the
     SAME transform you used in the encoder to the input, serialises the result
     to bytes THE SAME WAY the encoder did, and compares those bytes to `data`
     byte-by-byte; prints success_marker iff every byte matches (otherwise a
     failure message). `data` is an `unsigned char[]`; if your transform yields
     integers, compare their bytes, never cast `data` to a wider type;
   - the plaintext secret must NOT appear as a readable literal in
     challenge_code — only the encoded/target array is embedded.

3. Both programs: standard C only. The encoder transform and the challenge's
   inverse/compare MUST be consistent so that the real secret round-trips.

4. CONSISTENCY RULE (critical for hashes/ciphers like FNV-1a, TEA, RC4, S-box):
   put the core transform in a named function (e.g. `uint32_t transform(const
   unsigned char *in, int len)` or `void transform(unsigned char *buf, int
   len)`) and copy that function BYTE-FOR-BYTE IDENTICALLY into BOTH
   encoder_code and challenge_code — same signature, same body, same constants.
   The two programs must share the exact same transform so results cannot
   diverge by even one bit. Ensure every loop terminates and every buffer is
   bounded (no infinite loops, no out-of-bounds writes).

Think carefully so the encoder and the challenge agree perfectly.""" % {"ph": ENCODED_PLACEHOLDER}


def _default_platform() -> str:
    """Describe the platform where the generated C is compiled and run, so the
    model avoids OS APIs that are not available there."""
    system = platform.system()
    if system == "Windows":
        return (
            "Windows, compiled with MinGW gcc. POSIX-only APIs are NOT available "
            "(no ptrace, no <sys/ptrace.h>, no dlopen). MSVC-only extensions are "
            "NOT available either (no __try/__except SEH). For anti-debug use "
            "Windows APIs (e.g. IsDebuggerPresent from <windows.h>) or portable "
            "tricks; otherwise prefer plain portable standard C."
        )
    if system in ("Linux", "Darwin"):
        return (
            f"{system} with gcc/clang. POSIX APIs are available (e.g. ptrace, "
            "dlopen). Windows-only APIs (<windows.h>, IsDebuggerPresent, "
            "LoadLibraryA) are NOT available. Prefer portable standard C."
        )
    return f"{system or 'a POSIX system'} with a C compiler; prefer portable standard C."


_TASK_SPEC = """\
Each entry of "tasks" is an object with:
  "question"       : one line; the exact thing to find. It MUST have a SINGLE
                     unambiguous answer — never ask something with a tie (e.g. two
                     usernames with the same count).
  "type"           : "extractive" if the answer appears VERBATIM in the log,
                     else "computed" (a count, a number of distinct values, a max…)
  "answer"         : the exact answer string
  "evidence_line"  : (extractive only) the 1-based line number whose text contains
                     the answer
  "evidence_text"  : (extractive only) that line's exact text
  "extractor_code" : a Python 3 script, run as `python extractor.py <logfile>`,
                     that reads the log, computes THIS task's answer, and prints
                     ONLY it to stdout. Deterministic, standard library only, and
                     it MUST derive the answer from the log — never hardcode it."""

_SCHEMA_RULES_LOG = """\
You are SlothCTF generating a LOG ANALYSIS (forensics) challenge: a realistic
synthetic log plus SEVERAL questions about it. You never just assert an answer —
each is derived by an extractor the machine runs.

Return ONE JSON object (no prose, no fences) with EXACTLY these keys:
  "title"          : short challenge name
  "category"       : e.g. "Forensics" or "Log Analysis"
  "difficulty"     : "easy", "medium" or "hard"
  "description"    : 1-2 sentences the player sees (no spoiler)
  "generator_code" : a Python 3 script, run as `python generator.py` (NO arguments),
                     that PRINTS the entire realistic synthetic log (syslog / HTTP /
                     SSH / firewall / JSON-lines style) with plenty of plausible
                     noise to STDOUT. Deterministic (use a fixed random seed).
                     Do NOT open or write any file; just print. Standard library only.
  "tasks"          : a list of 3 to 6 task objects (see below)

%(tasks)s

CODE RULES: write NORMAL, properly indented multi-line Python — each statement on
its OWN line. Do NOT cram the whole program onto one line with semicolons (`def`
and `for` cannot be collapsed that way — it is a syntax error). The only rule
about newlines is inside a STRING LITERAL: use "\\n" there, never a real line
break. The generator PRINTS the log (opens/writes no file); each extractor reads
its log-file argument.

HARD REQUIREMENTS (breaking any is rejected): running the generator then each
task's extractor must reproduce EXACTLY that task's answer; each extractor derives
its answer from the log (no hardcoding); every question must be unambiguous.""" % {"tasks": _TASK_SPEC}


_SCHEMA_RULES_LOG_ANALYSIS = """\
You are SlothCTF turning a PROVIDED log (shown below with 1-based line numbers)
into SEVERAL forensics questions. Pick concrete, verifiable facts (attacker IP,
leaked token, suspicious username, a count, number of distinct values, a max…)
and turn each into a task. GROUND every claim in the log — never invent a value.

Return ONE JSON object (no prose, no fences) with EXACTLY these keys:
  "title"       : short challenge name
  "category"    : e.g. "Forensics" or "Log Analysis"
  "difficulty"  : "easy", "medium" or "hard"
  "description" : 1-2 sentences the player sees (no spoiler)
  "tasks"       : a list of 3 to 6 task objects (see below)

%(tasks)s

HARD REQUIREMENTS (breaking any is rejected): for extractive tasks the answer
appears verbatim at evidence_line; each extractor derives its answer from the log
(no hardcoding); every question must be unambiguous (no ties).""" % {"tasks": _TASK_SPEC}


def _number_log(text: str, max_lines: int = 400, max_chars: int = 16000) -> str:
    """Render a log with 1-based line numbers for the prompt, truncated if huge."""
    lines = text.splitlines()
    truncated = len(lines) > max_lines
    shown = lines[:max_lines]
    body = "\n".join(f"{i}: {ln}" for i, ln in enumerate(shown, 1))
    if len(body) > max_chars:
        body = body[:max_chars]
        truncated = True
    if truncated:
        body += "\n… (log truncated for the prompt; the full log is on disk)"
    return body


# Concrete per-level guidance so easy/medium/hard actually differ, per genre.
_DIFFICULTY_RUBRIC = {
    "reverse": {
        "easy": "a SINGLE reversible byte transform (e.g. XOR or add with one key); "
                "a player can solve it by reading the source or just running the binary.",
        "medium": "a named cipher/encoding or a multi-step check (e.g. an S-box, RC4/TEA, "
                  "a state machine); solving requires re-implementing the transform.",
        "hard": "layered and non-obvious (e.g. a cipher PLUS an anti-debug/anti-VM gate, a "
                "custom bytecode VM, or self-modifying code); solving requires real reversing.",
    },
    "log": {
        "easy": "the answer is found in a SINGLE line (one grep / one regex).",
        "medium": "the answer requires filtering and aggregating, or correlating two fields "
                  "across lines (a count, number of distinct values, a max).",
        "hard": "the answer requires multi-step correlation across many lines — time windows, "
                "joining related events, or decoding embedded/encoded data.",
    },
}


def _random_flag() -> str:
    """A non-guessable default flag used when the prompt doesn't specify one."""
    import random
    import string
    token = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(12))
    return f"sloth{{{token}}}"


def _build_generation_prompt(
    user_prompt: str,
    difficulty: str,
    category: str | None,
    feedback: str | None,
    target_platform: str | None = None,
    schema: str = _SCHEMA_RULES,
    default_flag: str | None = None,
) -> str:
    parts = [schema, ""]
    if default_flag:
        parts.append(
            "FLAG POLICY: if the challenge request below specifies a concrete value to use "
            "(a flag, a password, an XOR key, a target string, etc.), use EXACTLY that value. "
            f"Otherwise, set the flag to exactly {default_flag} — do NOT invent a guessable "
            "word like 'password' or 'easy_peasy'."
        )
    if schema is _SCHEMA_RULES:
        parts.append(
            "BUILD TARGET — both C programs are compiled and run on: "
            f"{target_platform or _default_platform()} "
            "They MUST compile and run there, so do not call APIs that platform lacks."
        )
    genre_key = "reverse" if schema is _SCHEMA_RULES else "log"
    rubric = _DIFFICULTY_RUBRIC[genre_key].get(difficulty.lower())
    parts.append(f"Target difficulty: {difficulty}." + (f" At this level, {rubric}" if rubric else ""))
    if category:
        parts.append(f"Category / flavour: {category}.")
    parts.append("")
    parts.append("Challenge request from the user:")
    parts.append(user_prompt.strip() or DEFAULT_PROMPT)
    if feedback:
        parts.append("")
        parts.append(
            "Your PREVIOUS attempt failed automated verification. Fix it and "
            "return a corrected JSON object. What went wrong:"
        )
        parts.append(feedback.strip())
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Verification (the self-checking loop)
# --------------------------------------------------------------------------- #
def _write_challenge_files(base_dir: Path, data: dict[str, Any]) -> None:
    """Write the artifacts. `data` is expected to hold the (possibly injected)
    challenge_code, the organizer-only encoder_code, and the rendered prose."""
    private_dir = base_dir / "organizer"
    private_dir.mkdir(parents=True, exist_ok=True)

    # Only write the artifacts the genre actually produced.
    if "challenge_code" in data:
        (base_dir / "src").mkdir(parents=True, exist_ok=True)
        (base_dir / "src" / "challenge.c").write_text(data["challenge_code"], encoding="utf-8")
    if "encoder_code" in data:  # organizer-only tool
        (private_dir / "encoder.c").write_text(data["encoder_code"], encoding="utf-8")
    if "generator_code" in data:
        (private_dir / "generator.py").write_text(data["generator_code"], encoding="utf-8")
    for stale in private_dir.glob("extractor_*.py"):  # drop extractors of removed tasks
        stale.unlink()
    for i, t in enumerate(data.get("tasks", []), 1):  # per-task extractors (log genre)
        (private_dir / f"extractor_{i}.py").write_text(t.get("extractor_code", ""), encoding="utf-8")

    (base_dir / "README.md").write_text(data["readme"], encoding="utf-8")
    (base_dir / "solution.md").write_text(data["solution"], encoding="utf-8")
    meta = {
        "title": data["title"],
        "category": data["category"],
        "difficulty": data["difficulty"],
        "description": data["description"],
        "flag": data["flag"],
        "challenge_kind": data.get("challenge_kind", "decode"),
    }
    if data.get("challenge_kind") == "check":
        meta["success_marker"] = data.get("success_marker", "Access Granted")
    if data.get("challenge_kind") == "log":
        kept = data.get("tasks", [])
        meta["tasks"] = [
            {k: t.get(k) for k in ("question", "answer", "type", "evidence_line") if k in t}
            for t in kept
        ]
        # The flag summary must reflect the tasks actually KEPT (after per-task
        # acceptance dropped the broken ones), not the count the model proposed.
        meta["flag"] = f"{len(kept)} task(s)"
    if data.get("generation"):
        meta["generation"] = data["generation"]
    (base_dir / "challenge.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences a model sometimes wraps around code even when
    asked for raw source, e.g. ```c ... ``` leaking into source_code."""
    s = (code or "").strip()
    m = re.match(r"^```[a-zA-Z0-9+]*\n(.*?)\n?```$", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    s = re.sub(r"^```[a-zA-Z0-9+]*[ \t]*\r?\n?", "", s)
    s = re.sub(r"\r?\n?```[ \t]*$", "", s)
    return s.strip()


def _clean_answer(ans: str) -> str:
    """For log challenges the flag is a plain answer (IP, token…), not sloth{}."""
    return (ans or "").strip().strip("\"'`").strip()


def _log_name(data: dict[str, Any]) -> str:
    return data.get("log_file", "challenge.log")


def _normalize_challenge(
    data: dict[str, Any], difficulty: str, category: str | None, genre: str = "reverse"
) -> dict[str, Any]:
    """Fill defaults and normalize fields so downstream writes never KeyError."""
    data = dict(data)
    data.setdefault("title", "SlothCTF Challenge")
    # The requested difficulty wins over the model's self-label, so the recorded
    # level always matches what was asked for.
    data["difficulty"] = difficulty
    data.setdefault("readme", f"# {data.get('title')}\n")

    if genre == "log":
        data.setdefault("category", category or "Forensics")
        data["challenge_kind"] = "log"
        data.setdefault("description", "Analyze the log and answer the questions.")
        data.setdefault("solution", "See organizer/ for the per-task extractors.")
        if "generator_code" in data:
            data["generator_code"] = _strip_code_fences(str(data["generator_code"]))
        tasks = data.get("tasks")
        if not isinstance(tasks, list):
            tasks = []
        for t in tasks:
            if isinstance(t, dict):
                t["answer"] = _clean_answer(str(t.get("answer", "")))
                if "extractor_code" in t:
                    t["extractor_code"] = _strip_code_fences(str(t["extractor_code"]))
        data["tasks"] = tasks
        # catalog/display summary; the real answers live per-task
        data["flag"] = f"{len(tasks)} task(s)"
        return data

    # reverse / crypto (C encoder pipeline)
    data.setdefault("category", category or "Reverse Engineering")
    data.setdefault("description", "Reverse the binary and recover the flag.")
    kind = str(data.get("challenge_kind", "decode")).strip().lower()
    data["challenge_kind"] = "check" if kind == "check" else "decode"
    data.setdefault("success_marker", "Access Granted")
    data.setdefault("solution", "See organizer/encoder.c for how the flag was hidden.")
    for k in ("encoder_code", "challenge_code"):
        if k in data:
            data[k] = _strip_code_fences(str(data[k]))
    data["flag"] = _normalize_flag(str(data.get("flag", "")))
    return data


def _render_readme(data: dict[str, Any]) -> str:
    """Player-facing README, rendered locally (no LLM call) once the challenge
    is verified."""
    if data.get("challenge_kind") == "log":
        qs = "\n".join(f"{i}. {t.get('question','')}" for i, t in enumerate(data.get("tasks", []), 1))
        return (
            f"# {data['title']}\n\n"
            f"**Category:** {data['category']}  \n"
            f"**Difficulty:** {data['difficulty']}\n\n"
            f"{data['description']}\n\n"
            "## Files\n\n"
            f"- `evidence/{_log_name(data)}` — the log to analyze\n\n"
            "## Questions\n\n"
            f"{qs}\n"
        )
    if data.get("challenge_kind") == "check":
        goal = (
            "## Goal\n\n"
            "Reverse the binary to find the secret input it accepts. Run it, type a "
            "candidate on standard input, and keep going until it accepts you.\n\n"
            "## Run\n\n"
            "```\n"
            "./bin/challenge\n"
            "```\n"
        )
    else:
        goal = (
            "## Goal\n\n"
            "Reverse the binary and recover the flag in the form `sloth{...}`.\n\n"
            "## Run\n\n"
            "```\n"
            "./bin/challenge\n"
            "```\n"
        )
    return (
        f"# {data['title']}\n\n"
        f"**Category:** {data['category']}  \n"
        f"**Difficulty:** {data['difficulty']}\n\n"
        f"{data['description']}\n\n"
        "## Files\n\n"
        "- `bin/challenge` — the binary to reverse (`.exe` on Windows)\n\n"
        f"{goal}"
    )


def _render_solution(data: dict[str, Any]) -> str:
    """Organizer-only walkthrough, rendered locally once the challenge is
    verified. Running the challenge binary prints the flag, and the encoder
    (organizer tool) documents exactly how the bytes were produced."""
    if data.get("challenge_kind") == "log":
        mode = "synthetic log" if "generator_code" in data else "provided log"
        parts = [
            f"# Solution — {data['title']}\n",
            f"Built from a {mode}. Each answer is produced by its own extractor, which "
            "the tool ran against the log; computed answers are verified by the "
            "extractor's deterministic output, extractive ones also by the cited line. "
            "No extractor reproduces its answer on an unrelated log.\n",
        ]
        for i, t in enumerate(data.get("tasks", []), 1):
            parts.append(f"## Task {i} — {t.get('question','')}\n")
            parts.append(f"**Answer:** `{t.get('answer','')}`  ·  type: `{t.get('type','extractive')}`")
            if str(t.get("type", "")).lower() == "extractive" and t.get("evidence_line"):
                parts.append(f"Evidence line {t.get('evidence_line')}: `{t.get('evidence_text','')}`")
            parts.append(
                f"```\npython organizer/extractor_{i}.py evidence/{_log_name(data)}\n```\n"
                f"```python\n{t.get('extractor_code','')}\n```\n"
            )
        return "\n".join(parts)
    if data.get("challenge_kind") == "check":
        recover = (
            f"**Secret input:** `{data['flag']}`\n\n"
            "This is an input-validation challenge. It was verified automatically: "
            "feeding the secret above on stdin makes the binary print "
            f"`{data.get('success_marker', '')}`, while a wrong input does not. A "
            "plain `strings` run does not reveal the secret.\n\n"
            "## Recover the secret\n\n"
            "```\n"
            "echo <secret> | ./bin/challenge\n"
            "```\n\n"
        )
    else:
        recover = (
            f"**Flag:** `{data['flag']}`\n\n"
            "This challenge was verified automatically: the encoder below was "
            "compiled and run on the flag to produce the embedded array, the "
            "challenge binary was compiled and run, and its output was confirmed to "
            "equal the flag. A plain `strings` run does not reveal the flag.\n\n"
            "## Recover the flag\n\n"
            "The challenge decodes and prints the flag when run:\n\n"
            "```\n"
            "./bin/challenge\n"
            "```\n\n"
        )
    return (
        f"# Solution — {data['title']}\n\n"
        f"{recover}"
        "## How the secret was embedded (`organizer/encoder.c`)\n\n"
        "```c\n"
        f"{data.get('encoder_code', '')}\n"
        "```\n"
    )


_HEX_LIST_RE = re.compile(r"0[xX][0-9a-fA-F]{1,2}")


def _parse_encoder_output(text: str) -> str | None:
    """The encoder prints the encoded bytes as a comma-separated list of 0xNN
    tokens. Pull those out and return a clean `0xNN, 0xNN, ...` body, or None if
    the output does not look like a byte list."""
    tokens = _HEX_LIST_RE.findall(text or "")
    if not tokens:
        return None
    return ", ".join(t.lower() for t in tokens)


def _verify_challenge(base_dir: Path, data: dict[str, Any], compiler: str,
                      roundtrip: bool = False, adversary: bool = False,
                      adversary_budget: tuple[int, float] | None = None) -> AttemptResult:
    """Build and self-check a challenge using the model's own encoder.

    Pipeline: compile+run the encoder on the flag to get the exact encoded
    bytes (the machine does the arithmetic the LLM is bad at), inject them into
    challenge_code, compile+run the challenge, and confirm it prints the flag
    and does not leak it. Each failure returns actionable feedback for the model.

    With `roundtrip`, after the instance passes we re-run the whole pipeline on
    several random inputs and require decode(encode(x)) == x for each — proving
    the transform is a true inverse in general, not just for this one flag.
    """
    expected = data["flag"]
    template = str(data.get("challenge_code", ""))  # pre-injection, for round-trip
    src_dir = base_dir / "src"
    bin_dir = base_dir / "bin"
    org_dir = base_dir / "organizer"
    for d in (src_dir, bin_dir, org_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Encoder must compile.
    encoder_src = org_dir / "encoder.c"
    encoder_src.write_text(data.get("encoder_code", ""), encoding="utf-8")
    ok, stderr, encoder_bin = _compile_binary(encoder_src, bin_dir / "encoder", compiler)
    if not ok:
        trimmed = "\n".join(stderr.splitlines()[:20])
        return AttemptResult(0, False, "encoder-compile", f"encoder.c failed to compile:\n{trimmed}")

    # 2. Run encoder on the flag to get the exact encoded bytes.
    try:
        ran, out, err = _run_binary(encoder_bin, [expected])
    except subprocess.TimeoutExpired:
        return AttemptResult(0, False, "encoder-timeout", "encoder did not finish within 15s.")
    if not ran:
        return AttemptResult(0, False, "encoder-crash", f"encoder crashed: {err[:200]}")
    array_body = _parse_encoder_output(out)
    if not array_body:
        return AttemptResult(
            0, False, "encoder-output",
            "The encoder must print the encoded flag as a comma-separated list of "
            f"0xNN hex bytes and nothing else. It printed: {out[:120]!r}",
        )

    # 3. Inject the bytes into the challenge at the placeholder.
    challenge_code = data.get("challenge_code", "")
    if ENCODED_PLACEHOLDER not in challenge_code:
        return AttemptResult(
            0, False, "missing-placeholder",
            f"challenge_code must contain the exact token {ENCODED_PLACEHOLDER} where "
            "the encoded byte array is initialised; it was not found.",
        )
    injected = challenge_code.replace(ENCODED_PLACEHOLDER, array_body)
    data["challenge_code"] = injected  # persist the final, injected source
    challenge_src = src_dir / "challenge.c"
    challenge_src.write_text(injected, encoding="utf-8")

    # 4. Challenge must compile.
    ok, stderr, challenge_bin = _compile_binary(challenge_src, bin_dir / "challenge", compiler)
    if not ok:
        trimmed = "\n".join(stderr.splitlines()[:20])
        return AttemptResult(0, False, "challenge-compile", f"challenge.c failed to compile:\n{trimmed}")

    # 5. Behaviour check — depends on the kind of challenge.
    kind = data.get("challenge_kind", "decode")
    if kind == "check":
        marker = str(data.get("success_marker", "")).strip()
        if not marker:
            return AttemptResult(0, False, "no-marker", "A 'check' challenge needs a non-empty success_marker.")
        # Correct secret must be accepted...
        try:
            _, good_out, _ = _run_binary(challenge_bin, stdin_text=expected + "\n")
            # ...and an obviously wrong input must be rejected.
            _, bad_out, _ = _run_binary(challenge_bin, stdin_text="definitely_wrong_" + expected + "\n")
        except subprocess.TimeoutExpired:
            return AttemptResult(0, False, "challenge-timeout", "challenge did not finish within 15s.")
        if marker not in good_out:
            return AttemptResult(
                0, False, "check-reject",
                f"feeding the correct secret {expected!r} on stdin did not print the "
                f"success marker {marker!r} (got {good_out.strip()[:120]!r}). The "
                "challenge's compare step must accept exactly the secret.",
            )
        if marker in bad_out:
            return AttemptResult(
                0, False, "check-accept-wrong",
                f"a wrong input also printed the success marker {marker!r}; the check "
                "must only accept the exact secret.",
            )
    else:  # decode
        try:
            ran, stdout, err = _run_binary(challenge_bin)
        except subprocess.TimeoutExpired:
            return AttemptResult(0, False, "challenge-timeout", "challenge did not finish within 15s.")
        if not ran:
            return AttemptResult(0, False, "challenge-crash", f"challenge crashed: {err[:200]}")
        if stdout.strip() != expected:
            return AttemptResult(
                0, False, "flag-mismatch",
                f"the challenge printed {stdout.strip()[:120]!r} but the flag is {expected!r}. "
                "Your challenge's decode step is not the exact inverse of the encoder's "
                "transform — fix one of them so decoding the encoded flag yields the flag.",
            )

    # 6. The plaintext secret must not be readable in the binary (fail closed:
    #    if we can't read the binary, the check didn't run → treat as a failure).
    try:
        leaked = _flag_leaks_in_binary(challenge_bin, expected)
    except OSError as exc:
        return AttemptResult(0, False, "leak-check-failed",
                             f"could not read the challenge binary for the leak check: {exc}")
    if leaked:
        return AttemptResult(
            0, False, "flag-leak",
            "The plaintext secret is readable inside the challenge binary. Only the "
            "encoded/target array must be embedded; do not keep the secret as a literal.",
        )

    # 7. Optional: prove the transform is a true inverse for MANY inputs, not
    #    just this flag — re-run encode→inject→compile→decode on random inputs.
    if roundtrip:
        rt = _roundtrip_check(base_dir, template, encoder_bin, compiler, kind,
                              str(data.get("success_marker", "")).strip())
        if rt is not None:
            return rt

    # 8. Optional: a reference attacker must not crack a medium/hard challenge.
    if adversary:
        adv = _adversary_check(challenge_bin, kind, str(data.get("success_marker", "")).strip(),
                               str(data.get("difficulty", "easy")), budget=adversary_budget)
        if adv is not None:
            return adv

    extras = ("+round-trip" if roundtrip else "") + (" +adversary" if adversary else "")
    return AttemptResult(0, True, "passed", f"{kind} verified: {expected}"
                         + (f" ({extras.strip()})" if extras.strip() else ""))


def _roundtrip_check(base_dir: Path, template: str, encoder_bin: Path, compiler: str,
                     kind: str, marker: str, n: int = 4) -> AttemptResult | None:
    """Re-run the pipeline on random inputs; return a failure AttemptResult if
    decode(encode(x)) != x for any, else None. Reuses the real encoder and the
    challenge template (no extra model code), so it tests the actual transform."""
    import random
    import string
    src_dir = base_dir / "src"
    bin_dir = base_dir / "bin"
    rt_src = src_dir / "_roundtrip.c"
    alphabet = string.ascii_letters + string.digits + "_-"
    try:
        for _ in range(n):
            inner = "".join(random.choice(alphabet) for _ in range(random.randint(3, 20)))
            x = f"sloth{{{inner}}}"
            try:
                ran, out, _ = _run_binary(encoder_bin, [x])
            except subprocess.TimeoutExpired:
                return AttemptResult(0, False, "roundtrip", "encoder timed out on a random input.")
            body = _parse_encoder_output(out)
            if not ran or not body:
                return AttemptResult(0, False, "roundtrip",
                                     f"encoder produced no bytes for random input {x!r}.")
            rt_src.write_text(template.replace(ENCODED_PLACEHOLDER, body), encoding="utf-8")
            ok, stderr, rt_bin = _compile_binary(rt_src, bin_dir / "_roundtrip", compiler)
            if not ok:
                return AttemptResult(0, False, "roundtrip",
                                     f"challenge did not compile for random input {x!r}:\n{stderr[:200]}")
            try:
                if kind == "check":
                    _, good, _ = _run_binary(rt_bin, stdin_text=x + "\n")
                    if marker not in (good or ""):
                        return AttemptResult(
                            0, False, "roundtrip",
                            f"random input {x!r} is NOT accepted — your transform is not a true "
                            "inverse for all inputs (it only works for the chosen flag). Check for "
                            "byte overflow / signed-unsigned / modulo edges.")
                else:
                    _, o, _ = _run_binary(rt_bin)
                    o = (o or "").strip()
                    if o != x:
                        return AttemptResult(
                            0, False, "roundtrip",
                            f"decode(encode({x!r})) = {o[:80]!r} != {x!r} — your transform "
                            "is not a true inverse for all inputs (byte overflow / signed-unsigned / "
                            "modulo edges), it only happens to work for the chosen flag.")
            except subprocess.TimeoutExpired:
                return AttemptResult(0, False, "roundtrip", f"challenge timed out on random input {x!r}.")
        return None
    finally:
        rt_src.unlink(missing_ok=True)
        for ext in ("", ".exe"):
            (bin_dir / f"_roundtrip{ext}").unlink(missing_ok=True)


_COMMON_SECRETS = [
    "password", "passw0rd", "p@ssw0rd", "pass", "admin", "administrator", "root",
    "secret", "letmein", "welcome", "qwerty", "123456", "12345678", "111111",
    "test", "guest", "login", "changeme", "default", "master", "flag", "ctf",
    "easy", "easy_peasy", "sloth", "hunter2", "dragon", "monkey", "superman",
    "iloveyou", "abc123", "football", "princess", "shadow", "michael", "666666",
    "whatever", "trustno1", "access", "securepass", "s3cr3t", "god", "money",
]


# Default attacker budget per difficulty: a HARD challenge must resist a stronger
# attacker than a MEDIUM one. The dictionary is kept deliberately small (a lazy
# first-guess attacker, not a determined reverser) so the gate doesn't reject
# challenges that are actually fine for humans. Override with --adversary-budget.
_ADVERSARY_BUDGET = {"medium": (800, 8.0), "hard": (2000, 20.0)}


def _adversary_check(challenge_bin: Path, kind: str, marker: str, difficulty: str,
                     budget: tuple[int, float] | None = None) -> AttemptResult | None:
    """A reference attacker with a budget. For `check` challenges it tries a
    dictionary of common secrets plus a short brute-force; if a medium/hard
    challenge is cracked within budget it is too easy. decode challenges solve by
    being run, so the gate does not apply to them. The budget scales with the
    declared difficulty unless an explicit one is given."""
    budget_attempts, budget_time = budget or _ADVERSARY_BUDGET.get(difficulty, (1000, 10.0))
    import itertools
    import string
    import time

    if kind != "check" or not marker:
        return None  # not applicable (decode = run-to-solve)

    def cracked_by(candidate: str) -> bool:
        try:
            _, out, _ = _run_binary(challenge_bin, stdin_text=candidate + "\n", timeout=3.0)
        except subprocess.TimeoutExpired:
            return False
        return marker in out

    # Candidates: common secrets (bare and wrapped) then short brute force.
    def candidates():
        for w in _COMMON_SECRETS:
            yield w
            yield f"sloth{{{w}}}"
        alpha = string.ascii_lowercase + string.digits
        for n in (1, 2):
            for combo in itertools.product(alpha, repeat=n):
                s = "".join(combo)
                yield f"sloth{{{s}}}"

    start = time.monotonic()
    tried = 0
    for cand in candidates():
        if tried >= budget_attempts or time.monotonic() - start > budget_time:
            break
        tried += 1
        if cracked_by(cand):
            if difficulty in ("medium", "hard"):
                return AttemptResult(
                    0, False, "too-easy",
                    f"a baseline attacker cracked the secret within budget (found "
                    f"{cand!r} in {tried} tries) — too weak for a '{difficulty}' challenge. "
                    "Use a longer, non-dictionary secret.")
            return None  # easy: being guessable is acceptable
    return None


def _run_python(script: Path, args: list[str], timeout: float = 20.0) -> tuple[bool, str, str]:
    """Run a generated Python script in isolated mode."""
    result = subprocess.run(
        ["python", "-I", str(script), *args],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode == 0, result.stdout, result.stderr.strip()


_DECOY_LOG = (
    "2026-01-01 00:00:00 sshd[1]: Accepted password for svc from 192.0.2.254 port 2200 ssh2\n"
    "2026-01-01 00:00:01 kernel: nothing of interest here\n"
    "2026-01-01 00:00:02 web: GET /health 200 from 192.0.2.254\n"
) * 4


def _check_one_task(task: dict[str, Any], ext: Path, log_path: Path, log_text: str,
                    lines: list[str], decoy: Path) -> tuple[bool, str]:
    """Verify a single task's extractor. Returns (ok, reason-if-failed)."""
    answer = _clean_answer(str(task.get("answer", "")))
    task["answer"] = answer
    etype = str(task.get("type", "extractive")).strip().lower()
    ext.write_text(_strip_code_fences(str(task.get("extractor_code", ""))), encoding="utf-8")
    try:
        ok, out, err = _run_python(ext, [str(log_path)])
        ok2, out2, _ = _run_python(ext, [str(log_path)])
    except subprocess.TimeoutExpired:
        return False, "extractor timed out"
    if not ok or not ok2:
        return False, f"extractor crashed: {err.splitlines()[-1][:160] if err.strip() else 'error'}"
    if out.strip() != out2.strip():
        return False, "extractor is not deterministic"
    if out.strip() != answer:
        return False, f"extractor printed {out.strip()[:80]!r} but declared answer is {answer!r}"
    if etype == "extractive":
        if answer not in log_text:
            return False, f"extractive answer {answer!r} not present verbatim in the log"
        try:
            ln = int(task.get("evidence_line", 0))
        except (TypeError, ValueError):
            ln = 0
        if not (1 <= ln <= len(lines)) or answer not in lines[ln - 1]:
            return False, f"evidence_line does not point to a line containing {answer!r}"
    try:
        _ok, decoy_out, _e = _run_python(ext, [str(decoy)])
        if decoy_out.strip() == answer:
            return False, "extractor prints the answer on an unrelated log (hardcoded)"
    except subprocess.TimeoutExpired:
        pass
    return True, ""


def _verify_tasks_against_log(base_dir: Path, data: dict[str, Any], log_path: Path) -> AttemptResult:
    """Check each task independently and KEEP the ones that verify, dropping any
    that fail. The challenge is valid if at least one task survives — a log
    naturally supports several questions and we ship only the proven ones."""
    org = base_dir / "organizer"
    org.mkdir(parents=True, exist_ok=True)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = log_text.splitlines()

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return AttemptResult(0, False, "no-tasks", "tasks must be a non-empty list.")

    decoy = base_dir / "evidence" / "_decoy.log"
    decoy.write_text(_DECOY_LOG, encoding="utf-8")
    kept: list[dict[str, Any]] = []
    reasons: list[str] = []
    try:
        for i, task in enumerate(tasks, 1):
            ok, reason = _check_one_task(task, org / f"_probe_{i}.py", log_path, log_text, lines, decoy)
            if ok:
                kept.append(task)
            else:
                reasons.append(f"task {i} ({task.get('question','')[:50]}): {reason}")
    finally:
        decoy.unlink(missing_ok=True)
        for p in org.glob("_probe_*.py"):
            p.unlink(missing_ok=True)

    data["tasks"] = kept
    if not kept:
        detail = "no task verified. " + " | ".join(reasons[:4])
        return AttemptResult(0, False, "no-valid-tasks", detail)
    dropped = len(tasks) - len(kept)
    note = f"{len(kept)}/{len(tasks)} task(s) verified" + (f", {dropped} dropped" if dropped else "")
    return AttemptResult(0, True, "passed", note)


def _verify_log_challenge(base_dir: Path, data: dict[str, Any], compiler: str | None = None) -> AttemptResult:
    """Synthesis: run the generator to produce the log, then verify all tasks."""
    org = base_dir / "organizer"
    ev = base_dir / "evidence"
    org.mkdir(parents=True, exist_ok=True)
    ev.mkdir(parents=True, exist_ok=True)

    gen = org / "generator.py"
    gen.write_text(_strip_code_fences(str(data.get("generator_code", ""))), encoding="utf-8")
    log_path = ev / f"{base_dir.name}.log"
    data["log_file"] = log_path.name
    try:
        # The generator PRINTS the log to stdout; we capture and save it. This
        # avoids the model hardcoding an output filename / ignoring argv.
        ok, out, err = _run_python(gen, [])
    except subprocess.TimeoutExpired:
        return AttemptResult(0, False, "generator-timeout", "generator.py did not finish within 20s.")
    if not ok:
        return AttemptResult(0, False, "generator-crash", f"generator.py failed:\n{err[:300]}")
    if not out.strip():
        return AttemptResult(0, False, "no-log", "generator.py printed nothing; it must print the log to stdout.")
    log_path.write_text(out, encoding="utf-8")
    return _verify_tasks_against_log(base_dir, data, log_path)


def _verify_log_analysis(base_dir: Path, data: dict[str, Any], provided_log: Path) -> AttemptResult:
    """Analysis: copy the provided log in, then verify all tasks against it."""
    ev = base_dir / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    try:
        log_text = provided_log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return AttemptResult(0, False, "no-input", f"could not read the provided log: {exc}")
    log_path = ev / provided_log.name
    log_path.write_text(log_text, encoding="utf-8")
    data["log_file"] = log_path.name
    return _verify_tasks_against_log(base_dir, data, log_path)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_challenge(
    name: str,
    prompt: str = DEFAULT_PROMPT,
    challenge_type: str | None = None,
    difficulty: str = "easy",
    input_path: str | Path | None = None,  # reserved for future file-based mode
    context: str = "",
    model: str = DEFAULT_MODEL,
    surprise: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    target_platform: str | None = None,
    genre: str = "reverse",
    roundtrip: bool = False,
    adversary: bool = False,
    adversary_budget: tuple[int, float] | None = None,
    *,
    caller: Callable[[str, str], dict[str, Any]] | None = None,
    return_report: bool = False,
) -> str | GenerationReport:
    """Generate and self-verify a CTF challenge with a local LLM.

    The model invents the obfuscation and writes two C programs: an `encoder`
    (organizer tool) and the `challenge` (player binary). We compile and run the
    encoder on the flag to get the exact hidden bytes — so the model never does
    the byte arithmetic it is bad at — inject them into the challenge, compile
    and run it, and confirm it prints the flag without leaking it. Any failure
    is fed back for up to `max_attempts` rounds. No templates: if the model
    cannot produce a valid challenge, we raise rather than fake one.

    `challenge_type` is kept for backwards compatibility and is now treated as a
    free-form category hint. `caller` is an injection point for tests.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if adversary_budget is not None:
        a, s = adversary_budget
        if not (isinstance(a, int) and a >= 1) or not (isinstance(s, (int, float)) and math.isfinite(s) and s > 0):
            raise ValueError("adversary_budget must be (attempts>=1, seconds>0 and finite)")

    base_dir = Path("generated") / name
    base_dir.mkdir(parents=True, exist_ok=True)

    category = challenge_type if challenge_type and challenge_type not in {"reverse", "surprise"} else None
    user_prompt = (context.strip() + "\n" + prompt.strip()).strip() if context else prompt
    if surprise:
        category = random.choice(SURPRISE_CATEGORIES)
        user_prompt = f"Surprise me with a {difficulty} challenge: {category}."

    # Record how this challenge was asked for (saved into challenge.json), so the
    # original prompt is always recoverable. For log-analysis mode the provided
    # log is appended to user_prompt below, so capture the clean prompt here.
    gen_meta: dict[str, Any] = {
        "prompt": user_prompt, "genre": genre, "model": model,
        "difficulty": difficulty, "roundtrip": bool(roundtrip),
        "adversary": bool(adversary), "surprise": bool(surprise),
    }
    if input_path:
        gen_meta["input"] = Path(input_path).name

    # Per-genre configuration: schema, required JSON fields, and verifier.
    temperature = 0.7
    provided_log = Path(input_path) if input_path else None
    if genre == "log" and provided_log is not None:
        # Analysis mode: derive a challenge from a PROVIDED log (deterministic).
        if not provided_log.is_file():
            raise GenerationError(f"--input log not found: {provided_log}")
        temperature = 0.0
        schema, required = _SCHEMA_RULES_LOG_ANALYSIS, ("tasks",)
        verify_fn = lambda b, d, c: _verify_log_analysis(b, d, provided_log)  # noqa: E731
        compiler = None
        numbered = _number_log(provided_log.read_text(encoding="utf-8", errors="replace"))
        user_prompt = (user_prompt.strip() + "\n\n" if user_prompt.strip() else "") + \
            "LOG TO TURN INTO A CHALLENGE (1-based line numbers):\n" + numbered
    elif genre == "log":
        schema, required, verify_fn = _SCHEMA_RULES_LOG, ("generator_code", "tasks"), _verify_log_challenge
        compiler = None  # log challenges are Python, no C compiler needed
    else:
        schema, required = _SCHEMA_RULES, ("encoder_code", "challenge_code")
        verify_fn = lambda b, d, c: _verify_challenge(  # noqa: E731
            b, d, c, roundtrip=roundtrip, adversary=adversary, adversary_budget=adversary_budget)
        compiler = _find_compiler()
        if compiler is None:
            raise GenerationError(
                "No C compiler found. Install one (e.g. `python main.py install-toolchain`) "
                "and ensure gcc is on PATH — the self-verifying loop needs to compile."
            )

    ask = caller or (lambda p, m: _call_ollama(p, m, base_url=ollama_url, temperature=temperature))
    # Reverse challenges get a random default flag (used only if the prompt does
    # not specify one), so unspecified secrets aren't guessable model clichés.
    default_flag = _random_flag() if genre == "reverse" else None

    report = GenerationReport(name=name, model=model, directory=str(base_dir), success=False)
    feedback: str | None = None

    for attempt in range(1, max_attempts + 1):
        full_prompt = _build_generation_prompt(user_prompt, difficulty, category, feedback, target_platform, schema, default_flag)
        try:
            raw = ask(full_prompt, model)
        except GenerationError as exc:
            report.attempts.append(AttemptResult(attempt, False, "model-error", str(exc)))
            feedback = str(exc)
            continue

        missing = [f for f in required if f not in raw]
        if missing:
            feedback = f"Your JSON was missing required field(s): {', '.join(missing)}."
            report.attempts.append(AttemptResult(attempt, False, "bad-schema", feedback))
            continue

        data = _normalize_challenge(raw, difficulty, category, genre)

        result = verify_fn(base_dir, data, compiler)
        result.index = attempt
        report.attempts.append(result)
        if result.ok:
            # Only now that the challenge is valid do we spend effort on the
            # prose, rendered locally so it costs no extra model time.
            data["readme"] = _render_readme(data)
            data["solution"] = _render_solution(data)
            data["generation"] = gen_meta
            _write_challenge_files(base_dir, data)
            report.success = True
            try:  # best-effort: index into the catalog, never fail generation
                from slothctf import catalog
                catalog.record(base_dir, valid=True)
            except Exception:
                pass
            break
        feedback = result.detail

    if not report.success:
        if return_report:
            return report
        raise GenerationError(
            f"Could not produce a valid challenge for {name!r} after "
            f"{report.attempts_used} attempt(s).\n\n{report.as_text()}"
        )

    return report if return_report else str(base_dir)
