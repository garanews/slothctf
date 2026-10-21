from __future__ import annotations

import json
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


def _compile_binary(source_path: Path, binary_path: Path, compiler: str) -> tuple[bool, str, Path]:
    """Compile `source_path`. Returns (ok, stderr, actual_binary_path).

    MinGW gcc appends `.exe` when the output name has no extension, so we probe
    for the real artifact after a successful build instead of assuming the name.
    """
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [compiler, "-O2", "-s", "-o", str(binary_path), str(source_path)],
        capture_output=True,
        text=True,
    )
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
        timeout=timeout,
    )
    return result.returncode == 0, result.stdout, result.stderr.strip()


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
    the compiled binary — i.e. a `strings` run would give the answer away."""
    try:
        blob = binary_path.read_bytes()
    except OSError:
        return False
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
            "options": {"temperature": 0.7, "num_ctx": 4096},
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


def _build_generation_prompt(
    user_prompt: str,
    difficulty: str,
    category: str | None,
    feedback: str | None,
    target_platform: str | None = None,
) -> str:
    parts = [_SCHEMA_RULES, ""]
    parts.append(
        "BUILD TARGET — both C programs are compiled and run on: "
        f"{target_platform or _default_platform()} "
        "They MUST compile and run there, so do not call APIs that platform lacks."
    )
    parts.append(f"Target difficulty: {difficulty}.")
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
    src_dir = base_dir / "src"
    bin_dir = base_dir / "bin"
    private_dir = base_dir / "organizer"
    src_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    (src_dir / "challenge.c").write_text(data.get("challenge_code", ""), encoding="utf-8")
    # The encoder is the organizer's tool, kept out of the player-facing src/.
    (private_dir / "encoder.c").write_text(data.get("encoder_code", ""), encoding="utf-8")
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


def _normalize_challenge(data: dict[str, Any], difficulty: str, category: str | None) -> dict[str, Any]:
    """Fill defaults and normalize the flag so downstream writes never KeyError."""
    data = dict(data)
    data.setdefault("title", "SlothCTF Challenge")
    data.setdefault("category", category or "Reverse Engineering")
    data.setdefault("difficulty", difficulty)
    data.setdefault("description", "Reverse the binary and recover the flag.")
    kind = str(data.get("challenge_kind", "decode")).strip().lower()
    data["challenge_kind"] = "check" if kind == "check" else "decode"
    data.setdefault("success_marker", "Access Granted")
    # README / solution are rendered locally on success, not asked of the model.
    data.setdefault("readme", f"# {data['title']}\n\n{data['description']}\n")
    data.setdefault("solution", "See organizer/encoder.c for how the flag was hidden.")
    if "encoder_code" in data:
        data["encoder_code"] = _strip_code_fences(str(data["encoder_code"]))
    if "challenge_code" in data:
        data["challenge_code"] = _strip_code_fences(str(data["challenge_code"]))
    data["flag"] = _normalize_flag(str(data.get("flag", "")))
    return data


def _render_readme(data: dict[str, Any]) -> str:
    """Player-facing README, rendered locally (no LLM call) once the challenge
    is verified."""
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


def _verify_challenge(base_dir: Path, data: dict[str, Any], compiler: str) -> AttemptResult:
    """Build and self-check a challenge using the model's own encoder.

    Pipeline: compile+run the encoder on the flag to get the exact encoded
    bytes (the machine does the arithmetic the LLM is bad at), inject them into
    challenge_code, compile+run the challenge, and confirm it prints the flag
    and does not leak it. Each failure returns actionable feedback for the model.
    """
    expected = data["flag"]
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

    # 6. The plaintext secret must not be readable in the binary.
    if _flag_leaks_in_binary(challenge_bin, expected):
        return AttemptResult(
            0, False, "flag-leak",
            "The plaintext secret is readable inside the challenge binary. Only the "
            "encoded/target array must be embedded; do not keep the secret as a literal.",
        )

    return AttemptResult(0, True, "passed", f"{kind} verified: {expected}")


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

    base_dir = Path("generated") / name
    base_dir.mkdir(parents=True, exist_ok=True)

    category = challenge_type if challenge_type and challenge_type not in {"reverse", "surprise"} else None
    user_prompt = (context.strip() + "\n" + prompt.strip()).strip() if context else prompt
    if surprise:
        category = random.choice(SURPRISE_CATEGORIES)
        user_prompt = f"Surprise me with a {difficulty} challenge: {category}."

    compiler = _find_compiler()
    if compiler is None:
        raise GenerationError(
            "No C compiler found. Install one (e.g. `python main.py install-toolchain`) "
            "and ensure gcc is on PATH — the self-verifying loop needs to compile."
        )

    ask = caller or (lambda p, m: _call_ollama(p, m, base_url=ollama_url))

    report = GenerationReport(name=name, model=model, directory=str(base_dir), success=False)
    feedback: str | None = None

    for attempt in range(1, max_attempts + 1):
        full_prompt = _build_generation_prompt(user_prompt, difficulty, category, feedback, target_platform)
        try:
            raw = ask(full_prompt, model)
        except GenerationError as exc:
            report.attempts.append(AttemptResult(attempt, False, "model-error", str(exc)))
            feedback = str(exc)
            continue

        if "encoder_code" not in raw or "challenge_code" not in raw:
            missing = [f for f in ("encoder_code", "challenge_code") if f not in raw]
            feedback = f"Your JSON was missing required field(s): {', '.join(missing)}."
            report.attempts.append(AttemptResult(attempt, False, "bad-schema", feedback))
            continue

        data = _normalize_challenge(raw, difficulty, category)

        result = _verify_challenge(base_dir, data, compiler)
        result.index = attempt
        report.attempts.append(result)
        if result.ok:
            # Only now that the challenge is valid do we spend effort on the
            # prose, rendered locally so it costs no extra model time.
            data["readme"] = _render_readme(data)
            data["solution"] = _render_solution(data)
            _write_challenge_files(base_dir, data)
            report.success = True
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
