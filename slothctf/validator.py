from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


_COMPILE_TIMEOUT = 60.0  # a hanging compile of model-written C is a failure, not a wait


@dataclass
class ValidationResult:
    """Structured outcome so callers can branch on success / stage instead of
    parsing a string (and set a proper exit code)."""
    success: bool
    stage: str
    message: str

    def __str__(self) -> str:  # keep existing string usages working
        return self.message


def _ok(stage: str, message: str) -> ValidationResult:
    return ValidationResult(True, stage, message)


def _fail(stage: str, message: str) -> ValidationResult:
    return ValidationResult(False, stage, "Validation failed: " + message)


def _flag_leaks(binary_path: Path, flag: str) -> bool:
    """True if the plaintext flag (or its inner secret) is readable in the binary
    — the same `strings`-style check the generator applies, re-run at validate.

    Raises OSError if the binary can't be read: a security check that cannot run
    must fail closed (the caller reports a validation failure), never silently
    pass."""
    blob = binary_path.read_bytes()
    needles = {flag}
    m = re.match(r"sloth\{(.*)\}$", flag)
    if m and m.group(1):
        needles.add(m.group(1))
    return any(len(n) >= 4 and n.encode("utf-8") in blob for n in needles)


def install_toolchain_windows(dry_run: bool = False) -> str:
    if os.name != "nt":
        return "This install helper is intended for Windows systems. On Linux/macOS, install gcc with your system package manager."

    if shutil.which("winget") is None:
        return (
            "GCC compiler not found and winget is unavailable. "
            "Install MSYS2 or MinGW-w64 manually, then add C:\\msys64\\mingw64\\bin to PATH."
        )

    if dry_run:
        return (
            "GCC compiler not found. Recommended installation path: "
            "winget install --id MSYS2.MSYS2 -e --source winget; "
            "then add C:\\msys64\\mingw64\\bin to PATH and reopen the terminal."
        )

    result = subprocess.run(
        ["winget", "install", "--id", "MSYS2.MSYS2", "-e", "--source", "winget"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return (
            "MSYS2 was installed successfully. Reopen the terminal and run: "
            "python main.py validate --dir <challenge-dir>"
        )

    return (
        "GCC compiler not found. Automatic installation via winget failed. "
        "Install MSYS2 or MinGW-w64 manually and add C:\\msys64\\mingw64\\bin to PATH."
        f"\nwinget output:\n{result.stderr or result.stdout}"
    )


def _find_compiler() -> str | None:
    for candidate in ("gcc", "clang", "x86_64-w64-mingw32-gcc", "gcc.exe"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _compile_binary(challenge_dir: Path) -> tuple[bool, str, str]:
    compiler = _find_compiler()
    if compiler is None:
        return (
            False,
            "",
            "GCC compiler not found. Run: python main.py install-toolchain --dry-run\n"
            "or install MinGW-w64 / MSYS2 and add C:\\msys64\\mingw64\\bin to PATH."
        )

    source_path = challenge_dir / "src" / "challenge.c"
    binary_path = challenge_dir / "bin" / "challenge"
    binary_path.parent.mkdir(parents=True, exist_ok=True)  # bin/ may have been cleaned
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
        return (False, "",
                f"compilation timed out after {_COMPILE_TIMEOUT:.0f}s "
                "(possible pathological source)")
    return result.returncode == 0, result.stdout, result.stderr


def _resolve_binary(challenge_dir: Path) -> Path:
    """MinGW gcc appends `.exe`, so the artifact may be challenge.exe on Windows."""
    binary_path = challenge_dir / "bin" / "challenge"
    if binary_path.exists():
        return binary_path
    exe = binary_path.with_suffix(".exe")
    return exe if exe.exists() else binary_path


def _run_challenge(challenge_dir: Path, stdin_text: str | None = None) -> tuple[bool, str, str]:
    binary_path = _resolve_binary(challenge_dir)
    result = subprocess.run(
        [str(binary_path)],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.returncode == 0, result.stdout, result.stderr.strip()


def _evidence_log(challenge_dir: Path) -> Path | None:
    ev = challenge_dir / "evidence"
    if not ev.is_dir():
        return None
    logs = [p for p in sorted(ev.glob("*.log")) if p.name != "_decoy.log"]
    named = ev / f"{challenge_dir.name}.log"
    if named in logs:
        return named
    return logs[0] if logs else None


def _validate_log(challenge_dir: Path, metadata: dict) -> ValidationResult:
    """Re-run each task's extractor against the evidence log and confirm its
    declared answer — the real on-disk re-verification for log challenges.

    Note: the anti-hardcode decoy test (extractor must NOT reproduce the answer
    on an unrelated log) runs at generation time, not here; validate re-checks
    correctness against the real log only."""
    log = _evidence_log(challenge_dir)
    if log is None:
        return _fail("no-log", "no evidence log found")
    tasks = metadata.get("tasks") or []
    if not tasks:
        return _fail("no-tasks", "challenge.json has no tasks")
    ok = 0
    for i, task in enumerate(tasks, 1):
        ext = challenge_dir / "organizer" / f"extractor_{i}.py"
        if not ext.exists():
            return _fail("missing-extractor", f"missing organizer/extractor_{i}.py")
        try:
            res = subprocess.run(["python", "-I", str(ext), str(log)],
                                 capture_output=True, text=True, timeout=20)
        except Exception as exc:
            return _fail("extractor-error", f"task {i} extractor did not run\n{exc}")
        if res.returncode != 0:
            return _fail("extractor-crash", f"task {i} extractor crashed\n{res.stderr.strip()[:200]}")
        if res.stdout.strip() != str(task.get("answer", "")).strip():
            return _fail("answer-mismatch", f"task {i} answer mismatch\n"
                         f"Expected: {task.get('answer')!r}  Got: {res.stdout.strip()!r}")
        ok += 1
    return _ok("ok", f"Validation successful: {ok}/{len(tasks)} task(s) re-extracted the expected "
               f"answer from {log.name}.\nDir: {challenge_dir}")


def validate_challenge(challenge_dir: Path) -> ValidationResult:
    challenge_dir = Path(challenge_dir)
    required = [
        challenge_dir / "README.md",
        challenge_dir / "challenge.json",
        challenge_dir / "solution.md",
    ]

    for required_file in required:
        if not required_file.exists():
            return _fail("missing-file", f"missing {required_file.name}")

    try:
        metadata = json.loads((challenge_dir / "challenge.json").read_text(encoding="utf-8"))
    except Exception:
        return _fail("invalid-json", "invalid challenge.json")

    if metadata.get("challenge_kind") == "log":
        return _validate_log(challenge_dir, metadata)

    src_c = challenge_dir / "src" / "challenge.c"
    if src_c.exists():
        compile_ok, compile_out, compile_err = _compile_binary(challenge_dir)
        if not compile_ok:
            if "GCC compiler not found" in compile_err:
                return _fail("no-compiler", compile_err)
            return _fail("compile-error", f"compile error\n{compile_err}")

        try:
            expected_flag = metadata["flag"]
            kind = metadata.get("challenge_kind", "decode")
        except Exception:
            return _fail("invalid-json", "invalid challenge.json")

        # Fail closed: a leak check that can't read the binary is not a pass.
        try:
            leaked = _flag_leaks(_resolve_binary(challenge_dir), expected_flag)
        except OSError as exc:
            return _fail("leak-check-failed", f"could not read the compiled binary for the leak check\n{exc}")
        if leaked:
            return _fail("flag-leak", "the plaintext secret is readable in the binary (strings leak)")

        if kind == "check":
            marker = str(metadata.get("success_marker", "")).strip()
            if not marker:
                return _fail("no-marker", "check challenge missing success_marker")
            try:
                _, good_out, _ = _run_challenge(challenge_dir, stdin_text=expected_flag + "\n")
                _, bad_out, _ = _run_challenge(challenge_dir, stdin_text="definitely_wrong_" + expected_flag + "\n")
            except Exception as exc:
                return _fail("run-error", f"challenge did not run\n{exc}")
            if marker not in good_out:
                return _fail("check-reject", f"correct secret rejected (expected marker {marker!r})")
            if marker in bad_out:
                return _fail("check-accept-wrong", "a wrong input was also accepted")
            return _ok("ok", "Validation successful: challenge compiled; the secret is accepted, "
                       f"wrong input is rejected, and strings does not leak it.\nDir: {challenge_dir}")

        run_ok, out, run_err = _run_challenge(challenge_dir)
        if not run_ok:
            return _fail("run-error", f"challenge did not run\n{run_err}")
        if out.strip() != expected_flag:
            return _fail("flag-mismatch", f"incorrect flag\nExpected: {expected_flag}\nGot: {out.strip()}")

        return _ok("ok", "Validation successful: challenge compiled, ran, printed the expected flag, "
                   f"and strings does not leak it.\nDir: {challenge_dir}")

    try:
        expected_flag = metadata["flag"]
        title = metadata["title"]
        if not expected_flag or not title:
            raise ValueError("Invalid metadata")
    except Exception:
        return _fail("invalid-json", "invalid challenge.json or missing required metadata")

    return _ok("ok", "Validation successful: evidence-based challenge generated and metadata "
               f"verified.\nDir: {challenge_dir}")
