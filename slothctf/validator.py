from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .execution import ExecutionBackend, SandboxUnavailable, resolve_backend
from .targets import DEFAULT_TARGET, Target, TargetError, get_target, infer_target


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


def _validate_log(challenge_dir: Path, metadata: dict,
                  backend: ExecutionBackend) -> ValidationResult:
    """Re-run each task's extractor against the evidence log (isolated) and confirm
    its declared answer — the real on-disk re-verification for log challenges.

    Note: the anti-hardcode decoy test (extractor must NOT reproduce the answer
    on an unrelated log) runs at generation time, not here; validate re-checks
    correctness against the real log only."""
    log = _evidence_log(challenge_dir)
    if log is None:
        return _fail("no-log", "no evidence log found")
    log_rel = log.relative_to(challenge_dir).as_posix()
    tasks = metadata.get("tasks") or []
    if not tasks:
        return _fail("no-tasks", "challenge.json has no tasks")
    ok = 0
    for i, task in enumerate(tasks, 1):
        ext_rel = f"organizer/extractor_{i}.py"
        if not (challenge_dir / ext_rel).exists():
            return _fail("missing-extractor", f"missing {ext_rel}")
        res = backend.run_python(challenge_dir, ext_rel, [log_rel])
        if res.timed_out:
            return _fail("extractor-error", f"task {i} extractor timed out")
        if not res.ok:
            return _fail("extractor-crash", f"task {i} extractor crashed\n{res.stderr.strip()[:200]}")
        if res.stdout.strip() != str(task.get("answer", "")).strip():
            return _fail("answer-mismatch", f"task {i} answer mismatch\n"
                         f"Expected: {task.get('answer')!r}  Got: {res.stdout.strip()!r}")
        ok += 1
    return _ok("ok", f"Validation successful: {ok}/{len(tasks)} task(s) re-extracted the expected "
               f"answer from {log.name}.\nDir: {challenge_dir}")


def _validate_recover(challenge_dir: Path, metadata: dict, target: Target,
                      backend: ExecutionBackend) -> ValidationResult:
    """Re-validate a `recover` challenge from what is distributed on disk.

    REPEATED here (same as generation): the challenge compiles; running it does
    not reveal the flag; the plaintext flag does not leak in the binary (UTF-8 or
    UTF-16LE); the reference solver recovers the flag from the artifact alone, in
    an isolated workspace. NOT repeated here: the generation-time independence
    (flag-swap) proof — that a second, harness-chosen flag is recovered — which is
    a property of the generator, not of the distributed artifact."""
    from .generator import (  # internal reuse
        _flag_leaks_recover, _recover_execution_problem, _run_solver_isolated,
    )

    flag = str(metadata.get("flag", ""))
    challenge_rel = f"bin/{target.artifact('challenge')}"
    if not (challenge_dir / "src" / "challenge.c").exists():
        return _fail("missing-file", "missing src/challenge.c")
    solver = challenge_dir / "organizer" / "solver.py"
    if not solver.exists():
        return _fail("missing-solver", "missing organizer/solver.py")

    comp = backend.compile_c(challenge_dir, "src/challenge.c", challenge_rel, target)
    if not comp.ok:
        return _fail("compile-error", f"compile error\n{comp.stderr}")
    challenge_bin = challenge_dir / challenge_rel

    try:
        if _flag_leaks_recover(challenge_bin, flag):
            return _fail("flag-leak", "the plaintext flag is readable in the binary (UTF-8 or UTF-16LE)")
    except OSError as exc:
        return _fail("leak-check-failed", f"could not read the compiled binary\n{exc}")

    # The distributed program must work: accept the correct flag, reject wrong
    # input, never crash/time out, and never reveal the flag (same as generation).
    marker = str(metadata.get("success_marker", "")).strip()
    if not marker:
        return _fail("no-marker", "recover challenge missing success_marker")
    good = backend.run_binary(challenge_dir, challenge_rel, target, stdin_text=flag + "\n")
    bad = backend.run_binary(challenge_dir, challenge_rel, target,
                             stdin_text="definitely_wrong_" + flag + "\n")
    prob = _recover_execution_problem(good, bad, marker, flag)
    if prob is not None:
        return _fail(prob[0], prob[1])

    # The solver runs on a Python-capable backend; for a PE target resolve the
    # isolated ELF backend (no Wine needed — the solver only reads bytes).
    solver_backend = backend
    if backend.name == "docker-wincompat":
        try:
            solver_backend = resolve_backend("elf", "isolated")
        except SandboxUnavailable as exc:
            return _fail("sandbox-unavailable", f"solver needs the ELF sandbox: {exc}")

    ok, out, detail = _run_solver_isolated(solver.read_text(encoding="utf-8"), challenge_bin, solver_backend)
    if not ok:
        return _fail("solver-error", detail)
    if out.strip() != flag:
        return _fail("solver-mismatch", f"reference solver recovered {out.strip()!r}, expected {flag!r}")
    return _ok("ok", "Validation successful: the binary hides the flag, a plain run does not reveal it, "
               "and the reference solver recovers it from the artifact alone.\n"
               f"Dir: {challenge_dir}")


def _resolve_backend_for(challenge_dir: Path, metadata: dict, backend: ExecutionBackend | None,
                         *, is_log: bool) -> tuple[Target, ExecutionBackend]:
    """Determine the challenge's target and the backend to re-verify it with.
    Raises TargetError / SandboxUnavailable, which the caller turns into a result."""
    if is_log:
        target = get_target(metadata.get("target", DEFAULT_TARGET))
    else:
        target = infer_target(challenge_dir, metadata)  # manifest, else binary magic bytes
    be = backend or resolve_backend(target, "isolated")
    if not is_log and not be.can_build(target):
        raise SandboxUnavailable(
            f"backend {be.name!r} cannot re-validate a {target.name!r} challenge "
            f"(it builds/runs: {', '.join(sorted(be.supported_targets)) or 'none'})")
    return target, be


def validate_challenge(challenge_dir: Path, *,
                       backend: ExecutionBackend | None = None) -> ValidationResult:
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

    is_log = metadata.get("challenge_kind") == "log"
    src_c = challenge_dir / "src" / "challenge.c"
    needs_exec = is_log or src_c.exists()
    target = None
    if needs_exec:
        # No silent fallback: a PE challenge with no isolated backend (Phase A) or a
        # missing Docker daemon is reported, not run unsandboxed. Pass an explicit
        # NativeBackend (--unsafe-native) to re-validate on the host deliberately.
        try:
            target, backend = _resolve_backend_for(challenge_dir, metadata, backend, is_log=is_log)
        except TargetError as exc:
            return _fail("target-unknown", str(exc))
        except SandboxUnavailable as exc:
            return _fail("sandbox-unavailable", str(exc))

    if is_log:
        return _validate_log(challenge_dir, metadata, backend)

    if metadata.get("challenge_kind") == "recover":
        return _validate_recover(challenge_dir, metadata, target, backend)

    if src_c.exists():
        challenge_rel = f"bin/{target.artifact('challenge')}"
        comp = backend.compile_c(challenge_dir, "src/challenge.c", challenge_rel, target)
        if not comp.ok:
            return _fail("compile-error", f"compile error\n{comp.stderr}")

        try:
            expected_flag = metadata["flag"]
            kind = metadata.get("challenge_kind", "decode")
        except Exception:
            return _fail("invalid-json", "invalid challenge.json")

        # Fail closed: a leak check that can't read the binary is not a pass.
        try:
            leaked = _flag_leaks(challenge_dir / challenge_rel, expected_flag)
        except OSError as exc:
            return _fail("leak-check-failed", f"could not read the compiled binary for the leak check\n{exc}")
        if leaked:
            return _fail("flag-leak", "the plaintext secret is readable in the binary (strings leak)")

        if kind == "check":
            marker = str(metadata.get("success_marker", "")).strip()
            if not marker:
                return _fail("no-marker", "check challenge missing success_marker")
            good = backend.run_binary(challenge_dir, challenge_rel, target, stdin_text=expected_flag + "\n")
            bad = backend.run_binary(challenge_dir, challenge_rel, target,
                                     stdin_text="definitely_wrong_" + expected_flag + "\n")
            if good.timed_out or bad.timed_out:
                return _fail("run-error", "challenge did not finish within the time limit")
            if marker not in good.stdout:
                return _fail("check-reject", f"correct secret rejected (expected marker {marker!r})")
            if marker in bad.stdout:
                return _fail("check-accept-wrong", "a wrong input was also accepted")
            return _ok("ok", "Validation successful: challenge compiled; the secret is accepted, "
                       f"wrong input is rejected, and strings does not leak it.\nDir: {challenge_dir}")

        run = backend.run_binary(challenge_dir, challenge_rel, target)
        if run.timed_out or not run.ok:
            return _fail("run-error", f"challenge did not run\n{run.stderr}")
        if run.stdout.strip() != expected_flag:
            return _fail("flag-mismatch", f"incorrect flag\nExpected: {expected_flag}\nGot: {run.stdout.strip()}")

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
