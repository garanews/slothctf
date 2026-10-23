"""Wine compatibility study for SlothCTF's existing Windows (PE) challenges.

Phase B asks a result-driven question: how faithfully does Wine run the .exe
challenges SlothCTF already produced? This tool measures that FUNCTIONALLY (flag /
success-marker correctness), not just by exit code, and separates true Wine
incompatibility from a broken challenge, an infrastructure error, or a
non-runnable case.

For each challenge under generated/ that has a PE binary it records, without ever
modifying the challenge:

  * native  — run the ORIGINAL .exe on this host (ground truth; only on Windows)
  * wine    — run the SAME original .exe under Wine in the isolated container
  * compile — can src/challenge.c be re-cross-compiled with MinGW-w64 (a separate
              data point; the run above still uses the original binary)
  * correctness of flag / input / output for each runner
  * WinAPI / anti-debug / anti-VM signals detected in the source
  * a verdict and, where identifiable, the failure cause

Verdicts:
  ok               native correct AND wine correct
  wine-incompat    native correct BUT wine wrong/failed  -> Wine is the problem
  challenge-problem native itself wrong/failed            -> the challenge, not Wine
  infra-error      docker/wine/container failure unrelated to the program
  not-runnable     no PE / unreadable / missing manifest fields
  needs-ground-truth  wine ran but no native result to compare (not on Windows)

Usage:
  python tools/wine_compat_report.py [--generated generated] [--out report.md]
                                     [--json report.json] [--only name1,name2]
Run it on the Windows + Docker Desktop host so the native column is populated.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Import the package whether run from the repo root or elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slothctf.execution import (  # noqa: E402
    DockerWinCompatBackend,
    NativeBackend,
    ResourceLimits,
    SandboxUnavailable,
    docker_available,
    win_image_available,
)
from slothctf.targets import PE, TargetError, detect_target_from_binary, host_native_target  # noqa: E402

# Source tokens that flag Windows-specific behaviour likely to stress Wine.
_WINAPI_TOKENS = ("windows.h", "winbase.h", "winuser.h", "processthreadsapi.h")
_ANTIDEBUG_TOKENS = ("IsDebuggerPresent", "CheckRemoteDebuggerPresent",
                     "NtQueryInformationProcess", "OutputDebugString", "BeingDebugged",
                     "__debugbreak", "DebugActiveProcess")
_ANTIVM_TOKENS = ("cpuid", "VMware", "VMWARE", "VirtualBox", "vbox", "hypervisor",
                  "rdtsc", "SMBIOS")


@dataclass
class RunOutcome:
    ran: bool = False          # the process started and finished (not infra failure)
    correct: bool | None = None  # flag/marker correctness, or None if it could not run
    stdout: str = ""
    detail: str = ""           # error / mismatch explanation


@dataclass
class Row:
    name: str
    kind: str = ""
    target_ok: bool = False
    uses_winapi: bool = False
    anti_debug: bool = False
    anti_vm: bool = False
    native: RunOutcome = field(default_factory=RunOutcome)
    wine: RunOutcome = field(default_factory=RunOutcome)
    compile_ok: bool | None = None   # MinGW re-compile from source (separate check)
    verdict: str = "not-runnable"
    cause: str = ""


def _scan_source(challenge_dir: Path) -> tuple[bool, bool, bool]:
    text = ""
    for c in sorted(challenge_dir.glob("src/*.c")) + sorted(challenge_dir.glob("organizer/*.c")):
        try:
            text += c.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    low = text.lower()
    winapi = any(tok.lower() in low for tok in _WINAPI_TOKENS)
    antidebug = any(tok.lower() in low for tok in _ANTIDEBUG_TOKENS)
    antivm = any(tok.lower() in low for tok in _ANTIVM_TOKENS)
    return winapi, antidebug, antivm


def _check_correct(kind: str, flag: str, marker: str, run_ok, run_stdout,
                   run_bad_stdout=None) -> tuple[bool, str]:
    """Functional correctness, not exit code. For decode: stdout equals the flag.
    For check: the marker appears for the correct secret and NOT for a wrong one."""
    if kind == "check":
        if not marker:
            return False, "check challenge without success_marker"
        if marker not in run_stdout:
            return False, f"correct secret did not print marker {marker!r} (got {run_stdout.strip()[:80]!r})"
        if run_bad_stdout is not None and marker in run_bad_stdout:
            return False, "a wrong input also printed the success marker"
        return True, ""
    # decode
    if run_stdout.strip() == flag:
        return True, ""
    return False, f"printed {run_stdout.strip()[:80]!r}, expected flag {flag!r}"


def _run_one(runner, challenge_dir: Path, exe_rel: str, kind: str, flag: str,
             marker: str) -> RunOutcome:
    """Run a PE through a backend (native host or Wine) and judge correctness.
    Distinguishes an infrastructure failure (ran=False) from a wrong result."""
    out = RunOutcome()
    try:
        if kind == "check":
            good = runner.run_binary(challenge_dir, exe_rel, PE, stdin_text=flag + "\n")
            bad = runner.run_binary(challenge_dir, exe_rel, PE,
                                    stdin_text="definitely_wrong_" + flag + "\n")
            if good.timed_out or bad.timed_out:
                out.detail = "timed out"
                return out
            out.ran, out.stdout = True, good.stdout
            out.correct, out.detail = _check_correct(kind, flag, marker, good.ok, good.stdout, bad.stdout)
        else:
            r = runner.run_binary(challenge_dir, exe_rel, PE)
            if r.timed_out:
                out.detail = "timed out"
                return out
            out.ran, out.stdout = True, r.stdout
            out.correct, out.detail = _check_correct(kind, flag, marker, r.ok, r.stdout)
    except SandboxUnavailable as exc:
        out.detail = f"infra: {exc}"
    except Exception as exc:  # unexpected backend/container error -> infrastructure
        out.detail = f"infra: {type(exc).__name__}: {exc}"
    return out


def _try_recompile(wine: DockerWinCompatBackend, challenge_dir: Path) -> bool | None:
    src = challenge_dir / "src" / "challenge.c"
    if not src.exists():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "src").mkdir()
        (ws / "src" / "challenge.c").write_text(src.read_text(encoding="utf-8", errors="replace"),
                                                encoding="utf-8")
        try:
            r = wine.compile_c(ws, "src/challenge.c", "bin/challenge.exe", PE,
                               limits=ResourceLimits(compile_timeout=90))
            return bool(r.ok)
        except Exception:
            return False


def _verdict(row: Row, native_available: bool) -> tuple[str, str]:
    if not row.target_ok:
        return "not-runnable", "no valid x86-64 PE binary"
    w = row.wine
    n = row.native
    # Wine infrastructure failed outright.
    if not w.ran and w.detail.startswith("infra"):
        return "infra-error", w.detail
    if not native_available:
        if w.ran and w.correct:
            return "needs-ground-truth", "wine correct; run on Windows for native comparison"
        return "needs-ground-truth", (w.detail or "no native ground truth on this host")
    # We have a native ground truth.
    native_correct = n.ran and bool(n.correct)
    wine_correct = w.ran and bool(w.correct)
    if not native_correct:
        return "challenge-problem", f"native run not correct: {n.detail or 'failed'}"
    if wine_correct:
        return "ok", ""
    return "wine-incompat", f"native correct but wine: {w.detail or 'failed/incorrect'}"


def evaluate(challenge_dir: Path, wine: DockerWinCompatBackend,
             native: NativeBackend | None) -> Row:
    row = Row(name=challenge_dir.name)
    manifest = challenge_dir / "challenge.json"
    exe = challenge_dir / "bin" / "challenge.exe"
    if not manifest.exists() or not exe.exists():
        row.cause = "missing challenge.json or bin/challenge.exe"
        return row
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        row.cause = f"bad manifest: {exc}"
        return row
    row.kind = str(meta.get("challenge_kind", "decode"))
    flag = str(meta.get("flag", ""))
    marker = str(meta.get("success_marker", "")).strip()
    row.uses_winapi, row.anti_debug, row.anti_vm = _scan_source(challenge_dir)

    # Verify PE signature + architecture (not the extension) on the ORIGINAL binary.
    try:
        row.target_ok = detect_target_from_binary(exe).name == "pe"
    except TargetError as exc:
        row.cause = f"not a valid PE: {exc}"
        return row

    exe_rel = "bin/challenge.exe"
    row.wine = _run_one(wine, challenge_dir, exe_rel, row.kind, flag, marker)
    if native is not None:
        row.native = _run_one(native, challenge_dir, exe_rel, row.kind, flag, marker)
    row.compile_ok = _try_recompile(wine, challenge_dir)
    row.verdict, row.cause = _verdict(row, native is not None)
    return row


def _fmt_oc(oc: RunOutcome) -> str:
    if not oc.ran and not oc.detail:
        return "—"
    if not oc.ran:
        return f"ERR ({oc.detail[:40]})"
    return "correct" if oc.correct else f"wrong ({oc.detail[:40]})"


def render_markdown(rows: list[Row], native_available: bool) -> str:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    lines = ["# Wine compatibility report — SlothCTF PE challenges", ""]
    lines.append(f"Challenges examined: **{len(rows)}**  ·  native ground truth: "
                 f"**{'yes (Windows host)' if native_available else 'no (run on Windows for it)'}**")
    lines.append("")
    lines.append("## Summary")
    for v in ("ok", "wine-incompat", "challenge-problem", "infra-error",
              "needs-ground-truth", "not-runnable"):
        if counts.get(v):
            lines.append(f"- **{v}**: {counts[v]}")
    lines.append("")
    lines.append("## Matrix")
    lines.append("| Challenge | Kind | WinAPI | anti-dbg | anti-VM | compile | native | wine | verdict | cause |")
    lines.append("|---|---|:--:|:--:|:--:|:--:|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x.verdict, x.name)):
        def y(b): return "✓" if b else ""
        comp = {True: "ok", False: "FAIL", None: "—"}[r.compile_ok]
        lines.append(
            f"| {r.name} | {r.kind} | {y(r.uses_winapi)} | {y(r.anti_debug)} | {y(r.anti_vm)} | "
            f"{comp} | {_fmt_oc(r.native)} | {_fmt_oc(r.wine)} | **{r.verdict}** | {r.cause[:60]} |")
    lines.append("")
    focus = [r for r in rows if r.uses_winapi or r.anti_debug or r.anti_vm
             or r.name in ("anti_debug_demo", "anti_vm_demo")]
    if focus:
        lines.append("## Focus: WinAPI / anti-debug / anti-VM challenges")
        for r in sorted(focus, key=lambda x: x.name):
            tags = ", ".join(t for t, b in (("WinAPI", r.uses_winapi), ("anti-debug", r.anti_debug),
                                            ("anti-VM", r.anti_vm)) if b) or "—"
            lines.append(f"- **{r.name}** ({tags}): {r.verdict}"
                         + (f" — {r.cause}" if r.cause else ""))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Wine compatibility report for PE challenges")
    ap.add_argument("--generated", default="generated", help="Directory of generated challenges")
    ap.add_argument("--out", default="docs/wine-compat-report.md", help="Markdown report path")
    ap.add_argument("--json", default=None, help="Also write the raw rows as JSON here")
    ap.add_argument("--only", default=None, help="Comma-separated challenge names to limit to")
    args = ap.parse_args(argv)

    if not docker_available():
        print("Docker is not available — start Docker Desktop.", file=sys.stderr)
        return 2
    if not win_image_available():
        print("Windows-compat image missing — run: python main.py sandbox-build --win", file=sys.stderr)
        return 2

    wine = DockerWinCompatBackend()
    native = NativeBackend() if host_native_target().name == "pe" else None  # only on Windows

    gen = Path(args.generated)
    names = set(args.only.split(",")) if args.only else None
    dirs = [d for d in sorted(gen.iterdir()) if d.is_dir()
            and (d / "bin" / "challenge.exe").exists()
            and (names is None or d.name in names)]
    print(f"Evaluating {len(dirs)} PE challenge(s) "
          f"({'with' if native else 'WITHOUT'} native ground truth)...", file=sys.stderr)

    rows = []
    for d in dirs:
        print(f"  {d.name} ...", file=sys.stderr)
        rows.append(evaluate(d, wine, native))

    md = render_markdown(rows, native is not None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
