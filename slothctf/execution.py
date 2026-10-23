"""Execution backends for model-written code.

SlothCTF compiles and runs code an LLM produced (C programs; Python generators and
extractors for the log genre). That is untrusted code, so WHERE it runs matters as
much as that it runs. This module is the single choke point for every such step.

Two backends implement the same small contract:

- ``NativeBackend``  — runs on the host with the user's own toolchain. This is the
  pre-sandbox behaviour and is **unsafe**: no isolation, the code runs as you. It is
  for controlled development only and must be opted into explicitly.
- ``DockerBackend``  — runs each step inside a throwaway container with no network,
  a non-root user, a read-only root filesystem, a size-limited tmpfs, and CPU / RAM
  / PID / time / output caps, cleaning up even on error.

``get_backend`` selects one. It **never falls back**: if Docker is requested (the
default) and unavailable, it raises — it does not silently run unsandboxed. Native
mode requires a conscious opt-in.

Security note: a container is a strong boundary but not a VM. It shares the host
kernel, so a kernel-level escape is in scope in a way it is not for a hypervisor.
For hostile, not merely untrusted, input, run SlothCTF inside a dedicated VM. See
docs/sandbox.md.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .targets import Target, get_target, host_native_target


class SandboxUnavailable(RuntimeError):
    """Raised when the requested backend cannot run and no fallback is allowed."""


@dataclass
class ResourceLimits:
    """Caps applied to a single compile or run step."""
    cpus: float = 1.0
    memory: str = "512m"        # Docker memory string (e.g. "512m", "1g")
    pids: int = 256
    compile_timeout: float = 60.0
    run_timeout: float = 15.0
    output_bytes: int = 1_000_000  # captured stdout/stderr is truncated past this


@dataclass
class ExecResult:
    """Outcome of one step. ``ok`` means exit code 0 and not timed out."""
    ok: bool
    stdout: str
    stderr: str
    exit_code: int | None = None
    timed_out: bool = False
    truncated: bool = False


def _truncate(s: str, limit: int) -> tuple[str, bool]:
    if len(s) <= limit:
        return s, False
    return s[:limit] + f"\n...[output truncated at {limit} bytes]", True


# --------------------------------------------------------------------------- #
# Backend contract
# --------------------------------------------------------------------------- #
class ExecutionBackend(ABC):
    """Runs compile / binary / python steps against a *workspace* directory.

    Paths are given as the workspace plus POSIX-relative strings so the Docker
    backend can map them to container paths. Artifacts (e.g. the compiled binary)
    are written back into the workspace so host-side checks — reading the binary
    bytes for the strings-leak test, comparing stdout — work unchanged.
    """
    name: str = "base"
    safe: bool = False
    supported_targets: frozenset[str] = frozenset()

    def can_build(self, target: Target) -> bool:
        return target.name in self.supported_targets

    def _require_target(self, target: Target) -> None:
        if not self.can_build(target):
            raise SandboxUnavailable(
                f"backend {self.name!r} cannot build/run target {target.name!r} "
                f"(supports: {', '.join(sorted(self.supported_targets)) or 'none'})")

    @abstractmethod
    def compile_c(self, workspace: Path, source_rel: str, out_rel: str, target: Target,
                  limits: ResourceLimits | None = None) -> ExecResult:
        ...

    @abstractmethod
    def run_binary(self, workspace: Path, binary_rel: str, target: Target,
                   args: list[str] | None = None, stdin_text: str | None = None,
                   limits: ResourceLimits | None = None) -> ExecResult:
        ...

    @abstractmethod
    def run_python(self, workspace: Path, script_rel: str, args: list[str] | None = None,
                   stdin_text: str | None = None, limits: ResourceLimits | None = None) -> ExecResult:
        ...


# --------------------------------------------------------------------------- #
# Native (unsafe) backend — current host behaviour
# --------------------------------------------------------------------------- #
class NativeBackend(ExecutionBackend):
    """Runs on the host with no isolation, using the host's own toolchain. UNSAFE;
    development only. It can only build and run the host-native target (PE on
    Windows, ELF on Linux) — it does not cross-compile or emulate."""
    name = "native"
    safe = False

    def __init__(self):
        self._native = host_native_target()
        self.supported_targets = frozenset({self._native.name})

    def _compiler_for(self, target: Target) -> str:
        for cand in target.native_compilers:
            found = shutil.which(cand)
            if found:
                return found
        raise SandboxUnavailable(
            f"no native compiler for target {target.name!r} on PATH "
            f"(looked for: {', '.join(target.native_compilers)})")

    def _run(self, argv: list[str], timeout: float, limits: ResourceLimits,
             workspace: Path, stdin_text: str | None = None) -> ExecResult:
        try:
            r = subprocess.run(
                argv, input=stdin_text, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
                cwd=str(workspace),  # match the container's /work cwd, so relative args resolve the same
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, "", f"timed out after {timeout:.0f}s", timed_out=True)
        out, t1 = _truncate(r.stdout or "", limits.output_bytes)
        err, t2 = _truncate((r.stderr or "").strip(), limits.output_bytes)
        return ExecResult(r.returncode == 0, out, err, exit_code=r.returncode, truncated=t1 or t2)

    def compile_c(self, workspace, source_rel, out_rel, target, limits=None):
        self._require_target(target)
        limits = limits or ResourceLimits()
        compiler = self._compiler_for(target)
        (workspace / out_rel).parent.mkdir(parents=True, exist_ok=True)
        return self._run([compiler, "-O2", "-s", "-o", out_rel, source_rel],
                         limits.compile_timeout, limits, workspace)

    def run_binary(self, workspace, binary_rel, target, args=None, stdin_text=None, limits=None):
        self._require_target(target)
        limits = limits or ResourceLimits()
        # Absolute path for the binary itself (not every OS resolves ./rel), cwd=workspace.
        return self._run([str((workspace / binary_rel).resolve()), *(args or [])],
                         limits.run_timeout, limits, workspace, stdin_text=stdin_text)

    def run_python(self, workspace, script_rel, args=None, stdin_text=None, limits=None):
        limits = limits or ResourceLimits()
        return self._run(["python", "-I", script_rel, *(args or [])],
                         limits.run_timeout, limits, workspace, stdin_text=stdin_text)


# --------------------------------------------------------------------------- #
# Docker backend — isolated execution
# --------------------------------------------------------------------------- #
DEFAULT_IMAGE = os.environ.get("SLOTHCTF_DOCKER_IMAGE", "slothctf-sandbox:latest")
_WORK = "/work"  # workspace mount point inside the container


def build_docker_run_argv(image: str, inner: list[str], workspace: Path,
                          limits: ResourceLimits, name: str,
                          stdin: bool = False) -> list[str]:
    """Build the full ``docker run`` argv for one step — pure and side-effect free
    so the isolation flags can be unit-tested without a daemon.

    The container has: no network, a non-root user, no added capabilities and
    no-new-privileges, a read-only root filesystem with a small writable tmpfs for
    /tmp, CPU / memory / PID caps, and the workspace bind-mounted read-write at
    /work (the compiled artifact must land back on the host). ``--rm`` plus an
    explicit --name let the caller force-remove a timed-out container.
    """
    argv = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--user", "65534:65534",          # nobody:nogroup
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=64m,noexec",
        "--pids-limit", str(limits.pids),
        "--memory", limits.memory,
        "--cpus", str(limits.cpus),
        "--workdir", _WORK,
        "-v", f"{str(workspace.resolve())}:{_WORK}:rw",
    ]
    if stdin:
        argv.append("-i")
    argv.append(image)
    argv.extend(inner)
    return argv


class DockerLinuxBackend(ExecutionBackend):
    """Runs each step inside a throwaway, network-less, non-root container and
    builds Linux ELF (Phase A). Works with Docker Desktop + WSL2 on Windows and
    native Docker on Linux. The workspace is bind-mounted, so the ELF artifact is
    readable on the host (for the byte-level leak check) and runnable in-container.

    Windows PE under this backend is a later phase (DockerWinCompat, MinGW + Wine);
    here it is explicitly unsupported rather than silently cross-compiled.
    """
    name = "docker-linux"
    safe = True
    supported_targets = frozenset({"elf"})

    def __init__(self, image: str = DEFAULT_IMAGE):
        self.image = image

    def _exec(self, inner: list[str], workspace: Path, timeout: float,
              limits: ResourceLimits, stdin_text: str | None = None) -> ExecResult:
        name = f"slothctf-{uuid.uuid4().hex[:12]}"
        argv = build_docker_run_argv(self.image, inner, workspace, limits, name,
                                     stdin=stdin_text is not None)
        try:
            r = subprocess.run(
                argv, input=stdin_text, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # The CLI was killed but the container may linger — force-remove it.
            self._force_remove(name)
            return ExecResult(False, "", f"timed out after {timeout:.0f}s", timed_out=True)
        out, t1 = _truncate(r.stdout or "", limits.output_bytes)
        err, t2 = _truncate((r.stderr or "").strip(), limits.output_bytes)
        return ExecResult(r.returncode == 0, out, err, exit_code=r.returncode, truncated=t1 or t2)

    @staticmethod
    def _force_remove(name: str) -> None:
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                           text=True, timeout=15)
        except Exception:
            pass  # best-effort cleanup; nothing else to do if docker itself is gone

    def compile_c(self, workspace, source_rel, out_rel, target, limits=None):
        self._require_target(target)
        limits = limits or ResourceLimits()
        (workspace / out_rel).parent.mkdir(parents=True, exist_ok=True)
        src = PurePosixPath(_WORK) / source_rel
        out = PurePosixPath(_WORK) / out_rel
        inner = ["gcc", "-O2", "-s", "-o", str(out), str(src)]
        return self._exec(inner, workspace, limits.compile_timeout, limits)

    def run_binary(self, workspace, binary_rel, target, args=None, stdin_text=None, limits=None):
        self._require_target(target)
        limits = limits or ResourceLimits()
        inner = [str(PurePosixPath(_WORK) / binary_rel), *(args or [])]
        return self._exec(inner, workspace, limits.run_timeout, limits, stdin_text=stdin_text)

    def run_python(self, workspace, script_rel, args=None, stdin_text=None, limits=None):
        limits = limits or ResourceLimits()
        inner = ["python3", "-I", str(PurePosixPath(_WORK) / script_rel), *(args or [])]
        return self._exec(inner, workspace, limits.run_timeout, limits, stdin_text=stdin_text)


# --------------------------------------------------------------------------- #
# Selection — explicit, no silent fallback
# --------------------------------------------------------------------------- #
def docker_available() -> bool:
    """True only if the docker CLI exists AND its daemon responds."""
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False


def _native_opt_in(explicit: bool) -> bool:
    return explicit or os.environ.get("SLOTHCTF_ALLOW_NATIVE", "") == "1"


def get_backend(mode: str | None = None, *, allow_native: bool = False,
                image: str = DEFAULT_IMAGE) -> ExecutionBackend:
    """Return the selected backend, or raise. Never falls back to native.

    mode: "docker" (default) or "native"; taken from SLOTHCTF_EXECUTION if omitted.
    Docker is required to be available when selected. Native must be opted into via
    allow_native=True or SLOTHCTF_ALLOW_NATIVE=1, because it runs model code with no
    isolation.
    """
    mode = (mode or os.environ.get("SLOTHCTF_EXECUTION", "docker")).strip().lower()
    if mode in ("docker", "docker-linux"):
        if not docker_available():
            raise SandboxUnavailable(
                "Docker is not available (CLI missing or daemon not responding), and "
                "SlothCTF does not silently run model-written code unsandboxed. Start "
                "Docker Desktop and build the image (python main.py sandbox-build), or "
                "explicitly opt into the UNSAFE native backend with "
                "SLOTHCTF_EXECUTION=native SLOTHCTF_ALLOW_NATIVE=1.")
        return DockerLinuxBackend(image=image)
    if mode in ("native", "native-unsafe"):
        if not _native_opt_in(allow_native):
            raise SandboxUnavailable(
                "Native execution runs model-written code on the host with NO isolation. "
                "Opt in explicitly with allow_native=True or SLOTHCTF_ALLOW_NATIVE=1 if you "
                "really intend to run untrusted code as yourself (development only).")
        return NativeBackend()
    raise ValueError(f"unknown execution mode {mode!r} (use 'docker' or 'native')")


ISOLATED = "isolated"
UNSAFE_NATIVE = "unsafe-native"


def resolve_backend(target: str | Target, isolation: str = ISOLATED, *,
                    allow_native: bool = False, image: str = DEFAULT_IMAGE) -> ExecutionBackend:
    """Pick the backend for a (target, isolation) request, or raise with a clear
    reason. This is the main entry point: it couples the two axes and never falls
    back silently from isolated to native or across targets.

    isolation: "isolated" (default) or "unsafe-native" (requires an opt-in).
    """
    tgt = target if isinstance(target, Target) else get_target(target)

    if isolation == ISOLATED:
        if tgt.name == "elf":
            if not docker_available():
                raise SandboxUnavailable(
                    "Isolated ELF needs Docker, which is not available. Start Docker "
                    "Desktop and run `python main.py sandbox-build`, or use "
                    "--unsafe-native (dev only). No automatic fallback.")
            return DockerLinuxBackend(image=image)
        if tgt.name == "pe":
            raise SandboxUnavailable(
                "Isolated Windows PE is not available in Phase A. It arrives in Phase B "
                "(DockerWinCompat: MinGW + Wine). For now choose target 'elf' for an "
                "isolated build, or '--unsafe-native' on a Windows host to build the .exe "
                "with no isolation. SlothCTF will not convert PE to ELF.")
        raise SandboxUnavailable(f"no isolated backend for target {tgt.name!r}")

    if isolation == UNSAFE_NATIVE:
        if not _native_opt_in(allow_native):
            raise SandboxUnavailable(
                "The native (unsafe, no isolation) backend must be opted into explicitly: "
                "pass --unsafe-native / allow_native=True or set SLOTHCTF_ALLOW_NATIVE=1.")
        backend = NativeBackend()
        if not backend.can_build(tgt):
            raise SandboxUnavailable(
                f"this host's native toolchain emits {host_native_target().name!r}, not "
                f"{tgt.name!r}; a native {tgt.name!r} build is not possible here. Use Docker "
                "for an isolated build of that target, or run on a matching host.")
        return backend

    raise ValueError(f"unknown isolation {isolation!r} (use {ISOLATED!r} or {UNSAFE_NATIVE!r})")
