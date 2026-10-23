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
import threading
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
    output_bytes: int = 1_000_000  # hard cap on bytes of stdout/stderr retained in
    #                                memory during a run; the excess is drained and
    #                                discarded, not buffered then trimmed.


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


_READ_CHUNK = 65536  # bytes pulled from a pipe per read


def _drain(stream, limit: int, sink: list) -> None:
    """Read a child's pipe to EOF, keeping at most ``limit`` bytes.

    Runs in its own thread so stdout and stderr are consumed concurrently — a
    child can fill either pipe without the parent deadlocking. Bytes past the cap
    are read and DISCARDED (never buffered), so a process that prints gigabytes
    costs the host a bounded amount of memory, not its whole output. ``sink`` is
    ``[bytearray, truncated_flag]``; the flag is set once anything is dropped.
    """
    buf: bytearray = sink[0]
    try:
        while True:
            chunk = stream.read(_READ_CHUNK)
            if not chunk:
                break
            room = limit - len(buf)
            if room > 0:
                buf.extend(chunk[:room])
            if len(chunk) > max(room, 0):
                sink[1] = True
    except (OSError, ValueError):
        pass  # pipe closed under us (e.g. the process was killed) — stop draining
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _kill(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except Exception:
        pass


def _run_capped(argv: list[str], *, timeout: float, output_bytes: int,
                stdin_text: str | None = None, cwd: str | None = None,
                on_timeout=None) -> tuple[int | None, str, str, bool, bool]:
    """Run ``argv`` with a wall-clock timeout and truly bounded output capture.

    stdout and stderr are drained by two threads and each retained to at most
    ``output_bytes`` *bytes* (not characters); the excess is discarded while the
    process keeps running, so neither a full pipe nor a flood of output can
    deadlock or exhaust the host. Decoding to text happens only after capping, as
    UTF-8 with replacement. On timeout the process is killed and ``on_timeout`` (if
    given) is invoked for extra cleanup (e.g. force-removing a container).

    Returns ``(returncode, stdout, stderr, timed_out, truncated)``.
    """
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
    )
    out: list = [bytearray(), False]
    err: list = [bytearray(), False]
    t_out = threading.Thread(target=_drain, args=(proc.stdout, output_bytes, out), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, output_bytes, err), daemon=True)
    t_out.start()
    t_err.start()

    # stdin here is always small and trusted (a flag / candidate), so a plain
    # write+close cannot block on a full pipe the way captured output could.
    if stdin_text is not None and proc.stdin is not None:
        try:
            proc.stdin.write(stdin_text.encode("utf-8", "replace"))
        except Exception:
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill(proc)
        if on_timeout is not None:
            on_timeout()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    # The kill (or normal exit) closes the pipes, so the drain threads reach EOF.
    t_out.join(timeout=5)
    t_err.join(timeout=5)

    # Decode after capping, and normalise newlines the way the previous text-mode
    # capture did (universal newlines) so a Windows child's CRLF output is treated
    # identically to a container's LF — downstream code assumes '\n'-only text.
    def _decode(b: bytearray) -> str:
        return b.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")

    out_s = _decode(out[0])
    err_s = _decode(err[0])
    if out[1]:
        out_s += f"\n...[output truncated at {output_bytes} bytes]"
    if err[1]:
        err_s += f"\n...[output truncated at {output_bytes} bytes]"
    return proc.returncode, out_s, err_s, timed_out, out[1] or err[1]


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
        rc, out, err, timed_out, truncated = _run_capped(
            argv, timeout=timeout, output_bytes=limits.output_bytes,
            stdin_text=stdin_text,
            cwd=str(workspace),  # match the container's /work cwd, so relative args resolve the same
        )
        if timed_out:
            return ExecResult(False, "", f"timed out after {timeout:.0f}s", timed_out=True)
        return ExecResult(rc == 0, out, err.strip(), exit_code=rc, truncated=truncated)

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
                          stdin: bool = False, *,
                          mounts: list[tuple[str, str, str]] | None = None,
                          tmpfs: str = "/tmp:rw,size=64m,noexec",
                          env: list[tuple[str, str]] | None = None,
                          workdir: str = _WORK) -> list[str]:
    """Build the full ``docker run`` argv for one step — pure and side-effect free
    so the isolation flags can be unit-tested without a daemon.

    The container has: no network, a non-root user, no added capabilities and
    no-new-privileges, a read-only root filesystem with a writable tmpfs for /tmp,
    and CPU / memory / PID caps. ``--rm`` plus an explicit --name let the caller
    force-remove a timed-out container.

    Mounts: by default the whole workspace is bind-mounted read-write at /work (the
    compiled artifact must land back on the host). Pass ``mounts`` as a list of
    ``(host_src, container_dst, mode)`` to instead expose ONLY those paths — the
    Windows-compat backend uses this to run a PE with just that one file visible,
    read-only, so model code under Wine cannot read or write the rest of the
    workspace. ``tmpfs`` and ``env`` are overridable for backends (e.g. Wine needs a
    larger, exec-capable /tmp and its WINEPREFIX env). These are isolation knobs;
    they do not change the fixed security flags above.
    """
    argv = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--user", "65534:65534",          # nobody:nogroup
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--tmpfs", tmpfs,
        "--pids-limit", str(limits.pids),
        "--memory", limits.memory,
        "--cpus", str(limits.cpus),
        "--workdir", workdir,
    ]
    for k, v in (env or []):
        argv += ["-e", f"{k}={v}"]
    if mounts is None:
        argv += ["-v", f"{str(workspace.resolve())}:{_WORK}:rw"]
    else:
        for src, dst, mode in mounts:
            argv += ["-v", f"{str(Path(src).resolve())}:{dst}:{mode}"]
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
        # On timeout the `docker run` CLI is killed but the container may linger —
        # force-remove it by name so it never outlives the step.
        rc, out, err, timed_out, truncated = _run_capped(
            argv, timeout=timeout, output_bytes=limits.output_bytes,
            stdin_text=stdin_text, on_timeout=lambda: self._force_remove(name),
        )
        if timed_out:
            return ExecResult(False, "", f"timed out after {timeout:.0f}s", timed_out=True)
        return ExecResult(rc == 0, out, err.strip(), exit_code=rc, truncated=truncated)

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
# Docker Windows-compat backend — isolated PE via MinGW-w64 + Wine (Phase B)
# --------------------------------------------------------------------------- #
DEFAULT_WIN_IMAGE = os.environ.get("SLOTHCTF_WIN_IMAGE", "slothctf-sandbox-win:latest")
_RUN_EXE = "/run/challenge.exe"  # read-only mount point for the PE under test


class DockerWinCompatBackend(ExecutionBackend):
    """Cross-compiles Windows PE with MinGW-w64 and runs it under Wine, inside the
    same throwaway, network-less, non-root, read-only-root container as the Linux
    backend. It builds the ``pe`` target only; ELF is explicitly unsupported here
    (use DockerLinuxBackend), never silently retargeted.

    Fidelity caveat: Wine is a compatibility layer, not Windows. A PE verified here
    is "likely correct", not "behaves exactly as on Windows" — ``fidelity`` records
    that so callers can surface it. Container isolation is independent of, and must
    not be weakened for, emulation fidelity.
    """
    name = "docker-wincompat"
    safe = True
    supported_targets = frozenset({"pe"})
    fidelity = "wine"  # vs "native" — NOT real Windows

    # Wine needs a writable, exec-capable per-run WINEPREFIX in tmpfs; tmpfs pages
    # are charged to the container's memory cgroup, so the default memory floor is
    # raised and sized to hold the staged prefix.
    _TMPFS = "/tmp:rw,size=384m"
    _DEFAULT_MEMORY = "1g"
    _WIN_ENV = [("HOME", "/tmp"), ("WINEPREFIX", "/tmp/wineprefix"), ("WINEDEBUG", "-all")]

    def __init__(self, image: str = DEFAULT_WIN_IMAGE):
        self.image = image

    def _limits(self, limits: ResourceLimits | None) -> ResourceLimits:
        # Default runs get the larger Wine/tmpfs memory floor; an explicit caller
        # limit is respected as-is.
        return limits or ResourceLimits(memory=self._DEFAULT_MEMORY)

    def _capped(self, argv: list[str], name: str, timeout: float,
                limits: ResourceLimits, stdin_text: str | None = None) -> ExecResult:
        rc, out, err, timed_out, truncated = _run_capped(
            argv, timeout=timeout, output_bytes=limits.output_bytes,
            stdin_text=stdin_text,
            on_timeout=lambda: DockerLinuxBackend._force_remove(name),
        )
        if timed_out:
            return ExecResult(False, "", f"timed out after {timeout:.0f}s", timed_out=True)
        return ExecResult(rc == 0, out, err.strip(), exit_code=rc, truncated=truncated)

    def compile_c(self, workspace, source_rel, out_rel, target, limits=None):
        self._require_target(target)  # pe only
        limits = self._limits(limits)
        (workspace / out_rel).parent.mkdir(parents=True, exist_ok=True)
        src = PurePosixPath(_WORK) / source_rel
        out = PurePosixPath(_WORK) / out_rel
        cc = target.mingw_cross or "x86_64-w64-mingw32-gcc"
        # -static -pthread makes the .exe self-contained: libgcc and winpthread are
        # linked IN, not left as external DLL imports. Without this a threaded PE
        # imports libwinpthread-1.dll, which is absent from the Wine prefix, so it
        # fails to load (status c0000135) — the compatibility study's one failure.
        # It also means generated PE need no MinGW runtime DLLs alongside them.
        inner = [cc, "-O2", "-s", "-static", "-pthread", "-o", str(out), str(src)]
        # Compile needs the workspace writable so the .exe lands back on the host
        # (for the strings-leak check). No Wine here, so the default tmpfs is fine.
        name = f"slothctf-{uuid.uuid4().hex[:12]}"
        argv = build_docker_run_argv(self.image, inner, workspace, limits, name)
        return self._capped(argv, name, limits.compile_timeout, limits)

    def run_binary(self, workspace, binary_rel, target, args=None, stdin_text=None, limits=None):
        self._require_target(target)
        limits = self._limits(limits)
        # Expose ONLY the PE under test, read-only — the rest of the workspace
        # (sources, other binaries, the flag) stays invisible to the Wine process.
        exe_host = workspace / binary_rel
        mounts = [(str(exe_host), _RUN_EXE, "ro")]
        inner = ["winrun", _RUN_EXE, *(args or [])]
        name = f"slothctf-{uuid.uuid4().hex[:12]}"
        argv = build_docker_run_argv(
            self.image, inner, workspace, limits, name,
            stdin=stdin_text is not None, mounts=mounts, tmpfs=self._TMPFS, env=self._WIN_ENV)
        return self._capped(argv, name, limits.run_timeout, limits, stdin_text=stdin_text)

    def run_python(self, workspace, script_rel, args=None, stdin_text=None, limits=None):
        # The Windows-compat image has no Python; Python steps (log genre) run on
        # the Linux backend. Refuse rather than pretend.
        raise SandboxUnavailable(
            "the Windows-compat backend runs PE binaries only; Python steps use the "
            "Linux backend (target 'elf').")


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


def image_available(image: str = DEFAULT_IMAGE) -> bool:
    """True if the sandbox image is already built locally (never auto-pulled)."""
    try:
        r = subprocess.run(["docker", "image", "inspect", image],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False


def win_image_available(image: str = DEFAULT_WIN_IMAGE) -> bool:
    """True if the Windows-compat sandbox image (MinGW + Wine) is built locally."""
    return image_available(image)


def build_sandbox_image(image: str | None = None, context_dir: str = "docker",
                        dockerfile: str = "Dockerfile") -> bool:
    """Build a sandbox image, streaming output. Returns True on success. Raises
    SandboxUnavailable if the docker CLI or the Dockerfile is missing.

    ``dockerfile`` selects which image: the default ELF image, or
    ``Dockerfile.wincompat`` (MinGW + Wine) for the Windows-compat image."""
    if shutil.which("docker") is None:
        raise SandboxUnavailable("docker CLI not found; install Docker Desktop (Windows) or docker (Linux).")
    df = Path(context_dir) / dockerfile
    if not df.exists():
        raise SandboxUnavailable(f"no {dockerfile} under {context_dir}/")
    if image is None:
        image = DEFAULT_WIN_IMAGE if dockerfile.endswith("wincompat") else DEFAULT_IMAGE
    r = subprocess.run(["docker", "build", "-t", image, "-f", str(df), context_dir])  # inherit stdio
    return r.returncode == 0


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
                    allow_native: bool = False, image: str = DEFAULT_IMAGE,
                    win_image: str = DEFAULT_WIN_IMAGE) -> ExecutionBackend:
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
            if not docker_available():
                raise SandboxUnavailable(
                    "Isolated Windows PE needs Docker (MinGW + Wine), which is not "
                    "available. Start Docker Desktop, or use --unsafe-native on a Windows "
                    "host to build the .exe with no isolation. No automatic fallback.")
            if not win_image_available(image=win_image):
                raise SandboxUnavailable(
                    f"Isolated Windows PE needs the Windows-compat image {win_image!r}, "
                    "which is not built. Run `python main.py sandbox-build --win`. SlothCTF "
                    "will not silently build the .exe unsandboxed or convert PE to ELF.")
            return DockerWinCompatBackend(image=win_image)
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
