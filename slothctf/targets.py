"""Compilation targets for challenge artifacts.

A *target* is the binary format a challenge is built for — independent of the host
OS SlothCTF runs on and of the execution backend that builds it. Phase A ships two
first-class, equal targets:

- ``elf`` — Linux ELF x86-64
- ``pe``  — Windows PE x86-64 (``.exe``)

The descriptor also records the architecture and the compilers that can emit it, so
new targets/arches can be added without touching call sites. A challenge records
its target in ``challenge.json``; for older challenges written before this field
existed, :func:`detect_target_from_binary` recognises the format from the file's
magic bytes (never from the extension alone), and ambiguity is an explicit error.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


class TargetError(ValueError):
    """Raised when a target cannot be determined or is unknown."""


@dataclass(frozen=True)
class Target:
    name: str                       # "elf" | "pe"
    arch: str                       # "x86_64"
    out_name: str                   # artifact filename: "challenge" | "challenge.exe"
    description: str
    magic: tuple[bytes, ...]        # leading-byte signatures, for content detection
    native_compilers: tuple[str, ...]  # host compilers that emit this format directly
    mingw_cross: str = ""           # Linux cross-compiler that emits this (pe only), for Phase B

    def matches_magic(self, head: bytes) -> bool:
        return any(head.startswith(sig) for sig in self.magic)


ELF = Target(
    name="elf", arch="x86_64", out_name="challenge",
    description="Linux ELF x86-64",
    magic=(b"\x7fELF",),
    native_compilers=("gcc", "clang", "cc"),
)

PE = Target(
    name="pe", arch="x86_64", out_name="challenge.exe",
    description="Windows PE x86-64 (.exe)",
    magic=(b"MZ",),
    # On a Windows host, MinGW gcc already emits PE — that is why today's challenges
    # are .exe. x86_64-w64-mingw32-gcc is the Linux cross-compiler used in Phase B.
    native_compilers=("x86_64-w64-mingw32-gcc", "gcc", "clang"),
    mingw_cross="x86_64-w64-mingw32-gcc",
)

TARGETS: dict[str, Target] = {t.name: t for t in (ELF, PE)}
DEFAULT_TARGET = "elf"


def host_native_target() -> Target:
    """The target the host's own default toolchain emits when run as-is: PE on
    Windows (MinGW gcc → .exe), ELF elsewhere. This is the ONLY target the unsafe
    NativeBackend can build and run without cross-compilation / emulation."""
    return PE if sys.platform == "win32" else ELF


def get_target(name: str) -> Target:
    try:
        return TARGETS[name.strip().lower()]
    except (KeyError, AttributeError):
        raise TargetError(f"unknown target {name!r} (known: {', '.join(sorted(TARGETS))})")


def detect_target_from_binary(binary_path: Path) -> Target:
    """Identify a compiled artifact's target from its magic bytes.

    Content-based, not extension-based. Raises TargetError if the file is missing,
    unreadable, or does not match exactly one known format (ambiguous/unknown)."""
    try:
        head = Path(binary_path).read_bytes()[:8]
    except OSError as exc:
        raise TargetError(f"cannot read binary to detect target: {exc}") from exc
    matched = [t for t in TARGETS.values() if t.matches_magic(head)]
    if len(matched) == 1:
        return matched[0]
    if not matched:
        raise TargetError(
            f"cannot determine target of {binary_path}: magic {head[:4]!r} matches no "
            "known format (expected ELF \\x7fELF or PE MZ)")
    raise TargetError(
        f"ambiguous target for {binary_path}: magic matches {[t.name for t in matched]}")


def _artifact_candidates(challenge_dir: Path) -> list[Path]:
    bin_dir = Path(challenge_dir) / "bin"
    names = ["challenge", "challenge.exe"]
    return [bin_dir / n for n in names if (bin_dir / n).exists()]


def infer_target(challenge_dir: Path, meta: dict | None = None) -> Target:
    """Resolve a challenge's target: trust the manifest field when present and
    valid; otherwise detect from the actual compiled binary. Explicit error on
    ambiguity (e.g. both a `challenge` and a `challenge.exe` present that disagree)
    or when no binary is available to inspect."""
    meta = meta or {}
    declared = meta.get("target")
    if declared:
        return get_target(declared)
    candidates = _artifact_candidates(Path(challenge_dir))
    if not candidates:
        raise TargetError(
            f"no 'target' in manifest and no binary under {challenge_dir}/bin to detect it")
    detected = {detect_target_from_binary(p).name for p in candidates}
    if len(detected) != 1:
        raise TargetError(
            f"ambiguous target for {challenge_dir}: binaries disagree ({sorted(detected)})")
    return get_target(next(iter(detected)))
