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

    @property
    def exe_suffix(self) -> str:
        return ".exe" if self.name == "pe" else ""

    def artifact(self, stem: str) -> str:
        """Filename for an auxiliary binary of this target, e.g. artifact('encoder')
        -> 'encoder.exe' for PE, 'encoder' for ELF."""
        return f"{stem}{self.exe_suffix}"

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


# Machine codes we accept (x86-64 only, matching the Target descriptors).
_ELF_EM_X86_64 = 0x3E          # e_machine in the ELF header
_PE_MACHINE_AMD64 = 0x8664     # Machine in the PE/COFF header
_ELFCLASS64 = 2                # EI_CLASS


def _u16le(data: bytes, off: int) -> int | None:
    return int.from_bytes(data[off:off + 2], "little") if len(data) >= off + 2 else None


def detect_target_from_binary(binary_path: Path) -> Target:
    """Identify a compiled artifact's target by PARSING its header, not just the
    leading magic or the filename.

    - ELF: ``\\x7fELF`` + 64-bit class + e_machine == x86-64.
    - PE:  ``MZ`` + a valid ``PE\\0\\0`` signature at e_lfanew + Machine == AMD64.

    This rejects a stray ``MZ`` that is not a real PE, and a PE/ELF of the wrong
    architecture, instead of mislabelling it. Raises TargetError if the file is
    missing/unreadable, is not a recognised x86-64 ELF or PE, or is ambiguous."""
    try:
        data = Path(binary_path).read_bytes()[:65536]  # header lives near the start
    except OSError as exc:
        raise TargetError(f"cannot read binary to detect target: {exc}") from exc

    is_elf = data[:4] == b"\x7fELF"
    is_pe = False
    if data[:2] == b"MZ" and len(data) >= 0x40:
        e_lfanew = int.from_bytes(data[0x3C:0x40], "little")
        if 0 < e_lfanew <= len(data) - 6 and data[e_lfanew:e_lfanew + 4] == b"PE\x00\x00":
            is_pe = _u16le(data, e_lfanew + 4) == _PE_MACHINE_AMD64

    if is_elf and is_pe:  # shouldn't happen for real files, but never guess
        raise TargetError(f"ambiguous target for {binary_path}: header looks like both ELF and PE")
    if is_elf:
        if data[4:5] != bytes([_ELFCLASS64]) or _u16le(data, 18) != _ELF_EM_X86_64:
            raise TargetError(
                f"{binary_path} is an ELF but not x86-64 (class={data[4:5]!r}, "
                f"e_machine={_u16le(data, 18)!r}); only x86-64 is supported")
        return ELF
    if is_pe:
        return PE
    if data[:2] == b"MZ":
        raise TargetError(
            f"{binary_path} starts with 'MZ' but is not a valid x86-64 PE "
            "(no PE\\0\\0 signature / wrong machine) — refusing to guess from the extension")
    raise TargetError(
        f"cannot determine target of {binary_path}: not a recognised x86-64 ELF or PE")


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
