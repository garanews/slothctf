import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from slothctf.targets import (
    DEFAULT_TARGET,
    PE,
    ELF,
    TARGETS,
    TargetError,
    detect_target_from_binary,
    get_target,
    infer_target,
)

def _elf_bytes(machine: int = 0x3E, cls: int = 2) -> bytes:
    """A minimal but structurally valid ELF header (x86-64 by default)."""
    b = bytearray(64)
    b[0:4] = b"\x7fELF"
    b[4] = cls          # EI_CLASS: 2 = 64-bit
    b[5] = 1            # EI_DATA: little-endian
    b[6] = 1            # EI_VERSION
    b[16:18] = (2).to_bytes(2, "little")        # e_type = ET_EXEC
    b[18:20] = machine.to_bytes(2, "little")    # e_machine
    return bytes(b)


def _pe_bytes(machine: int = 0x8664) -> bytes:
    """A minimal but structurally valid PE: MZ, a PE\\0\\0 at e_lfanew, a machine."""
    b = bytearray(0x50)
    b[0:2] = b"MZ"
    e_lfanew = 0x40
    b[0x3C:0x40] = e_lfanew.to_bytes(4, "little")
    b[e_lfanew:e_lfanew + 4] = b"PE\x00\x00"
    b[e_lfanew + 4:e_lfanew + 6] = machine.to_bytes(2, "little")
    return bytes(b)


ELF_MAGIC = _elf_bytes()
PE_MAGIC = _pe_bytes()


def _write(dirpath: Path, name: str, data: bytes) -> Path:
    (dirpath / "bin").mkdir(parents=True, exist_ok=True)
    p = dirpath / "bin" / name
    p.write_bytes(data)
    return p


class RegistryTests(unittest.TestCase):
    def test_both_targets_first_class(self):
        self.assertEqual(set(TARGETS), {"elf", "pe"})
        self.assertEqual(PE.out_name, "challenge.exe")
        self.assertEqual(ELF.out_name, "challenge")
        self.assertEqual(PE.arch, "x86_64")

    def test_get_target_unknown_errors(self):
        with self.assertRaises(TargetError):
            get_target("wasm")


class DetectTests(unittest.TestCase):
    def test_detect_elf_and_pe_by_content(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertEqual(detect_target_from_binary(_write(d, "challenge", ELF_MAGIC)).name, "elf")
            self.assertEqual(detect_target_from_binary(_write(d, "challenge.exe", PE_MAGIC)).name, "pe")

    def test_extension_does_not_override_content(self):
        # a file NAMED .exe but holding ELF bytes is detected as ELF (content wins)
        with TemporaryDirectory() as tmp:
            p = _write(Path(tmp), "challenge.exe", ELF_MAGIC)
            self.assertEqual(detect_target_from_binary(p).name, "elf")

    def test_unknown_magic_errors(self):
        with TemporaryDirectory() as tmp:
            p = _write(Path(tmp), "challenge", b"garbage!")
            with self.assertRaises(TargetError):
                detect_target_from_binary(p)

    def test_bare_mz_without_pe_signature_is_rejected(self):
        # 'MZ' alone (e.g. a DOS stub or a stray file) is NOT a PE: detection parses
        # the PE header, it does not guess from the leading two bytes or the name.
        with TemporaryDirectory() as tmp:
            p = _write(Path(tmp), "challenge.exe", b"MZ" + b"\x00" * 200)
            with self.assertRaises(TargetError):
                detect_target_from_binary(p)

    def test_wrong_architecture_pe_is_rejected(self):
        with TemporaryDirectory() as tmp:
            p = _write(Path(tmp), "challenge.exe", _pe_bytes(machine=0x014C))  # i386
            with self.assertRaises(TargetError):
                detect_target_from_binary(p)

    def test_wrong_architecture_elf_is_rejected(self):
        with TemporaryDirectory() as tmp:
            p = _write(Path(tmp), "challenge", _elf_bytes(machine=0xB7))  # aarch64
            with self.assertRaises(TargetError):
                detect_target_from_binary(p)

    def test_missing_file_errors(self):
        with self.assertRaises(TargetError):
            detect_target_from_binary(Path("nope/challenge"))


class InferTests(unittest.TestCase):
    def test_manifest_target_trusted(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(infer_target(Path(tmp), {"target": "pe"}).name, "pe")

    def test_detect_when_manifest_missing_target(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write(d, "challenge.exe", PE_MAGIC)
            self.assertEqual(infer_target(d, {}).name, "pe")

    def test_no_binary_and_no_manifest_errors(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(TargetError):
                infer_target(Path(tmp), {})

    def test_disagreeing_binaries_are_ambiguous(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write(d, "challenge", ELF_MAGIC)       # ELF
            _write(d, "challenge.exe", PE_MAGIC)    # PE -> disagreement
            with self.assertRaises(TargetError):
                infer_target(d, {})


if __name__ == "__main__":
    unittest.main()
