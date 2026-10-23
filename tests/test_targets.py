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

ELF_MAGIC = b"\x7fELF\x02\x01\x01\x00rest"
PE_MAGIC = b"MZ\x90\x00\x03\x00\x00\x00rest"


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
