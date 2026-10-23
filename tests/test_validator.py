import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from slothctf import validator
from slothctf.execution import NativeBackend
from slothctf.targets import host_native_target
from slothctf.validator import _flag_leaks, validate_challenge

_HAS_GCC = validator._find_compiler() is not None


def _nb():
    """Explicit unsafe native backend: re-validate on the host (dev/test use)."""
    return NativeBackend()


def _make_reverse(base: Path, challenge_c: str, flag: str):
    (base / "src").mkdir(parents=True)
    (base / "src" / "challenge.c").write_text(challenge_c, encoding="utf-8")
    (base / "README.md").write_text("r", encoding="utf-8")
    (base / "solution.md").write_text("s", encoding="utf-8")
    (base / "challenge.json").write_text(json.dumps({
        "title": "t", "category": "Reverse Engineering", "difficulty": "easy",
        "flag": flag, "challenge_kind": "decode",
        "target": host_native_target().name,  # built with the host toolchain in tests
    }), encoding="utf-8")


class LeakCheckTests(unittest.TestCase):
    def test_flag_leaks(self):
        with TemporaryDirectory() as tmp:
            b = Path(tmp) / "bin"
            b.write_bytes(b"\x00\x01sloth{visible_secret}\x02padding")
            self.assertTrue(_flag_leaks(b, "sloth{visible_secret}"))
            self.assertTrue(_flag_leaks(b, "visible_secret"))  # inner secret too
        with TemporaryDirectory() as tmp:
            b = Path(tmp) / "bin"
            b.write_bytes(b"\xde\xad\xbe\xef only encoded bytes here")
            self.assertFalse(_flag_leaks(b, "sloth{hidden}"))

    def test_flag_leaks_raises_when_unreadable(self):
        with TemporaryDirectory() as tmp:
            self.assertRaises(OSError, _flag_leaks, Path(tmp) / "nope", "sloth{x}")

    @unittest.skipUnless(_HAS_GCC, "needs a C compiler")
    def test_validate_full_path_catches_leak(self):
        # A binary that embeds the plaintext flag as a literal must be rejected
        # by the whole validate_challenge pipeline, not just the helper.
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "leaky"
            _make_reverse(d, '#include <stdio.h>\nint main(){printf("sloth{leaky}\\n");}\n', "sloth{leaky}")
            r = validate_challenge(d, backend=_nb())
            self.assertFalse(r.success)
            self.assertEqual(r.stage, "flag-leak")

    @unittest.skipUnless(_HAS_GCC, "needs a C compiler")
    def test_validate_full_path_passes_clean(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "clean"
            flag = "sloth{easy_reverse}"
            arr = ",".join(str(ord(c) ^ 0x42) for c in flag)  # bytes derived from the flag
            src = (f"#include <stdio.h>\nint main(){{unsigned char d[]={{{arr}}};"
                   f"for(int i=0;i<{len(flag)};i++)putchar(d[i]^0x42);putchar(10);}}\n")
            _make_reverse(d, src, flag)
            r = validate_challenge(d, backend=_nb())
            self.assertTrue(r.success, str(r))


def _make_log_challenge(base: Path, answer: str, extractor: str):
    (base / "evidence").mkdir(parents=True)
    (base / "organizer").mkdir(parents=True)
    (base / "evidence" / f"{base.name}.log").write_text(
        "noise\nALERT from 10.0.0.9\nmore noise\n", encoding="utf-8")
    (base / "organizer" / "extractor_1.py").write_text(extractor, encoding="utf-8")
    (base / "README.md").write_text("r", encoding="utf-8")
    (base / "solution.md").write_text("s", encoding="utf-8")
    (base / "challenge.json").write_text(json.dumps({
        "title": "t", "category": "Forensics", "difficulty": "easy",
        "flag": "1 task(s)", "challenge_kind": "log",
        "tasks": [{"question": "ip?", "answer": answer, "type": "extractive"}],
    }), encoding="utf-8")


class ValidateLogTests(unittest.TestCase):
    GOOD = ("import sys,re\n"
            "print(re.search(r'ALERT from (\\S+)', open(sys.argv[1]).read()).group(1))\n")

    def test_log_validate_success(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "case"
            _make_log_challenge(d, "10.0.0.9", self.GOOD)
            r = validate_challenge(d, backend=_nb())
            self.assertTrue(r.success)
            self.assertEqual(r.stage, "ok")

    def test_log_validate_mismatch(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "case"
            _make_log_challenge(d, "9.9.9.9", self.GOOD)  # declared answer != extracted
            r = validate_challenge(d, backend=_nb())
            self.assertFalse(r.success)
            self.assertEqual(r.stage, "answer-mismatch")

    def test_log_validate_no_log(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "case"
            _make_log_challenge(d, "10.0.0.9", self.GOOD)
            (d / "evidence" / "case.log").unlink()
            r = validate_challenge(d, backend=_nb())
            self.assertFalse(r.success)
            self.assertEqual(r.stage, "no-log")


if __name__ == "__main__":
    unittest.main()
