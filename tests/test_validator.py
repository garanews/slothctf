import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from slothctf.validator import validate_challenge


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
            self.assertIn("Validation successful", validate_challenge(d))

    def test_log_validate_mismatch(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "case"
            _make_log_challenge(d, "9.9.9.9", self.GOOD)  # declared answer != extracted
            self.assertIn("answer mismatch", validate_challenge(d))

    def test_log_validate_no_log(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "case"
            _make_log_challenge(d, "10.0.0.9", self.GOOD)
            (d / "evidence" / "case.log").unlink()
            self.assertIn("no evidence log", validate_challenge(d))


if __name__ == "__main__":
    unittest.main()
