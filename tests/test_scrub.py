import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import scrub_runs as sr  # noqa: E402


class ScrubTests(unittest.TestCase):
    def test_flag_body_redacted(self):
        rec = {"attempts": [{"stage": "passed", "detail": "decode verified: sloth{abc123}"}]}
        out = sr.scrub_record(rec)
        self.assertEqual(out["attempts"][0]["detail"], "decode verified: sloth{REDACTED}")

    def test_home_path_redacted_keeps_relative(self):
        rec = {"error": r"File \"C:\Users\alice\Documents\proj\generated\x\gen.py\", line 9"}
        out = sr.scrub_record(rec)
        self.assertNotIn("alice", out["error"])
        self.assertIn("<HOME>", out["error"])
        self.assertIn("generated", out["error"])  # useful relative part survives

    def test_posix_home_redacted(self):
        self.assertEqual(sr._scrub_str("/home/bob/x.py"), "<HOME>/x.py")

    def test_synthetic_answers_preserved(self):
        # RFC1918 IPs and AWS-docs EXAMPLE keys are challenge answers, not secrets.
        s = "answer is '192.168.1.100' and 'AKIAIOSFODNN7EXAMPLE'"
        self.assertEqual(sr._scrub_str(s), s)

    def test_idempotent(self):
        once = sr._scrub_str("sloth{x} at /home/bob/y")
        self.assertEqual(sr._scrub_str(once), once)


if __name__ == "__main__":
    unittest.main()
