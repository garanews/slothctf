import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from slothctf import stats


def _write(base: Path, name: str, **meta):
    d = base / name
    (d / "bin").mkdir(parents=True)
    payload = {"title": name, "category": "Reverse Engineering", "difficulty": "easy",
               "flag": "sloth{x}", "challenge_kind": "decode"}
    payload.update(meta)
    (d / "challenge.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


class StatsTests(unittest.TestCase):
    def test_collect(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            d1 = _write(base, "rev1", difficulty="hard",
                        generation={"genre": "reverse", "model": "qwen2.5-coder:32b", "roundtrip": True})
            (d1 / "bin" / "challenge.exe").write_bytes(b"x")  # built
            _write(base, "log1", category="Forensics", difficulty="medium", challenge_kind="log",
                   flag="3 task(s)",
                   tasks=[{"type": "extractive"}, {"type": "computed"}, {"type": "extractive"}],
                   generation={"genre": "log", "model": "qwen2.5-coder:32b"})
            s = stats.collect(base)
            self.assertEqual(s["total"], 2)
            self.assertEqual(s["by_genre"], {"reverse": 1, "log": 1})
            self.assertEqual(s["by_difficulty"], {"hard": 1, "medium": 1})
            self.assertEqual(s["built"], 1)
            self.assertEqual(s["gates"], {"roundtrip": 1})
            self.assertEqual(s["log"]["total_tasks"], 3)
            self.assertEqual(s["log"]["by_type"], {"extractive": 2, "computed": 1})
            self.assertIn("qwen2.5-coder:32b", s["by_model"])
            # render must not crash and mentions the total
            self.assertIn("2 challenge", stats.render(s))

    def test_empty(self):
        with TemporaryDirectory() as tmp:
            s = stats.collect(Path(tmp))
            self.assertEqual(s["total"], 0)
            self.assertEqual(s["built_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
