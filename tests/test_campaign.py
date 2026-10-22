import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import run_campaign as rc  # noqa: E402


def _rec(genre, diff, gates, success, attempts, dur, final_stage, stages):
    return {"genre": genre, "difficulty": diff, "gates": gates, "success": success,
            "attempts_used": attempts, "duration_s": dur, "final_stage": final_stage,
            "attempts": [{"index": i + 1, "ok": (i == len(stages) - 1 and success), "stage": s}
                         for i, s in enumerate(stages)]}


class AggregateTests(unittest.TestCase):
    def test_rates_and_grouping(self):
        records = [
            _rec("reverse", "easy", {}, True, 1, 10.0, "passed", ["passed"]),
            _rec("reverse", "easy", {}, True, 3, 40.0, "passed", ["compile", "compile", "passed"]),
            _rec("reverse", "easy", {}, False, 5, 90.0, "challenge-compile",
                 ["challenge-compile"] * 5),
            # a separate group: adversary gate rejection
            _rec("reverse", "hard", {"adversary": True}, False, 5, 50.0, "too-easy", ["too-easy"]),
        ]
        agg = rc.aggregate(records)
        self.assertEqual(agg["total_runs"], 4)
        easy = next(g for g in agg["groups"] if g["difficulty"] == "easy")
        self.assertEqual((easy["success"], easy["runs"]), (2, 3))
        self.assertEqual(easy["success_rate"], round(2 / 3, 3))
        self.assertEqual(easy["first_attempt_success"], 1)            # only the 1-attempt success
        self.assertEqual(easy["attempts_to_converge"]["mean"], 2.0)   # (1+3)/2 over successes
        hard = next(g for g in agg["groups"] if g["difficulty"] == "hard")
        self.assertEqual(hard["gate_rejections"], 1)                  # too-easy counts as a gate reject
        self.assertIn("compile", agg["stage_frequency"])             # stages aggregated across attempts
        # render must not crash
        self.assertIn("reliability report", rc.render_report(agg))

    def test_empty(self):
        agg = rc.aggregate([])
        self.assertEqual(agg["total_runs"], 0)
        self.assertEqual(agg["groups"], [])


if __name__ == "__main__":
    unittest.main()
