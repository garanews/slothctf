import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from slothctf import catalog


def _make_challenge(base: Path, name: str, **meta) -> None:
    d = base / name
    (d / "bin").mkdir(parents=True, exist_ok=True)
    payload = {"title": name, "category": "Reverse Engineering", "difficulty": "easy",
               "flag": f"sloth{{{name}}}", "challenge_kind": "decode"}
    payload.update(meta)
    (d / "challenge.json").write_text(json.dumps(payload), encoding="utf-8")


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name) / "generated"
        self.base.mkdir(parents=True)
        self.db = Path(self._tmp.name) / "catalog.db"

    def tearDown(self):
        self._tmp.cleanup()

    def test_index_and_query(self):
        _make_challenge(self.base, "xor_one", difficulty="easy", category="Reverse Engineering")
        _make_challenge(self.base, "rc4_two", difficulty="hard", category="Cryptography",
                        challenge_kind="check", success_marker="Access Granted")
        n = catalog.index_generated(self.base, self.db)
        self.assertEqual(n, 2)

        allrows = catalog.query(db_path=self.db)
        self.assertEqual(len(allrows), 2)

        hard = catalog.query(difficulty="hard", db_path=self.db)
        self.assertEqual([r["name"] for r in hard], ["rc4_two"])

        checks = catalog.query(kind="check", db_path=self.db)
        self.assertEqual(checks[0]["success_marker"], "Access Granted")

        crypto = catalog.query(category="cryptography", db_path=self.db)  # case-insensitive
        self.assertEqual(len(crypto), 1)

        found = catalog.query(search="xor", db_path=self.db)
        self.assertEqual(found[0]["name"], "xor_one")

    def test_reindex_prunes_deleted_challenges(self):
        import shutil
        _make_challenge(self.base, "keep")
        _make_challenge(self.base, "gone")
        catalog.index_generated(self.base, self.db)
        self.assertEqual(len(catalog.query(db_path=self.db)), 2)
        shutil.rmtree(self.base / "gone")  # user deletes a challenge folder
        catalog.index_generated(self.base, self.db)
        names = [r["name"] for r in catalog.query(db_path=self.db)]
        self.assertEqual(names, ["keep"])  # the deleted one is pruned

    def test_reindex_is_idempotent_and_keeps_valid(self):
        _make_challenge(self.base, "c1")
        catalog.index_generated(self.base, self.db)
        catalog.record(self.base / "c1", valid=True, db_path=self.db)
        # a plain reindex must not wipe a known-good verdict
        catalog.index_generated(self.base, self.db)
        rows = catalog.query(db_path=self.db)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["valid"], 1)

    def test_sort(self):
        _make_challenge(self.base, "bravo", difficulty="hard")
        _make_challenge(self.base, "alpha", difficulty="easy")
        _make_challenge(self.base, "charlie", difficulty="medium")
        catalog.index_generated(self.base, self.db)
        by_name = [r["name"] for r in catalog.query(sort="name", db_path=self.db)]
        self.assertEqual(by_name, ["alpha", "bravo", "charlie"])
        by_diff = [r["difficulty"] for r in catalog.query(sort="difficulty", db_path=self.db)]
        self.assertEqual(by_diff, ["easy", "medium", "hard"])

    def test_model_column_and_filter(self):
        import json
        d = self.base / "withmodel"
        (d / "bin").mkdir(parents=True)
        (d / "challenge.json").write_text(json.dumps({
            "title": "m", "category": "Reverse Engineering", "difficulty": "easy",
            "flag": "sloth{x}", "challenge_kind": "decode",
            "generation": {"model": "qwen2.5-coder:32b"},
        }), encoding="utf-8")
        _make_challenge(self.base, "nomodel")
        catalog.index_generated(self.base, self.db)
        rows = catalog.query(model="qwen2.5-coder:32b", db_path=self.db)
        self.assertEqual([r["name"] for r in rows], ["withmodel"])
        self.assertIn("qwen2.5-coder:32b", catalog.facets(self.db)["model"])

    def test_facets(self):
        _make_challenge(self.base, "a", category="Reverse Engineering", difficulty="easy")
        _make_challenge(self.base, "b", category="Cryptography", difficulty="hard", challenge_kind="check")
        catalog.index_generated(self.base, self.db)
        f = catalog.facets(self.db)
        self.assertIn("Cryptography", f["category"])
        self.assertIn("hard", f["difficulty"])
        self.assertIn("check", f["kind"])


if __name__ == "__main__":
    unittest.main()
