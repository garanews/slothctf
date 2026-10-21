import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from slothctf import web


class ClassifyTests(unittest.TestCase):
    def test_author_vs_player(self):
        self.assertEqual(web._classify("organizer/encoder.c"), "author")
        self.assertEqual(web._classify("src/challenge.c"), "author")
        self.assertEqual(web._classify("solution.md"), "author")
        self.assertEqual(web._classify("challenge.json"), "author")
        self.assertEqual(web._classify("README.md"), "player")
        self.assertEqual(web._classify("bin/challenge.exe"), "player")
        self.assertEqual(web._classify("bin/encoder.exe"), "author")  # organizer tool
        self.assertEqual(web._classify("evidence/server.log"), "player")


class SafeFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "generated" / "demo" / "src").mkdir(parents=True)
        (root / "generated" / "demo" / "challenge.json").write_text("{}", encoding="utf-8")
        (root / "generated" / "demo" / "src" / "challenge.c").write_text("int main(){}", encoding="utf-8")
        (root / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
        self._patch = mock.patch.object(web, "GENERATED", root / "generated")
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_resolves_legit_file(self):
        p = web._safe_file("demo", "src/challenge.c")
        self.assertIsNotNone(p)
        self.assertTrue(p.read_text().startswith("int main"))

    def test_rejects_parent_traversal(self):
        self.assertIsNone(web._safe_file("demo", "../../secret.txt"))
        self.assertIsNone(web._safe_file("demo", "../secret.txt"))

    def test_rejects_bad_challenge_name(self):
        self.assertIsNone(web._safe_file("../", "challenge.json"))
        self.assertIsNone(web._challenge_dir("../demo"))
        self.assertIsNone(web._challenge_dir("a/b"))

    def test_missing_file(self):
        self.assertIsNone(web._safe_file("demo", "nope.txt"))


if __name__ == "__main__":
    unittest.main()
