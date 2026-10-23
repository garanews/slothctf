import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from slothctf import web


class GenerateFormTests(unittest.TestCase):
    def test_safe_name(self):
        self.assertEqual(web._safe_name("my_challenge"), "my_challenge")
        self.assertIsNone(web._safe_name(""))
        self.assertIsNone(web._safe_name("a/b"))
        self.assertIsNone(web._safe_name("..\\x"))
        self.assertIsNone(web._safe_name("x" * 65))

    def test_build_generate_kwargs(self):
        with mock.patch.object(web, "resolve_backend", return_value="BE") as rb:
            k = web._build_generate_kwargs({
                "name": "c1", "genre": "log", "difficulty": "hard", "prompt": "p",
                "model": "m", "ollama_url": "http://h", "max_attempts": "7",
                "category": "forensics", "roundtrip": True, "surprise": False,
            })
        self.assertEqual(k["name"], "c1")
        self.assertEqual(k["genre"], "log")
        self.assertEqual(k["difficulty"], "hard")
        self.assertEqual(k["max_attempts"], 7)
        self.assertEqual(k["challenge_type"], "forensics")
        self.assertFalse(k["roundtrip"])  # roundtrip only applies to reverse
        self.assertEqual(k["target"], "elf")       # default target
        self.assertEqual(k["backend"], "BE")       # resolved backend passed through
        rb.assert_called_once()

    def test_build_generate_kwargs_defaults_and_validation(self):
        with self.assertRaises(ValueError):
            web._build_generate_kwargs({"name": "bad/name"})  # name checked before backend
        with mock.patch.object(web, "resolve_backend", return_value="BE"):
            k = web._build_generate_kwargs({"name": "c2", "genre": "reverse", "roundtrip": True,
                                            "difficulty": "nope", "target": "pe", "unsafe_native": True})
        self.assertEqual(k["genre"], "reverse")
        self.assertTrue(k["roundtrip"])
        self.assertEqual(k["difficulty"], "easy")  # invalid -> default
        self.assertEqual(k["target"], "pe")

    def test_build_generate_kwargs_isolation_choice(self):
        # unsafe_native toggles the isolation argument passed to resolve_backend
        with mock.patch.object(web, "resolve_backend", return_value="BE") as rb:
            web._build_generate_kwargs({"name": "c3", "target": "elf", "unsafe_native": False})
        self.assertEqual(rb.call_args.args[1], "isolated")
        with mock.patch.object(web, "resolve_backend", return_value="BE") as rb:
            web._build_generate_kwargs({"name": "c4", "target": "pe", "unsafe_native": True})
        self.assertEqual(rb.call_args.args[1], "unsafe-native")


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

    def test_search_code(self):
        # the demo challenge's src/challenge.c contains "int main(){}"
        res = web._search_code("int main")
        self.assertTrue(any(r["name"] == "demo" for r in res))
        hit = next(r for r in res if r["name"] == "demo")
        self.assertTrue(any(m["path"] == "src/challenge.c" for m in hit["matches"]))
        self.assertEqual(web._search_code("zzz_not_present"), [])


if __name__ == "__main__":
    unittest.main()
