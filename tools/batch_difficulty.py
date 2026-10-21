"""Generate the SAME prompt at easy / medium / hard to see the rubric in action.

The difficulty rubric steers the model per level; this runs one theme across all
three levels (reverse genre by default) and prints, for each, the result plus a
few size signals (source lines, whether an anti-debug/VM/cipher hint shows up) so
you can eyeball whether the levels actually differ.

Usage:
    python tools/batch_difficulty.py --model qwen2.5-coder:32b --ollama-url https://your-ollama
    python tools/batch_difficulty.py --prompt "a reverse challenge about a license key check"
    python tools/batch_difficulty.py --genre log --prompt "an nginx access log with an attack"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slothctf.generator import GenerationError, generate_challenge  # noqa: E402

_HARD_SIGNALS = ("ptrace", "isdebuggerpresent", "virtualprotect", "mprotect",
                 "vm", "bytecode", "sbox", "rc4", "tea", "__attribute__", "thread")


def _signals(challenge_dir: Path) -> str:
    src = challenge_dir / "src" / "challenge.c"
    org = challenge_dir / "organizer"
    text = ""
    if src.exists():
        text = src.read_text(encoding="utf-8", errors="replace")
    else:  # log genre: look at generator + extractors
        for p in sorted(org.glob("*.py")) if org.exists() else []:
            text += p.read_text(encoding="utf-8", errors="replace")
    lines = text.count("\n") + 1
    hits = sorted({s for s in _HARD_SIGNALS if s in text.lower()})
    return f"{lines} src lines; signals: {', '.join(hits) or 'none'}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate one prompt at easy/medium/hard")
    ap.add_argument("--model", default="qwen2.5-coder:32b")
    ap.add_argument("--ollama-url", default=None)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--genre", choices=["reverse", "log"], default="reverse")
    ap.add_argument("--prompt", default="a reverse engineering challenge that hides a flag")
    ap.add_argument("--name-prefix", default="diff")
    args = ap.parse_args()

    kwargs = {"model": args.model, "max_attempts": args.max_attempts,
              "genre": args.genre, "return_report": True}
    if args.ollama_url:
        kwargs["ollama_url"] = args.ollama_url

    t0 = time.time()
    for level in ("easy", "medium", "hard"):
        name = f"{args.name_prefix}_{level}"
        start = time.time()
        try:
            rep = generate_challenge(name, prompt=args.prompt, difficulty=level, **kwargs)
            tag = "VALID" if rep.success else f"FAIL({rep.attempts[-1].stage if rep.attempts else '?'})"
            sig = _signals(Path("generated") / name) if rep.success else ""
            print(f"{level:7} {tag:16} attempts={rep.attempts_used} {int(time.time()-start)}s  {sig}", flush=True)
        except GenerationError as exc:
            print(f"{level:7} ERROR {str(exc)[:70]} {int(time.time()-start)}s", flush=True)
    print(f"total {int(time.time()-t0)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
