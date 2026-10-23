"""Scrub secrets from a campaign runs.jsonl before publishing it as a dataset.

Every generated secret in SlothCTF is a `sloth{...}` flag (reverse decode flags
and check-password secrets alike), which the self-verifying loop echoes verbatim
into the human-readable `attempts[].detail` and `error` strings. Those strings
are the diagnostic value of the dataset, so we keep them — but we redact the
literal flag so the published log never ships a working secret.

    python tools/scrub_runs.py runs.jsonl campaign/baseline.jsonl

The structure of each message is preserved (e.g. "the challenge printed 'X' but
the flag is 'Y'" still shows the off-by-one bug); only the flag body is replaced
with REDACTED. Idempotent: re-running on scrubbed input is a no-op.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# sloth{...} is the only secret shape the generator ever emits.
_FLAG = re.compile(r"sloth\{[^}]*\}")
_REPL = "sloth{REDACTED}"

# Tracebacks from model-written generators can embed absolute paths that leak the
# local username (C:\Users\<name>\..., /home/<name>/..., /Users/<name>/...). We
# collapse the home prefix to <HOME> so the useful relative part (generated\...\
# generator.py) stays while the PII is removed. Synthetic log answers (RFC1918 IPs,
# AWS-docs AKIA...EXAMPLE keys) are NOT secrets and are intentionally preserved.
_HOME = re.compile(r"[A-Za-z]:\\Users\\[^\\/]+|/home/[^/]+|/Users/[^/]+")


def _scrub_str(s: str | None) -> str | None:
    if not s:
        return s
    s = _FLAG.sub(_REPL, s)
    s = _HOME.sub("<HOME>", s)
    return s


def scrub_record(rec: dict) -> dict:
    for a in rec.get("attempts", []):
        if "detail" in a:
            a["detail"] = _scrub_str(a["detail"])
    if "error" in rec:
        rec["error"] = _scrub_str(rec["error"])
    return rec


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python tools/scrub_runs.py <in.jsonl> <out.jsonl>", file=sys.stderr)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = redacted = 0
    with dst.open("w", encoding="utf-8") as fh:
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            before = line
            rec = scrub_record(json.loads(line))
            out = json.dumps(rec)
            n += 1
            if _FLAG.search(before):
                redacted += 1
            fh.write(out + "\n")
    print(f"scrubbed {n} record(s) -> {dst}  ({redacted} contained a flag)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
