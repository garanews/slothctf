"""Reproducible reliability campaign for SlothCTF.

Runs a FROZEN set of scenarios (several repeats each), logging one append-only
JSON record per generation run to runs.jsonl — including failed runs, so the
numbers don't suffer survivorship bias. Then `--report runs.jsonl` aggregates
honest metrics (success rate, first-attempt rate, attempts/time to convergence
on successes only, error distribution by stage, gate rejection rate), always
with numerator/denominator and the spread across repeats.

Methodology (per the review): gates apply to different challenge kinds, so they
are separate OFF/ON comparisons — disabling a gate is NOT "it would have passed",
only "it wasn't evaluated". Freeze code + config for a real campaign.

    python tools/run_campaign.py --model qwen2.5-coder:32b --ollama-url URL --repeats 3
    python tools/run_campaign.py --report runs.jsonl
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slothctf.generator import GenerationError, generate_challenge  # noqa: E402

GATE_STAGES = {"roundtrip", "too-easy"}  # opt-in gate rejections (not base checks)

# FROZEN scenario set — separated by kind so each gate is an OFF/ON comparison.
SCENARIOS: list[dict] = [
    # reverse decode: round-trip OFF / ON
    {"id": "rev_decode_xor", "genre": "reverse", "difficulty": "easy",
     "prompt": "An easy reverse engineering challenge with single-byte XOR that hides a flag and prints it.",
     "gates": {}},
    {"id": "rev_decode_xor_rt", "genre": "reverse", "difficulty": "easy",
     "prompt": "An easy reverse engineering challenge with single-byte XOR that hides a flag and prints it.",
     "gates": {"roundtrip": True}},
    # reverse check: adversary OFF / ON (hard, where guessable secrets matter)
    {"id": "rev_check_pwd", "genre": "reverse", "difficulty": "hard",
     "prompt": "A password checker: read a password on stdin, print Access Granted only if correct; "
               "obfuscate it so strings can't reveal it.",
     "gates": {}},
    {"id": "rev_check_pwd_adv", "genre": "reverse", "difficulty": "hard",
     "prompt": "A password checker: read a password on stdin, print Access Granted only if correct; "
               "obfuscate it so strings can't reveal it.",
     "gates": {"adversary": True}},
    # log analysis: standard validation (anti-hardcode runs in generation)
    {"id": "log_ssh", "genre": "log", "difficulty": "medium",
     "prompt": "A Linux auth.log with an SSH brute-force from one IP, then fail2ban bans it. "
               "Ask 4-6 questions mixing extractive facts and computed aggregates, each unambiguous.",
     "gates": {}},
    {"id": "log_cloudtrail", "genre": "log", "difficulty": "medium",
     "prompt": "An AWS CloudTrail JSON-lines log where a stolen access key calls RunInstances from a new "
               "IP. Ask 4-6 unambiguous questions mixing extractive and computed answers.",
     "gates": {}},
]


def _env() -> dict:
    def _cmd(args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        except Exception:
            return ""
    return {
        "git": _cmd(["git", "rev-parse", "HEAD"]),
        "compiler": _cmd(["gcc", "--version"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def run(args) -> int:
    env = _env()
    out = Path(args.out)
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    scenarios = [s for s in SCENARIOS if not only or s["id"] in only]
    print(f"campaign: {len(scenarios)} scenario(s) x {args.repeats} repeat(s) -> {out}", flush=True)

    with out.open("a", encoding="utf-8") as fh:
        for sc in scenarios:
            for rep in range(1, args.repeats + 1):
                name = f"camp_{sc['id']}_{rep}"
                gates = sc.get("gates", {})
                start = time.monotonic()
                rec = {
                    "run_id": uuid.uuid4().hex, "ts": time.time(), "prompt_id": sc["id"],
                    "genre": sc["genre"], "difficulty": sc["difficulty"], "repeat": rep,
                    "gates": gates, "model": args.model, **{f"env_{k}": v for k, v in env.items()},
                }
                try:
                    kw = {"model": args.model, "max_attempts": args.max_attempts, "genre": sc["genre"],
                          "difficulty": sc["difficulty"], "return_report": True}
                    if args.ollama_url:
                        kw["ollama_url"] = args.ollama_url
                    if gates.get("roundtrip"):
                        kw["roundtrip"] = True
                    if gates.get("adversary"):
                        kw["adversary"] = True
                    rep_obj = generate_challenge(name, prompt=sc["prompt"], **kw)
                    last = rep_obj.attempts[-1] if rep_obj.attempts else None
                    rec.update({
                        "success": rep_obj.success,
                        "attempts_used": rep_obj.attempts_used,
                        "final_stage": last.stage if last else None,
                        "duration_s": round(time.monotonic() - start, 2),
                        "attempts": [{"index": a.index, "ok": a.ok, "stage": a.stage,
                                      "detail": (a.detail or "")[:160]} for a in rep_obj.attempts],
                    })
                    # proposed vs kept tasks for log (parsed from the passing detail "k/n task(s)")
                    if sc["genre"] == "log" and rep_obj.success and last:
                        m = re.search(r"(\d+)/(\d+) task", last.detail or "")
                        if m:
                            rec["tasks_kept"], rec["tasks_proposed"] = int(m.group(1)), int(m.group(2))
                except GenerationError as exc:
                    rec.update({"success": False, "attempts_used": args.max_attempts, "final_stage": "error",
                                "duration_s": round(time.monotonic() - start, 2), "error": str(exc)[:200]})
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                tag = "VALID" if rec.get("success") else f"FAIL({rec.get('final_stage')})"
                print(f"  {sc['id']:22} rep{rep} {tag:16} {rec['duration_s']}s", flush=True)
    print(f"done -> {out}  (aggregate with: python tools/run_campaign.py --report {out})", flush=True)
    return 0


def aggregate(records: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (r.get("genre"), r.get("difficulty"), json.dumps(r.get("gates", {}), sort_keys=True))
        groups[key].append(r)
    stage_counts: dict[str, int] = defaultdict(int)
    out_groups = []
    for (genre, diff, gates), rs in sorted(groups.items()):
        n = len(rs)
        succ = [r for r in rs if r.get("success")]
        first = [r for r in succ if r.get("attempts_used") == 1]
        gate_rej = [r for r in rs if r.get("final_stage") in GATE_STAGES]
        for r in rs:
            for a in r.get("attempts", []):
                stage_counts[a["stage"]] += 1
        att = [r["attempts_used"] for r in succ]
        dur = [r["duration_s"] for r in succ if "duration_s" in r]
        out_groups.append({
            "genre": genre, "difficulty": diff, "gates": json.loads(gates),
            "runs": n, "success": len(succ), "success_rate": round(len(succ) / n, 3) if n else 0,
            "first_attempt_success": len(first),
            "attempts_to_converge": {"mean": round(mean(att), 2), "sd": round(pstdev(att), 2)} if att else None,
            "seconds_to_converge": {"mean": round(mean(dur), 1), "sd": round(pstdev(dur), 1)} if dur else None,
            "gate_rejections": len(gate_rej),
        })
    return {"total_runs": len(records), "groups": out_groups, "stage_frequency": dict(sorted(
        stage_counts.items(), key=lambda kv: -kv[1]))}


def render_report(agg: dict) -> str:
    L = [f"SlothCTF reliability report - {agg['total_runs']} run(s)", ""]
    for g in agg["groups"]:
        gates = ",".join(k for k, v in g["gates"].items() if v) or "none"
        L.append(f"[{g['genre']}/{g['difficulty']} gates={gates}]  "
                 f"success {g['success']}/{g['runs']} ({g['success_rate']*100:.0f}%)  "
                 f"1st-try {g['first_attempt_success']}/{g['success']}  "
                 f"gate-rejects {g['gate_rejections']}")
        if g["attempts_to_converge"]:
            a, s = g["attempts_to_converge"], g["seconds_to_converge"]
            L.append(f"    on successes: {a['mean']}+/-{a['sd']} attempts, {s['mean']}+/-{s['sd']}s")
    L.append("")
    L.append("Failure/stage frequency (all attempts): " +
             ", ".join(f"{k}={v}" for k, v in agg["stage_frequency"].items()))
    L.append("")
    L.append("Note: rates are numerator/denominator; 'to converge' stats use successful runs only. "
             "Disabling a gate means it was not evaluated, not that the run would have passed it.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="SlothCTF reliability campaign")
    ap.add_argument("--report", default=None, metavar="runs.jsonl", help="aggregate an existing log and print")
    ap.add_argument("--model", default="qwen2.5-coder:32b")
    ap.add_argument("--ollama-url", default=None)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default=None, help="comma-separated scenario ids")
    ap.add_argument("--out", default="runs.jsonl")
    args = ap.parse_args()

    if args.report:
        records = [json.loads(line) for line in Path(args.report).read_text(encoding="utf-8").splitlines() if line.strip()]
        print(render_report(aggregate(records)))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
