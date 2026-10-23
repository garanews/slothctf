"""Offline-capable measurement campaign for the `recover` genre.

This is a MEASUREMENT tool, not another development cycle: it drives the existing
generator/verifiers/backends unchanged and records what happens. It answers three
questions about autonomous Qwen generations: does it work (technical success
rate), does it produce DIFFERENT challenges (transform variety), and are they
interesting (difficulty / obvious shortcuts). Gameplay difficulty is NOT scored by
an LLM — a human playtest scorecard (tools/recover_campaign/playtest_scorecard.md)
captures that separately.

Design notes:
- Every run (success, model reject, infra error, timeout) is appended to a JSONL
  dataset. The runner is RESUMABLE: on restart it skips run_ids already recorded,
  so an interrupted campaign neither duplicates nor loses results.
- Heuristic classifications (transform family, "obvious shortcut") are labelled as
  heuristics. A challenge is NOT called hard just because the naive attacker fails.
- Real generations need Ollama (the A100). For offline development the runner takes
  a `caller_factory` that replays pre-recorded responses (and can simulate model
  failures, timeouts and interruptions). Docker is still used to really compile/run.

Usage:
  python tools/run_recover_campaign.py run --scenarios tools/recover_campaign/scenarios.json \
      --out-dir campaign_recover/pilot --jsonl campaign_recover/pilot/raw.jsonl \
      [--ollama-url URL] [--limit N]
  python tools/run_recover_campaign.py aggregate --jsonl raw.jsonl --md report.md --csv report.csv
  python tools/run_recover_campaign.py anonymize --jsonl raw.jsonl --out anon.jsonl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slothctf.execution import (  # noqa: E402
    DockerLinuxBackend, DockerWinCompatBackend, SandboxUnavailable,
)
from slothctf.generator import (  # noqa: E402
    GenerationError, _flag_leaks_recover, generate_challenge,
)

TOOL_VERSION = "1.0"

# Stages that mean "infrastructure, not the model's content" (vs the model
# producing something that failed a functional/solver check).
_INFRA_STAGES = {"model-error", "solver-error", "swap-solver-error", "challenge-timeout"}


# --------------------------------------------------------------------------- #
# Provenance (recorded with every run so a campaign is reproducible)
# --------------------------------------------------------------------------- #
def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _ollama_base(ollama_url: str) -> str:
    base = ollama_url.rstrip("/")
    if base.endswith("/api/generate"):
        base = base[: -len("/api/generate")]
    return base


def _ollama_model_digest(model: str, ollama_url: str) -> str | None:
    """Best effort: the model's digest from Ollama. Tries /api/show, then falls back
    to /api/tags (matching the model name). None when offline/unavailable."""
    from urllib import request
    base = _ollama_base(ollama_url)
    try:
        req = request.Request(base + "/api/show", data=json.dumps({"name": model}).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        dig = (data.get("details", {}) or {}).get("digest") or data.get("digest")
        if dig:
            return dig
    except Exception:
        pass
    try:  # fallback: /api/tags lists local models with their digests
        with request.urlopen(request.Request(base + "/api/tags"), timeout=10) as resp:
            tags = json.loads(resp.read().decode()).get("models", [])
        for m in tags:
            if m.get("name") == model or m.get("model") == model:
                return m.get("digest")
    except Exception:
        pass
    return None


def _docker_image_id(image: str) -> str | None:
    """Best effort local image id (digest) for provenance; None if docker/image absent."""
    try:
        r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def provenance(model: str, ollama_url: str, online: bool) -> dict:
    return {
        "tool_version": TOOL_VERSION,
        "git_commit": _git_commit(),
        "model": model,
        "model_digest": _ollama_model_digest(model, ollama_url) if online else None,
        "inference": {"temperature": 0.7, "num_ctx": 4096},  # fixed in generator._call_ollama
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
def load_campaign(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def campaign_fingerprint(campaign: dict, prov: dict) -> str:
    """A stable hash of everything that defines the campaign: scenarios + prompts,
    model, reps, max_attempts, inference params, tool version and code commit. On
    resume a different fingerprint means a changed config/code and the runner
    refuses to append to the same dataset."""
    payload = {
        "scenarios": [{k: sc.get(k) for k in ("id", "target", "difficulty", "prompt")}
                      for sc in campaign.get("scenarios", [])],
        "model": prov.get("model"),
        "reps": campaign.get("reps"),
        "max_attempts": campaign.get("max_attempts"),
        "inference": prov.get("inference"),
        "tool_version": prov.get("tool_version"),
        "git_commit": prov.get("git_commit"),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _existing_fingerprint(jsonl_path: str | Path) -> str | None:
    """The campaign_fp of the first readable record, or None for an empty/absent set."""
    p = Path(jsonl_path)
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if "campaign_fp" in rec:
            return rec["campaign_fp"]
    return None


def run_specs(campaign: dict) -> list[dict]:
    """Expand scenarios x reps into concrete run specs with stable run_ids."""
    reps = int(campaign.get("reps", 1))
    specs = []
    for sc in campaign["scenarios"]:
        for rep in range(1, reps + 1):
            specs.append({"run_id": f"{sc['id']}#rep{rep}", "scenario": sc, "rep": rep})
    return specs


# --------------------------------------------------------------------------- #
# Append-only JSONL + resume
# --------------------------------------------------------------------------- #
def completed_run_ids(jsonl_path: str | Path) -> set[str]:
    p = Path(jsonl_path)
    done: set[str] = set()
    if not p.exists():
        return done
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["run_id"])
        except Exception:
            continue  # ignore a torn final line from a hard interruption
    return done


def append_record(jsonl_path: str | Path, record: dict) -> None:
    p = Path(jsonl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def classify(record: dict) -> str:
    if record.get("success"):
        return "ok"
    if record.get("exception"):
        return "infra"
    stage = record.get("fail_stage") or ""
    return "infra" if stage in _INFRA_STAGES else "model-reject"


# --------------------------------------------------------------------------- #
# Heuristic quality analysis (labelled as heuristics — NOT ground truth)
# --------------------------------------------------------------------------- #
_TRANSFORM_PATTERNS = {
    "xor": r"\^|\bxor\b",
    "add_sub": r"[^+]\+[^+=]|\bADD\b|\- 0x|\bsub\b",
    "rotate": r"<<|>>|\brot[lr]?\b",
    "sbox_table": r"\[\s*256\s*\]|\bsbox\b|\bS\[|\btable\b",
    "named_cipher": r"\bfnv\b|\brc4\b|\btea\b|\baes\b|\bxtea\b|\bchacha\b",
    "position_dependent": r"\[\s*i\s*\]|\+\s*i\b|\^\s*\(?\s*0x[0-9a-fA-F]+\s*\+\s*i|\bi\s*\+\s*1",
}


def analyze_transform(challenge_dir: Path) -> dict:
    """Tag the obfuscation 'family' from the organizer encoder+solver source. This
    is a coarse heuristic for VARIETY, not a semantic classification."""
    text = ""
    for rel in ("organizer/encoder.c", "organizer/solver.py", "src/challenge.c"):
        p = challenge_dir / rel
        if p.exists():
            text += p.read_text(encoding="utf-8", errors="replace") + "\n"
    tags = sorted(k for k, pat in _TRANSFORM_PATTERNS.items() if re.search(pat, text))
    return {"families": tags, "family_signature": "+".join(tags) or "unknown"}


def _shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    return round(-sum((c / n) * math.log2(c / n) for c in counts if c), 3)


def naive_attacker(blob: bytes, expected: str) -> dict:
    """A deliberately WEAK attacker: try every single-byte XOR and every single-byte
    ADD over the whole binary. A shortcut counts ONLY when the EXACT expected flag is
    recovered — a stray `sloth{...}` that is not the real flag is recorded as a
    `candidate` but is NOT a shortcut (it would be a false positive). Failing to
    recover the flag does NOT mean the challenge is hard — it only rules out the
    laziest attack."""
    pat = re.compile(rb"sloth\{[ -~]{1,62}\}")
    exp = expected.encode("utf-8")
    candidate = None
    for k in range(256):
        for op, x in (("xor", bytes(b ^ k for b in blob)),
                      ("add", bytes((b - k) & 0xFF for b in blob))):
            if exp and exp in x:
                return {"recovered_expected": True, "op": op, "key": k}
            if candidate is None:
                m = pat.search(x)
                if m:
                    candidate = {"op": op, "key": k, "text": m.group().decode("latin1")}
    return {"recovered_expected": False, "op": None, "key": None, "candidate": candidate}


def analyze_quality(challenge_dir: Path, flag: str, artifact_rel: str) -> dict:
    out = {"heuristic": True}
    out.update(analyze_transform(challenge_dir))
    binp = challenge_dir / artifact_rel
    if binp.exists():
        blob = binp.read_bytes()
        out["binary_size"] = len(blob)
        out["binary_entropy"] = _shannon_entropy(blob)
        try:
            out["strings_shortcut"] = bool(_flag_leaks_recover(binp, flag))
        except OSError:
            out["strings_shortcut"] = None
        na = naive_attacker(blob, flag)
        out["naive_attacker_shortcut"] = bool(na.get("recovered_expected"))
        out["naive_attacker"] = na
    else:
        out["binary_present"] = False
    return out


# --------------------------------------------------------------------------- #
# Running one scenario
# --------------------------------------------------------------------------- #
def _backend_for(target: str):
    return DockerWinCompatBackend() if target == "pe" else DockerLinuxBackend()


def _artifact_rel(target: str) -> str:
    return "bin/challenge.exe" if target == "pe" else "bin/challenge"


def run_one(spec: dict, out_dir: Path, *, caller, model: str, ollama_url: str,
            max_attempts: int, prov: dict) -> dict:
    """Generate one recover challenge and return its record. Never raises for a
    model/verification failure; only genuine infrastructure errors are caught and
    recorded as such (category 'infra')."""
    sc = spec["scenario"]
    run_dir = Path(out_dir) / spec["run_id"].replace("#", "_")
    run_dir.mkdir(parents=True, exist_ok=True)
    name = "ch_" + spec["run_id"].replace("#", "_").replace("-", "_")
    backend = _backend_for(sc["target"])
    record = {
        "run_id": spec["run_id"], "scenario": sc["id"], "target": sc["target"],
        "difficulty": sc["difficulty"], "rep": spec["rep"],
        "provenance": prov, "timestamp": time.time(),
        "backend": backend.name, "image": getattr(backend, "image", None),
        "image_id": _docker_image_id(getattr(backend, "image", "")),
    }
    prev = os.getcwd()
    os.chdir(run_dir)
    t0 = time.monotonic()
    try:
        report = generate_challenge(
            name, prompt=sc["prompt"], genre="recover", difficulty=sc["difficulty"],
            target=sc["target"], backend=backend,
            model=model, ollama_url=ollama_url, caller=caller,
            max_attempts=max_attempts, return_report=True)
        record["success"] = bool(report.success)
        record["attempts_used"] = report.attempts_used
        record["stages"] = [a.stage for a in report.attempts]
        record["fail_stage"] = None if report.success else (report.attempts[-1].stage if report.attempts else "no-attempt")
        record["detail"] = "" if report.success else (report.attempts[-1].detail[:300] if report.attempts else "")
    except (KeyboardInterrupt, SystemExit):
        raise  # an interruption must propagate so the run is NOT recorded (resume re-does it)
    except SandboxUnavailable as exc:
        record.update(success=False, exception="SandboxUnavailable", fail_stage="infra", detail=str(exc)[:300])
    except Exception as exc:  # any other infra failure
        record.update(success=False, exception=type(exc).__name__, fail_stage="infra", detail=str(exc)[:300])
    finally:
        record["total_time_s"] = round(time.monotonic() - t0, 2)
        os.chdir(prev)

    # On success, read the written manifest and analyse quality (heuristic).
    if record.get("success"):
        cdir = run_dir / "generated" / name
        try:
            meta = json.loads((cdir / "challenge.json").read_text(encoding="utf-8"))
            record["flag"] = meta.get("flag", "")
            record["success_marker"] = meta.get("success_marker", "")
            record["quality"] = analyze_quality(cdir, record["flag"], _artifact_rel(sc["target"]))
        except Exception as exc:
            record["quality_error"] = f"{type(exc).__name__}: {exc}"
    record["category"] = classify(record)
    return record


# --------------------------------------------------------------------------- #
# Campaign driver (resumable)
# --------------------------------------------------------------------------- #
def run_campaign(campaign: dict, out_dir: str | Path, jsonl_path: str | Path, *,
                 caller_factory=None, model: str | None = None, ollama_url: str = "http://localhost:11434",
                 limit: int | None = None, online: bool | None = None, log=print) -> list[dict]:
    """Run (or resume) a campaign, appending one JSONL record per completed run.

    caller_factory: None -> real Ollama. Otherwise a callable(spec) -> caller used
    to REPLAY pre-recorded responses offline. `online` defaults to (caller_factory
    is None)."""
    model = model or campaign.get("model", "qwen2.5-coder:32b")
    max_attempts = int(campaign.get("max_attempts", 3))
    online = (caller_factory is None) if online is None else online
    prov = provenance(model, ollama_url, online)
    fp = campaign_fingerprint(campaign, prov)

    existing_fp = _existing_fingerprint(jsonl_path)
    if existing_fp is not None and existing_fp != fp:
        raise RuntimeError(
            f"campaign fingerprint mismatch: the dataset {jsonl_path} was produced with a "
            f"different configuration/code (existing {existing_fp}, current {fp}). Refusing to "
            "mix configurations in one dataset — start a new JSONL (and bump scenarios 'version').")

    done = completed_run_ids(jsonl_path)
    specs = [s for s in run_specs(campaign) if s["run_id"] not in done]
    if limit is not None:
        specs = specs[:limit]
    log(f"campaign '{campaign.get('name')}' : {len(done)} already done, {len(specs)} to run")

    produced = []
    for spec in specs:
        log(f"  run {spec['run_id']} ...")
        caller = caller_factory(spec) if caller_factory else None
        record = run_one(spec, Path(out_dir), caller=caller, model=model,
                         ollama_url=ollama_url, max_attempts=max_attempts, prov=prov)
        record["campaign_fp"] = fp
        append_record(jsonl_path, record)   # recorded only after it completed
        produced.append(record)
        log(f"    -> {record['category']} ({record.get('fail_stage') or 'passed'}, "
            f"{record['total_time_s']}s)")
    return produced


def replay_caller_factory(responses: dict):
    """Build a caller_factory that replays canned responses per run_id for offline
    tests. responses[run_id] is a list; each element is a dict (returned) or a
    control dict {"__raise__": "GenerationError"|"<ExcName>", "msg": ...}. The last
    element is reused for any extra repair attempts."""
    def factory(spec):
        items = list(responses.get(spec["run_id"], responses.get("*", [{}])))
        state = {"i": 0}

        def caller(prompt, model):
            item = items[min(state["i"], len(items) - 1)]
            state["i"] += 1
            if isinstance(item, dict) and "__raise__" in item:
                kind = item["__raise__"]
                if kind == "GenerationError":
                    raise GenerationError(item.get("msg", "simulated model error"))
                raise RuntimeError(item.get("msg", f"simulated {kind}"))
            return dict(item)
        return caller
    return factory


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _rate(num: int, den: int) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


def aggregate(records: list[dict]) -> dict:
    total = len(records)
    ok = [r for r in records if r.get("success")]
    by_cat: dict[str, int] = {}
    by_target: dict[str, list[int]] = {}
    by_diff: dict[str, list[int]] = {}
    for r in records:
        by_cat[r.get("category", "?")] = by_cat.get(r.get("category", "?"), 0) + 1
        by_target.setdefault(r.get("target", "?"), [0, 0])
        by_diff.setdefault(r.get("difficulty", "?"), [0, 0])
        by_target[r["target"]][1] += 1
        by_diff[r["difficulty"]][1] += 1
        if r.get("success"):
            by_target[r["target"]][0] += 1
            by_diff[r["difficulty"]][0] += 1
    per_scenario: dict[str, list[int]] = {}
    fail_stage_hist: dict[str, int] = {}
    attempts_hist: dict[str, int] = {}
    for r in records:
        sid = r.get("scenario", "?")
        per_scenario.setdefault(sid, [0, 0])
        per_scenario[sid][1] += 1
        if r.get("success"):
            per_scenario[sid][0] += 1
        else:
            fs = r.get("fail_stage") or "unknown"
            fail_stage_hist[fs] = fail_stage_hist.get(fs, 0) + 1
        if "attempts_used" in r:
            key = str(r["attempts_used"])
            attempts_hist[key] = attempts_hist.get(key, 0) + 1
    fam_hist: dict[str, int] = {}
    strings_short = naive_short = 0
    for r in ok:
        q = r.get("quality", {})
        fam_hist[q.get("family_signature", "unknown")] = fam_hist.get(q.get("family_signature", "unknown"), 0) + 1
        if q.get("strings_shortcut"):
            strings_short += 1
        if q.get("naive_attacker_shortcut"):
            naive_short += 1
    attempts = [r.get("attempts_used", 0) for r in records if "attempts_used" in r]
    times = [r.get("total_time_s", 0) for r in records]
    return {
        "total_runs": total,
        "success_rate_pct": _rate(len(ok), total),
        "category_counts": by_cat,
        "per_scenario": {s: {"ok": v[0], "total": v[1], "rate_pct": _rate(v[0], v[1])}
                         for s, v in sorted(per_scenario.items())},
        "success_by_target": {t: _rate(v[0], v[1]) for t, v in sorted(by_target.items())},
        "success_by_difficulty": {d: _rate(v[0], v[1]) for d, v in sorted(by_diff.items())},
        "fail_stage_histogram": dict(sorted(fail_stage_hist.items(), key=lambda kv: -kv[1])),
        "attempts_histogram": dict(sorted(attempts_hist.items())),
        "transform_family_histogram": dict(sorted(fam_hist.items(), key=lambda kv: -kv[1])),
        "distinct_family_signatures": len(fam_hist),
        "variety_ratio": round(len(fam_hist) / len(ok), 3) if ok else 0.0,
        "obvious_shortcuts": {"strings": strings_short, "naive_single_op": naive_short,
                              "of_successful": len(ok)},
        "mean_attempts": round(sum(attempts) / len(attempts), 2) if attempts else 0.0,
        "mean_time_s": round(sum(times) / len(times), 2) if times else 0.0,
    }


def render_markdown(summary: dict) -> str:
    L = ["# recover campaign — aggregate report", "",
         f"Runs: **{summary['total_runs']}**  ·  technical success rate: "
         f"**{summary['success_rate_pct']}%**", "",
         "---", "", "# Technical metrics", "", "## Outcomes", ""]
    for k, v in summary["category_counts"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## Success per scenario (target × difficulty)", "",
          "| scenario | ok / total | rate |", "|---|---|---|"]
    for s, v in summary["per_scenario"].items():
        L.append(f"| {s} | {v['ok']}/{v['total']} | {v['rate_pct']}% |")
    L += ["", "## Success by target", ""]
    for t, r in summary["success_by_target"].items():
        L.append(f"- {t}: {r}%")
    L += ["", "## Success by difficulty", ""]
    for d, r in summary["success_by_difficulty"].items():
        L.append(f"- {d}: {r}%")
    L += ["", "## Failure distribution by stage", ""]
    if summary["fail_stage_histogram"]:
        L += ["| stage | count |", "|---|---|"]
        for stg, c in summary["fail_stage_histogram"].items():
            L.append(f"| {stg} | {c} |")
    else:
        L.append("- (no failures)")
    L += ["", "## Attempts used (distribution)", ""]
    for a, c in summary["attempts_histogram"].items():
        L.append(f"- {a} attempt(s): {c}")
    L += [f"", f"Mean attempts: {summary['mean_attempts']}  ·  mean time: {summary['mean_time_s']}s", "",
          "---", "", "# Heuristics (NOT ground truth)", "",
          "> A challenge is not 'hard' just because the naive attacker failed; these are "
          "signals only. Real difficulty/interest comes from the human playtest scorecard.", "",
          "## Transform variety (heuristic)", "",
          f"Distinct family signatures: **{summary['distinct_family_signatures']}** over "
          f"{summary['obvious_shortcuts']['of_successful']} successful "
          f"(variety ratio {summary['variety_ratio']}).", "",
          "| family signature | count |", "|---|---|"]
    for fam, c in summary["transform_family_histogram"].items():
        L.append(f"| {fam} | {c} |")
    sc = summary["obvious_shortcuts"]
    L += ["", "## Obvious shortcuts (heuristic)", "",
          f"- recoverable by `strings`: {sc['strings']}/{sc['of_successful']}",
          f"- recoverable by a single-byte XOR/ADD sweep (exact flag): "
          f"{sc['naive_single_op']}/{sc['of_successful']}", ""]
    return "\n".join(L)


_CSV_FIELDS = ["run_id", "scenario", "target", "difficulty", "rep", "category", "success",
               "fail_stage", "attempts_used", "total_time_s", "family_signature",
               "binary_size", "binary_entropy", "strings_shortcut", "naive_attacker_shortcut"]


def write_csv(records: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in records:
            q = r.get("quality", {})
            w.writerow({
                "run_id": r.get("run_id"), "scenario": r.get("scenario"), "target": r.get("target"),
                "difficulty": r.get("difficulty"), "rep": r.get("rep"), "category": r.get("category"),
                "success": r.get("success"), "fail_stage": r.get("fail_stage"),
                "attempts_used": r.get("attempts_used"), "total_time_s": r.get("total_time_s"),
                "family_signature": q.get("family_signature"), "binary_size": q.get("binary_size"),
                "binary_entropy": q.get("binary_entropy"), "strings_shortcut": q.get("strings_shortcut"),
                "naive_attacker_shortcut": q.get("naive_attacker_shortcut"),
            })


# --------------------------------------------------------------------------- #
# Anonymisation for publication (never alters the original dataset)
# --------------------------------------------------------------------------- #
_WIN_PATH_RE = re.compile(r"[A-Za-z]:\\\\[^\s\"']+|[A-Za-z]:\\[^\s\"']+")


def _scrub_text(s: str, secrets: list[str]) -> str:
    """Redact known secrets (flag + inner) and local Windows paths from a string."""
    for sec in secrets:
        if sec and len(sec) >= 3:
            s = s.replace(sec, "<flag>")
    return _WIN_PATH_RE.sub("<path>", s)


def _scrub(obj, secrets: list[str]):
    if isinstance(obj, str):
        return _scrub_text(obj, secrets)
    if isinstance(obj, list):
        return [_scrub(x, secrets) for x in obj]
    if isinstance(obj, dict):
        return {k: _scrub(v, secrets) for k, v in obj.items()}
    return obj


def anonymize(records: list[dict]) -> list[dict]:
    """Return publication-safe copies: the plaintext flag is hashed, and the flag
    (and its inner secret) plus any local Windows path are redacted from EVERY
    string field — including detail, quality_error and the naive-attacker candidate —
    not just the top-level `flag`. Host details are dropped. Metrics and categories
    are preserved. The original dataset is never mutated."""
    out = []
    for r in records:
        c = json.loads(json.dumps(r))  # deep copy; original untouched
        flag = r.get("flag", "")
        secrets = [flag]
        m = re.match(r"sloth\{(.*)\}$", flag or "")
        if m and m.group(1):
            secrets.append(m.group(1))
        c = _scrub(c, secrets)  # redact flag/paths everywhere first
        if flag:
            c["flag"] = "sha256:" + hashlib.sha256(flag.encode()).hexdigest()[:12]  # then hash the field
        c.pop("directory", None)
        prov = c.get("provenance", {})
        if isinstance(prov, dict):
            prov.pop("platform", None)  # host detail
        out.append(c)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _read_records(jsonl_path: str | Path) -> list[dict]:
    """Read all valid JSONL records, skipping blank lines and a torn final line
    (a hard interruption can leave one) without losing earlier valid records."""
    recs = []
    lines = [l for l in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, line in enumerate(lines):
        try:
            recs.append(json.loads(line))
        except Exception:
            if i == len(lines) - 1:
                break  # tolerate only a damaged LAST line
            raise ValueError(f"corrupt JSONL record at line {i + 1} (not the last line)")
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description="recover genre measurement campaign")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run or resume a campaign (real Ollama)")
    r.add_argument("--scenarios", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--jsonl", required=True)
    r.add_argument("--ollama-url", default=os.environ.get("SLOTHCTF_OLLAMA_URL", "http://localhost:11434"))
    r.add_argument("--model", default=None)
    r.add_argument("--limit", type=int, default=None)

    a = sub.add_parser("aggregate", help="aggregate a JSONL dataset into MD + CSV")
    a.add_argument("--jsonl", required=True)
    a.add_argument("--md", default=None)
    a.add_argument("--csv", default=None)

    an = sub.add_parser("anonymize", help="write a publication-safe copy of a dataset")
    an.add_argument("--jsonl", required=True)
    an.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        campaign = load_campaign(args.scenarios)
        run_campaign(campaign, args.out_dir, args.jsonl, model=args.model,
                     ollama_url=args.ollama_url, limit=args.limit)
    elif args.cmd == "aggregate":
        recs = _read_records(args.jsonl)
        summary = aggregate(recs)
        print(json.dumps(summary, indent=2))
        if args.md:
            Path(args.md).write_text(render_markdown(summary), encoding="utf-8")
            print("wrote", args.md, file=sys.stderr)
        if args.csv:
            write_csv(recs, args.csv)
            print("wrote", args.csv, file=sys.stderr)
    elif args.cmd == "anonymize":
        recs = _read_records(args.jsonl)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out).open("w", encoding="utf-8") as fh:
            for r in anonymize(recs):
                fh.write(json.dumps(r) + "\n")
        print("wrote", args.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
