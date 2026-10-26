"""Batch Campaign Runner for SlothCTF.

Runs a configured campaign of CTF generations (monolithic or agentic, any genre /
difficulty / target) sequentially, with crash-safe resume and automatic stats — a
stable replacement for the throwaway campaign scripts used during development.

Commands (single entry point):
    python -m slothctf.batch run    --config CONFIG.json
    python -m slothctf.batch resume --dir   generated/CAMPAIGN   [--ollama-url URL]
    python -m slothctf.batch report --dir   generated/CAMPAIGN

Design guarantees:
  * Experiment identity is a FINGERPRINT of commit + model + digest + prompt +
    inference + budget + workflow + target + scenario config. The Ollama URL is
    transport-only and NOT in the fingerprint, so the tunnel may change between a
    run and its resume as long as the model digest is unchanged.
  * Append-only dataset (campaign.jsonl), per-line fsync, tolerant of a truncated
    trailing line from an interrupted write. Completed runs are never repeated.
  * Interrupted attempts are preserved and classified separately: `infra` (a caught
    InfraError / transport fault) and `incomplete` (process killed with no result).
    Neither counts in the model success-rate denominator. No automatic retries.
  * Isolated execution only (Docker); never a non-isolated native fallback.
  * Confidential data (flags, transcripts) stays inside each run directory and is
    NEVER written into campaign.jsonl or the reports; nothing is published.

This module changes no generator; it orchestrates the existing ones.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

import slothctf.generator as g
from slothctf.agentic import ProjectState, InfraError
from slothctf.execution import SandboxUnavailable

GENERATED = Path("generated")

# Real infrastructure faults (transport/sandbox/OS). OSError already covers ConnectionError,
# TimeoutError and urllib's URLError. Anything NOT in here is a runner/verifier fault.
_INFRA_EXC = (InfraError, SandboxUnavailable, OSError)


class BatchError(Exception):
    pass


# --------------------------------------------------------------------------- #
# config, fingerprint, digest
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for req in ("model", "ollama_url", "matrix", "repeats", "dest"):
        if req not in cfg:
            raise BatchError(f"config missing required key: {req!r}")
    m = cfg["matrix"]
    for req in ("genres", "difficulties", "targets", "workflows"):
        if not isinstance(m.get(req), list) or not m[req]:
            raise BatchError(f"config.matrix.{req} must be a non-empty list")
    dest = cfg["dest"]
    if not isinstance(dest, str) or "/" in dest or "\\" in dest or dest in (".", ".."):
        raise BatchError("config.dest must be a single safe folder name (lives under generated/)")
    cfg.setdefault("isolation", "isolated")
    cfg.setdefault("inference", {})
    cfg.setdefault("budget", {})
    cfg.setdefault("prompt", g.DEFAULT_PROMPT)
    inf = cfg.get("inference", {})
    for wf in m["workflows"]:
        if wf not in ("monolithic", "agentic"):
            raise BatchError(f"unknown workflow {wf!r}")
        if wf == "agentic" and any(gen != "recover" for gen in m["genres"]):
            raise BatchError("agentic workflow supports the 'recover' genre only")
        if wf == "monolithic":
            # monolithic's transport sets temperature internally and fixes num_ctx; it
            # cannot honour these overrides, so reject them rather than fingerprint a lie.
            unsupported = [k for k in ("temperature", "num_ctx") if k in inf]
            if unsupported:
                raise BatchError(f"inference override(s) {unsupported} are not supported by the monolithic "
                                 f"workflow; remove them or use the agentic workflow")
    return cfg


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _live_digest(ollama_url: str, model: str) -> str | None:
    """Query the Ollama endpoint for the model's digest. Injectable for tests."""
    url = ollama_url.rstrip("/") + "/api/tags"
    with request.urlopen(url, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    for mdl in data.get("models", []):
        if mdl.get("name") == model:
            return mdl.get("digest")
    return None


def fingerprint(config: dict, commit: str, digest: str | None) -> str:
    """Experiment identity. Deliberately EXCLUDES ollama_url (transport only)."""
    exp = {
        "commit": commit, "model": config["model"], "digest": digest,
        "prompt": config.get("prompt", ""), "inference": config.get("inference", {}),
        "budget": config.get("budget", {}), "matrix": config["matrix"],
        "repeats": config["repeats"], "isolation": config.get("isolation", "isolated"),
    }
    return hashlib.sha256(_canonical(exp).encode("utf-8")).hexdigest()


def plan(config: dict) -> list[dict]:
    """Deterministic, ordered list of planned runs with stable ids (for resume)."""
    m, out = config["matrix"], []
    for wf in m["workflows"]:
        for genre in m["genres"]:
            for diff in m["difficulties"]:
                for tgt in m["targets"]:
                    for k in range(1, int(config["repeats"]) + 1):
                        out.append({"id": f"{genre}_{diff}_{tgt}_{wf}_{k:02d}",
                                    "genre": genre, "difficulty": diff, "target": tgt, "workflow": wf})
    return out


# --------------------------------------------------------------------------- #
# append-only dataset (crash-safe)
# --------------------------------------------------------------------------- #
def _append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush(); os.fsync(fh.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    """STRICT read of an already-clean dataset: every non-empty line must be valid JSON.
    A malformed line raises (no valid event is ever silently dropped). Truncated trailing
    lines are handled only by `_repair_jsonl`, which runs before a run/resume/report."""
    out: list[dict] = []
    if not path.exists():
        return out
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = raw.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise BatchError(f"corrupt record on line {i+1} of {path.name}: {exc} "
                             f"(run a repair / inspect the diagnostic copy)")
    return out


def _repair_jsonl(path: Path) -> None:
    """Make the dataset safe before use. Only a TRUNCATED FINAL line (an interrupted
    append that left bytes without a terminating newline) is recoverable: the full
    original bytes are first preserved to a diagnostic copy, then the file is rewritten
    with only the complete, newline-terminated, valid records — so no later append
    concatenates onto damaged bytes. A malformed COMPLETE/intermediate record is NOT
    silently dropped: the original is preserved and BatchError is raised."""
    if not path.exists():
        return
    data = path.read_text(encoding="utf-8")
    if data == "":
        return
    ends_nl = data.endswith("\n")
    parts = data.split("\n")
    if ends_nl:
        parts = parts[:-1]            # trailing "" after the final newline
        complete, trailing = parts, None
    else:
        complete, trailing = parts[:-1], parts[-1]   # last part is a truncated line

    bad = next((i for i, ln in enumerate(complete) if ln.strip() and not _is_json(ln)), None)
    if bad is not None:
        diag = _preserve_corrupt(path, data)
        raise BatchError(f"corrupt complete record on line {bad+1} of {path.name}; original preserved "
                         f"at {diag.name}. Refusing to continue (intermediate corruption is not auto-dropped).")
    if trailing is not None and trailing.strip():
        # recoverable: a single truncated final line from an interrupted write
        _preserve_corrupt(path, data)
        rebuilt = "".join(ln + "\n" for ln in complete if ln.strip())
        _write_atomic(path, rebuilt)
    elif trailing is not None:
        # final newline missing but the trailing fragment is empty/whitespace -> normalise
        rebuilt = "".join(ln + "\n" for ln in complete if ln.strip())
        if rebuilt != data:
            _write_atomic(path, rebuilt)


def _is_json(line: str) -> bool:
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError:
        return False


def _preserve_corrupt(path: Path, data: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    diag = path.with_name(f"{path.name}.corrupt-{stamp}")
    diag.write_bytes(data.encode("utf-8"))   # integral copy of the damaged bytes
    return diag


def _read_transcript(path: Path) -> list[dict]:
    """Tolerant read for a per-run transcript (NOT the authoritative campaign dataset):
    a run killed mid-write may leave a partial line — skip unparseable lines here."""
    out: list[dict] = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    with tmp.open("r+", encoding="utf-8") as fh:
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# mechanic heuristic (kept separate from verified results)
# --------------------------------------------------------------------------- #
_FAMILIES = {
    "xor": r"\bxor\b|\^", "rotate": r"rotat|\brol\b|\bror\b",
    "add/caesar": r"caesar|addition|\badd\b|subtract|increment|shift by",
    "state-machine": r"state machine|state-dependent|state-based|per-state",
    "substitution": r"substitution|s-box|sbox|lookup table",
    "permutation": r"permutation|shuffle|rearrange|reorder",
    "multiply/mod": r"multipl|modular|modulo| mod ", "keystream/prng": r"keystream|lfsr|prng|pseudo-random|feedback",
}


def mechanic_families(design: str | None) -> list[str]:
    d = (design or "").lower()
    return sorted({f for f, pat in _FAMILIES.items() if re.search(pat, d)})


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def _resolve_isolated_backend(target: str, isolation: str):
    if isolation != "isolated":
        raise BatchError(f"isolation={isolation!r} not allowed; only 'isolated' (no native fallback)")
    be = g.resolve_backend(g.get_target(target), "isolated")
    if not str(getattr(be, "name", "")).startswith("docker"):
        raise BatchError(f"resolved non-isolated backend {getattr(be,'name','?')!r}; refusing (no native fallback)")
    return be


def _default_caller_factory(workflow: str, url: str, inference: dict):
    """Build the transport actually used, applying the configured inference params so the
    real payload matches the fingerprint. Agentic: a parameterised Ollama transport
    (temperature + num_ctx). Monolithic: None — the generator uses its own transport at
    the configured endpoint (inference overrides are rejected at config time)."""
    from slothctf.agentic import ollama_transport
    if workflow == "agentic":
        return ollama_transport(url, temperature=inference.get("temperature", 0.7),
                                num_ctx=inference.get("num_ctx", 4096))
    return None


def _generate(spec: dict, name: str, config: dict, caller, url: str):
    """Run ONE generation through the frozen generators. Injectable for tests. `url` is the
    CURRENT transport endpoint (may have been updated on resume)."""
    backend = _resolve_isolated_backend(spec["target"], config.get("isolation", "isolated"))
    b = config.get("budget", {})
    if spec["workflow"] == "agentic":
        from slothctf.agentic import generate_recover_agentic, Budget
        budget = Budget(max_llm_calls=b.get("max_llm_calls", 24),
                        max_repairs_per_stage=b.get("max_repairs_per_stage", 4),
                        max_wall_seconds=b.get("max_wall_seconds", 1800))
        return generate_recover_agentic(name, prompt=config.get("prompt", ""), difficulty=spec["difficulty"],
                                        target=spec["target"], backend=backend, model=config["model"],
                                        ollama_url=url, caller=caller, budget=budget, resume=False)
    # monolithic: its own transport, but at the CONFIGURED endpoint (was defaulting before)
    return g.generate_challenge(name, prompt=config.get("prompt", g.DEFAULT_PROMPT), difficulty=spec["difficulty"],
                                model=config["model"], ollama_url=url, genre=spec["genre"], target=spec["target"],
                                backend=backend, caller=caller, max_attempts=b.get("max_attempts", 8),
                                return_report=True)


def _run_dir(config: dict, pid: str, attempt: int) -> Path:
    return GENERATED / config["dest"] / pid / f"attempt_{attempt:02d}"


def _partial_metrics(run_dir: Path) -> dict:
    """Best-effort measurable cost for an interrupted/incomplete attempt. Unknown
    values stay null (never faked to zero)."""
    out = {"llm_calls": None, "tokens": None, "wall_s_recorded": None}
    sp = run_dir / "agentic_state.json"
    if not sp.exists():
        return out
    try:
        st = ProjectState.load(sp)
        tr = _read_transcript(run_dir / "transcript.jsonl")
        known = [e.get("tokens") for e in tr if e.get("type") == "llm" and isinstance(e.get("tokens"), int)]
        out["llm_calls"] = st.budget_calls
        out["tokens"] = sum(known) if known else None
    except Exception:
        pass
    return out


def _metrics(rep, config: dict, spec: dict, name: str, wall_s: float) -> dict:
    tele = getattr(rep, "telemetry", {}) or {}
    cat = (tele.get("error") or {}).get("category")
    success = bool(getattr(rep, "success", False))
    if cat == "infra":
        oc = "infra"
    elif cat == "verifier-error":
        oc = "runner-error"            # harness/verifier fault — NOT infra, NOT a model failure
    elif success:
        oc = "valid"
    else:
        oc = "model-failure"           # includes 'model-invalid' (the model's own bad output)
    final_stage = tele.get("final_stage") or (rep.attempts[-1].stage if getattr(rep, "attempts", None) else None)
    m = {"outcome_class": oc, "success": success, "final_stage": final_stage,
         "attempts_used": len(getattr(rep, "attempts", []) or []), "wall_s": round(wall_s, 1),
         "llm_calls": None, "tokens": None, "tokens_complete": None, "repairs": None,
         "effective_repairs": None, "no_progress": None, "failure_stage": None,
         "mechanic_families": None, "mechanic_design": None}
    d = GENERATED / name
    sp = d / "agentic_state.json"
    if sp.exists():
        try:
            st = ProjectState.load(sp)
            tr = _read_transcript(d / "transcript.jsonl")
            per = [e.get("tokens") for e in tr if e.get("type") == "llm"]
            known = [t for t in per if isinstance(t, int)]
            m["llm_calls"] = st.budget_calls
            m["tokens"] = sum(known) if known else None            # no fake zero
            m["tokens_complete"] = (len(known) == len(per)) if per else None
            m["repairs"] = sum(getattr(st, c).repairs for c in ("encoder", "challenge", "solver"))
            m["effective_repairs"] = sum(1 for e in tr if e.get("type") == "revision" and e.get("applied"))
            m["no_progress"] = sum(1 for e in tr if e.get("type") == "repair-no-progress")
            m["mechanic_design"] = st.design
            m["mechanic_families"] = mechanic_families(st.design)
            status = {c: getattr(st, c).status for c in ("encoder", "challenge", "solver")}
            if oc == "valid":
                m["failure_stage"] = "none"
            elif status["encoder"] != "verified":
                m["failure_stage"] = "encoder(B)"
            elif status["challenge"] != "verified":
                m["failure_stage"] = "challenge(C)"
            elif status["solver"] != "verified":
                m["failure_stage"] = "solver(D)"
            else:
                m["failure_stage"] = "final(E)"
        except Exception:
            pass
    else:
        m["failure_stage"] = "none" if oc == "valid" else (final_stage or "unknown")
    return m


def _copy_player_artifacts(name: str, dest: Path) -> None:
    src = GENERATED / name
    dest.mkdir(parents=True, exist_ok=True)
    for rel in ("README.md", "bin/challenge", "bin/challenge.exe"):
        p = src / rel
        if p.exists():
            (dest / Path(rel).name).write_bytes(p.read_bytes())


def _execute(camp_dir: Path, config: dict, manifest: dict, digest: str | None, caller_factory=None):
    """Shared run/resume loop. Skips model-terminal runs, re-attempts infra/incomplete
    ones (preserving prior attempts), stops on an infra fault without auto-retry."""
    caller_factory = caller_factory or _default_caller_factory
    camp = camp_dir / "campaign.jsonl"

    def reload():
        recs = _read_jsonl(camp)
        results, attempts_seen = {}, {}
        for r in recs:
            ev = r.get("event")
            if ev in ("attempt_start", "attempt_result", "attempt_incomplete"):
                attempts_seen.setdefault(r["id"], set()).add(r["attempt"])
            if ev == "attempt_result":
                results.setdefault(r["id"], []).append(r)
        return recs, results, attempts_seen

    recs, results, attempts_seen = reload()
    # classify orphan attempts (start without a terminal result => process was killed)
    for r in recs:
        if r.get("event") != "attempt_start":
            continue
        pid, a = r["id"], r["attempt"]
        has_result = any(x["attempt"] == a for x in results.get(pid, []))
        already = any(e.get("event") == "attempt_incomplete" and e.get("id") == pid and e.get("attempt") == a for e in recs)
        if not has_result and not already:
            _append(camp, {"event": "attempt_incomplete", "id": pid, "attempt": a, "ts": time.time(),
                           "note": "process ended without a final result (preserved; not a model failure)",
                           **_partial_metrics(_run_dir(config, pid, a))})
    recs, results, attempts_seen = reload()

    def is_done(pid):   # a model-terminal result exists (valid or model-failure)
        return any(x["outcome_class"] in ("valid", "model-failure") for x in results.get(pid, []))

    for spec in manifest["plan"]:
        pid = spec["id"]
        if is_done(pid):
            continue
        used = attempts_seen.get(pid, set())
        k = (max(used) + 1) if used else 1
        name = f"{config['dest']}/{pid}/attempt_{k:02d}"
        _append(camp, {"event": "attempt_start", "id": pid, "attempt": k, "name": name,
                       "ts": time.time(), "commit": manifest["commit"], "digest": digest, **spec})
        url = manifest["url_current"]
        caller = caller_factory(spec["workflow"], url, config.get("inference", {}))
        t0 = time.time()

        def _record(outcome, exc):   # preserve the attempt's artifacts + measurable cost
            _append(camp, {"event": "attempt_result", "id": pid, "attempt": k, "outcome_class": outcome,
                           "error": (f"{type(exc).__name__}: {exc}"[:300] if exc else None),
                           "wall_s": round(time.time() - t0, 1), **spec,
                           **_partial_metrics(_run_dir(config, pid, k))})

        try:
            rep = _generate(spec, name, config, caller, url)
        except (KeyboardInterrupt, SystemExit) as exc:   # explicit interruption: preserve + propagate
            _record("interrupted", exc)
            _append(camp, {"event": "STOP", "reason": "user interrupt", "id": pid})
            raise
        except _INFRA_EXC as exc:                          # real infrastructure fault
            _record("infra", exc)
            _append(camp, {"event": "STOP", "reason": "infra exception", "id": pid})
            return "stopped-infra"
        except Exception as exc:                            # unexpected runner/verifier fault
            _record("runner-error", exc)
            _append(camp, {"event": "STOP", "reason": "runner error", "id": pid})
            return "stopped-error"
        m = _metrics(rep, config, spec, name, time.time() - t0)
        _append(camp, {"event": "attempt_result", "id": pid, "attempt": k, "commit": manifest["commit"],
                       "digest": digest, **spec, **m})
        attempts_seen.setdefault(pid, set()).add(k)
        results.setdefault(pid, []).append({"attempt": k, "outcome_class": m["outcome_class"]})
        print(f"{pid} a{k}: {m['outcome_class']} final={m['final_stage']} "
              f"calls={m['llm_calls']} tok={m['tokens']} {m['wall_s']}s", flush=True)
        if m["outcome_class"] == "valid":
            _copy_player_artifacts(name, camp_dir / "_valid_playtest" / pid)
        if m["outcome_class"] in ("infra", "runner-error"):   # generator caught + reported it
            _append(camp, {"event": "STOP", "reason": f"{m['outcome_class']} category", "id": pid})
            return "stopped-infra" if m["outcome_class"] == "infra" else "stopped-error"
    _append(camp, {"event": "campaign_done"})
    return "done"


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_run(config_path: str, caller_factory=None, digest_fn=None):
    """Returns (campaign_dir, status). status in {done, stopped-infra, stopped-error}."""
    digest_fn = digest_fn or _live_digest
    cfg = load_config(config_path)
    camp_dir = GENERATED / cfg["dest"]
    if (camp_dir / "manifest.json").exists():
        raise BatchError(f"{camp_dir}/manifest.json exists — use `resume` instead of overwriting")
    camp_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    try:
        digest = digest_fn(cfg["ollama_url"], cfg["model"])
    except OSError as exc:
        raise BatchError(f"cannot reach Ollama at {cfg['ollama_url']} to verify the model digest: {exc}")
    if digest is None:
        raise BatchError(f"model {cfg['model']!r} not found at the endpoint (cannot fingerprint)")
    fp = fingerprint(cfg, commit, digest)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "fingerprint": fp,
                "commit": commit, "model": cfg["model"], "digest": digest,
                "url_current": cfg["ollama_url"], "config": cfg, "plan": plan(cfg)}
    _write_atomic(camp_dir / "manifest.json", json.dumps(manifest, indent=2))
    _append(camp_dir / "campaign.jsonl", {"event": "campaign_start", "fingerprint": fp, "commit": commit,
                                          "model": cfg["model"], "digest": digest, "runs_planned": len(manifest["plan"])})
    status = _execute(camp_dir, cfg, manifest, digest, caller_factory)
    write_report(camp_dir)
    print(f"[batch] run {status}; report in {camp_dir}", flush=True)
    return camp_dir, status


def cmd_resume(camp_dir: str | Path, ollama_url: str | None = None, caller_factory=None, digest_fn=None):
    """Returns (campaign_dir, status). status in {done, stopped-infra, stopped-error}."""
    digest_fn = digest_fn or _live_digest
    camp_dir = Path(camp_dir)
    mpath = camp_dir / "manifest.json"
    if not mpath.exists():
        raise BatchError(f"no manifest.json in {camp_dir}")
    # make the append-only dataset safe BEFORE any further append (truncated-tail recovery
    # or refusal on intermediate corruption), so resume never concatenates onto damaged bytes
    _repair_jsonl(camp_dir / "campaign.jsonl")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    cfg = manifest["config"]
    # the URL is transport-only: allow updating it, re-verify the digest, never let it break resume
    if ollama_url and ollama_url != manifest.get("url_current"):
        _append(camp_dir / "campaign.jsonl", {"event": "endpoint_change", "ts": time.time(),
                                              "from": manifest.get("url_current"), "to": ollama_url})
        manifest["url_current"] = ollama_url
        _write_atomic(mpath, json.dumps(manifest, indent=2))
    url = manifest["url_current"]
    commit = git_commit()
    try:
        digest = digest_fn(url, cfg["model"])
    except OSError as exc:
        raise BatchError(f"cannot reach Ollama at {url} to verify the model digest: {exc}")
    fp_now = fingerprint(cfg, commit, digest)
    if fp_now != manifest["fingerprint"]:
        why = []
        if commit != manifest["commit"]:
            why.append(f"commit {manifest['commit'][:7]}->{commit[:7]}")
        if digest != manifest["digest"]:
            why.append("model digest changed")
        why = why or ["config/model mismatch"]
        raise BatchError("refusing to resume: experiment fingerprint changed (" + ", ".join(why) +
                         "). URL changes are allowed; code/model/config/digest changes are not.")
    status = _execute(camp_dir, cfg, manifest, digest, caller_factory)
    write_report(camp_dir)
    print(f"[batch] resume {status}; report in {camp_dir}", flush=True)
    return camp_dir, status


# --------------------------------------------------------------------------- #
# reporting (markdown + csv) — infra/incomplete kept out of the success denominator
# --------------------------------------------------------------------------- #
def _stats(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {"n": len(xs), "median": round(statistics.median(xs), 1), "mean": round(statistics.mean(xs), 1),
            "min": min(xs), "max": max(xs)}


def collect(camp_dir: Path) -> dict:
    _repair_jsonl(camp_dir / "campaign.jsonl")   # recover a truncated tail before reading
    recs = _read_jsonl(camp_dir / "campaign.jsonl")
    results = [r for r in recs if r.get("event") == "attempt_result"]
    completed = [r for r in results if r["outcome_class"] in ("valid", "model-failure")]
    infra = [r for r in results if r["outcome_class"] == "infra"]
    runner_err = [r for r in results if r["outcome_class"] == "runner-error"]
    interrupted = [r for r in results if r["outcome_class"] == "interrupted"]
    incomplete = [r for r in recs if r.get("event") == "attempt_incomplete"]
    endpoint_changes = [r for r in recs if r.get("event") == "endpoint_change"]
    return {"recs": recs, "completed": completed, "infra": infra, "runner_error": runner_err,
            "interrupted": interrupted, "incomplete": incomplete, "endpoint_changes": endpoint_changes}


def write_report(camp_dir: Path) -> tuple[Path, Path]:
    if not (camp_dir / "campaign.jsonl").exists() and not (camp_dir / "manifest.json").exists():
        raise BatchError(f"no campaign found at {camp_dir}")
    c = collect(camp_dir)
    completed, infra, incomplete = c["completed"], c["infra"], c["incomplete"]

    def group_key(r):
        return (r.get("genre"), r.get("difficulty"), r.get("target"), r.get("workflow"))
    groups: dict = {}
    for r in completed:
        groups.setdefault(group_key(r), []).append(r)

    lines = ["# Campaign report", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
             "> Success rate is over COMPLETED model generations only. Infra-interrupted and "
             "incomplete (killed) attempts are reported separately and excluded from the denominator. "
             "Unavailable token counts are shown blank, never zero. Mechanic families are a heuristic, "
             "separate from verified results.", "",
             f"- Completed model generations: **{len(completed)}**",
             f"- Model successes (fully valid): **{sum(1 for r in completed if r['success'])}**",
             f"- Model failures: **{sum(1 for r in completed if not r['success'])}**",
             f"- Infra-interrupted attempts (excluded): **{len(infra)}**",
             f"- Runner/verifier-error attempts (excluded): **{len(c['runner_error'])}**",
             f"- User-interrupted attempts (excluded): **{len(c['interrupted'])}**",
             f"- Incomplete/killed attempts (excluded): **{len(incomplete)}**",
             f"- Endpoint changes during the campaign: **{len(c['endpoint_changes'])}**", ""]

    lines.append("## Reliability by genre / difficulty / target / workflow\n")
    lines.append("| genre | diff | target | workflow | completed | valid | success rate | failure stages |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        nv = sum(1 for r in rs if r["success"])
        rate = f"{nv}/{len(rs)} ({round(100*nv/len(rs))}%)" if rs else "0/0"
        fstages = {}
        for r in rs:
            if not r["success"]:
                fstages[r.get("failure_stage") or "unknown"] = fstages.get(r.get("failure_stage") or "unknown", 0) + 1
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(rs)} | {nv} | {rate} | "
                     f"{', '.join(f'{k}:{v}' for k,v in sorted(fstages.items())) or '-'} |")

    def col(rs, k):
        return [r.get(k) for r in rs]
    lines += ["", "## Efficiency (completed generations)", ""]
    lines.append(f"- LLM calls: {_stats(col(completed,'llm_calls'))}")
    lines.append(f"- tokens (measured only): {_stats(col(completed,'tokens'))}")
    lines.append(f"- duration s: {_stats(col(completed,'wall_s'))}")
    lines.append(f"- effective repairs: {_stats(col(completed,'effective_repairs'))}")
    lines.append(f"- no-progress events: total {sum(x for x in col(completed,'no_progress') if isinstance(x,int))}")
    valids = [r for r in completed if r["success"]]
    if valids:
        lines.append(f"- cost per valid challenge (median over {len(valids)}): "
                     f"calls {_stats(col(valids,'llm_calls'))['median']}, "
                     f"tokens {_stats(col(valids,'tokens'))['median']}, "
                     f"{_stats(col(valids,'wall_s'))['median']}s")
    else:
        lines.append("- cost per valid challenge: n/a (no valid challenge yet)")

    lines += ["", "## Infrastructure (separate — not counted as model results)", ""]
    lines.append(f"- infra-interrupted attempts: {len(infra)}; "
                 f"measured calls {_stats(col(infra,'llm_calls'))}, tokens {_stats(col(infra,'tokens'))}")
    lines.append(f"- incomplete/killed attempts: {len(incomplete)}; "
                 f"measured calls {_stats(col(incomplete,'llm_calls'))}, tokens {_stats(col(incomplete,'tokens'))}")

    lines += ["", "## Mechanics (heuristic — NOT a verified property)", ""]
    fam = {}
    for r in completed:
        for f in (r.get("mechanic_families") or []):
            fam[f] = fam.get(f, 0) + 1
    composed = sum(1 for r in completed if len(r.get("mechanic_families") or []) >= 2)
    lines.append(f"- family frequency (over completed): {fam or '-'}")
    lines.append(f"- composed designs (>=2 families): {composed}/{len(completed)}")

    md = camp_dir / "report.md"
    _write_atomic(md, "\n".join(lines) + "\n")

    # CSV: one row per attempt_result + incomplete (confidential fields excluded)
    csv_path = camp_dir / "report.csv"
    cols = ["id", "attempt", "genre", "difficulty", "target", "workflow", "outcome_class", "success",
            "final_stage", "failure_stage", "attempts_used", "llm_calls", "tokens", "tokens_complete",
            "repairs", "effective_repairs", "no_progress", "wall_s", "mechanic_families"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in completed + infra + c["runner_error"] + c["interrupted"]:
            row = dict(r)
            row["mechanic_families"] = "|".join(r.get("mechanic_families") or [])
            w.writerow(row)
        for r in incomplete:
            w.writerow({"id": r.get("id"), "attempt": r.get("attempt"), "outcome_class": "incomplete",
                        "llm_calls": r.get("llm_calls"), "tokens": r.get("tokens")})
    return md, csv_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m slothctf.batch", description="SlothCTF batch campaign runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="start a new campaign from a config file")
    r.add_argument("--config", required=True)
    rs = sub.add_parser("resume", help="resume an interrupted campaign (no duplication)")
    rs.add_argument("--dir", required=True)
    rs.add_argument("--ollama-url", default=None, help="update the transport endpoint (digest re-verified)")
    rp = sub.add_parser("report", help="(re)generate report.md and report.csv")
    rp.add_argument("--dir", required=True)
    return p


# exit codes (documented, consistent across run/resume):
#   0  campaign completed (even with model-attributable failures)
#   2  usage / config / fingerprint error (BatchError)
#   3  stopped by a real infrastructure fault
#   4  stopped by an unexpected runner/verifier error
#   130 interrupted by the user (Ctrl-C)
_EXIT = {"done": 0, "stopped-infra": 3, "stopped-error": 4}


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "run":
            _, status = cmd_run(args.config)
            return _EXIT.get(status, 1)
        if args.cmd == "resume":
            _, status = cmd_resume(args.dir, ollama_url=args.ollama_url)
            return _EXIT.get(status, 1)
        if args.cmd == "report":
            write_report(Path(args.dir))
            print(f"[batch] report written in {args.dir}")
            return 0
    except BatchError as exc:
        print(f"[batch] error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[batch] interrupted (data preserved; resume to continue)", file=sys.stderr)
        return 130
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
