"""Offline metrics over the generated challenges — a measurable snapshot of the
corpus (counts by genre/difficulty/kind/model, binary coverage, log richness,
which optional gates were used). No model or GPU needed; it reads challenge.json.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _binary_present(d: Path) -> bool:
    b = d / "bin" / "challenge"
    return b.exists() or b.with_suffix(".exe").exists()


def _genre(meta: dict) -> str:
    g = (meta.get("generation") or {}).get("genre")
    if g:
        return g
    return "log" if meta.get("challenge_kind") == "log" else "reverse"


def collect(base_dir: str | Path = "generated") -> dict[str, Any]:
    base = Path(base_dir)
    metas: list[tuple[str, dict, bool, float]] = []
    if base.exists():
        for d in sorted(base.iterdir()):
            j = d / "challenge.json"
            if not d.is_dir() or not j.exists():
                continue
            try:
                meta = json.loads(j.read_text(encoding="utf-8"))
            except Exception:
                continue
            metas.append((d.name, meta, _binary_present(d), j.stat().st_mtime))

    total = len(metas)
    by_genre, by_diff, by_kind, by_model = Counter(), Counter(), Counter(), Counter()
    by_cat = Counter()
    built = 0
    task_counts: list[int] = []
    task_types = Counter()
    gates = Counter()
    created = [m[3] for m in metas]

    for _name, meta, has_bin, _mt in metas:
        by_genre[_genre(meta)] += 1
        by_diff[meta.get("difficulty", "?")] += 1
        by_kind[meta.get("challenge_kind", "?")] += 1
        by_cat[meta.get("category", "?")] += 1
        built += 1 if has_bin else 0
        gen = meta.get("generation") or {}
        if gen.get("model"):
            by_model[gen["model"]] += 1
        if gen.get("roundtrip"):
            gates["roundtrip"] += 1
        if gen.get("adversary"):
            gates["adversary"] += 1
        if meta.get("challenge_kind") == "log":
            tasks = meta.get("tasks") or []
            task_counts.append(len(tasks))
            for t in tasks:
                task_types[str(t.get("type", "?"))] += 1

    log_stats = {}
    if task_counts:
        log_stats = {
            "challenges": len(task_counts),
            "total_tasks": sum(task_counts),
            "avg_tasks": round(sum(task_counts) / len(task_counts), 2),
            "min_tasks": min(task_counts),
            "max_tasks": max(task_counts),
            "by_type": dict(task_types),
        }

    return {
        "total": total,
        "built": built,
        "built_pct": round(100 * built / total, 1) if total else 0.0,
        "by_genre": dict(by_genre),
        "by_difficulty": dict(by_diff),
        "by_kind": dict(by_kind),
        "by_model": dict(by_model),
        "by_category": dict(by_cat),
        "gates": dict(gates),
        "log": log_stats,
        "created_oldest": min(created) if created else None,
        "created_newest": max(created) if created else None,
    }


def _bars(counter: dict, total: int, width: int = 24) -> list[str]:
    out = []
    for k, v in sorted(counter.items(), key=lambda kv: -kv[1]):
        n = int(round(width * v / total)) if total else 0
        out.append(f"  {str(k)[:26]:26} {v:4}  {'#' * n}")
    return out


def render(s: dict[str, Any]) -> str:
    import datetime
    L = [f"SlothCTF - {s['total']} challenge(s), {s['built']} with a compiled binary ({s['built_pct']}%)", ""]
    L.append("By genre:");       L += _bars(s["by_genre"], s["total"])
    L.append("By difficulty:");  L += _bars(s["by_difficulty"], s["total"])
    L.append("By kind:");        L += _bars(s["by_kind"], s["total"])
    if s["by_model"]:
        L.append("By model:");   L += _bars(s["by_model"], s["total"])
    L.append("Top categories:"); L += _bars(dict(sorted(s["by_category"].items(), key=lambda kv: -kv[1])[:8]), s["total"])
    if s["log"]:
        lg = s["log"]
        L.append("")
        L.append(f"Log challenges: {lg['challenges']} log(s), {lg['total_tasks']} questions total, "
                 f"avg {lg['avg_tasks']}/challenge (min {lg['min_tasks']}, max {lg['max_tasks']})")
        L.append(f"  question types: " + ", ".join(f"{k}={v}" for k, v in lg["by_type"].items()))
    if s["gates"]:
        L.append("")
        L.append("Optional gates used at generation: " +
                 ", ".join(f"{k}={v}" for k, v in s["gates"].items()))
    if s["created_oldest"]:
        fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
        L.append("")
        L.append(f"Created between {fmt(s['created_oldest'])} and {fmt(s['created_newest'])}")
    L.append("")
    L.append("Note: this counts only challenges that made it to disk (survivors). It does not "
             "include failed generation runs — attempts, failure stages and pass-rate need a "
             "tracked generation campaign.")
    return "\n".join(L)
