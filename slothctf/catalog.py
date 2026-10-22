"""A tiny SQLite catalog of generated challenges.

The catalog is the shared data model: the CLI `list` command and the web GUI
both read it. It is built by scanning the `generated/` tree for `challenge.json`
files, and `generate` records each new challenge here on success.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DEFAULT_DB = Path("generated") / "catalog.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS challenges (
    name           TEXT PRIMARY KEY,
    title          TEXT,
    category       TEXT,
    difficulty     TEXT,
    kind           TEXT,
    flag           TEXT,
    success_marker TEXT,
    path           TEXT,
    built          INTEGER,            -- 1 if a compiled binary is present
    valid          INTEGER,            -- 1/0 if known-good/bad, NULL if unknown
    created_at     REAL,
    indexed_at     REAL
);
"""


def connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def _binary_present(challenge_dir: Path) -> bool:
    b = challenge_dir / "bin" / "challenge"
    return b.exists() or b.with_suffix(".exe").exists()


def _row_from_dir(challenge_dir: Path) -> dict[str, Any] | None:
    meta_path = challenge_dir / "challenge.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        "name": challenge_dir.name,
        "title": meta.get("title", challenge_dir.name),
        "category": meta.get("category", ""),
        "difficulty": meta.get("difficulty", ""),
        "kind": meta.get("challenge_kind", "decode"),
        "flag": meta.get("flag", ""),
        "success_marker": meta.get("success_marker", ""),
        "path": str(challenge_dir),
        "built": 1 if _binary_present(challenge_dir) else 0,
        "created_at": meta_path.stat().st_mtime,
    }


def _upsert(conn: sqlite3.Connection, row: dict[str, Any], valid: int | None) -> None:
    conn.execute(
        """
        INSERT INTO challenges
            (name, title, category, difficulty, kind, flag, success_marker,
             path, built, valid, created_at, indexed_at)
        VALUES
            (:name, :title, :category, :difficulty, :kind, :flag, :success_marker,
             :path, :built, :valid, :created_at, :indexed_at)
        ON CONFLICT(name) DO UPDATE SET
            title=excluded.title, category=excluded.category,
            difficulty=excluded.difficulty, kind=excluded.kind, flag=excluded.flag,
            success_marker=excluded.success_marker, path=excluded.path,
            built=excluded.built, created_at=excluded.created_at,
            indexed_at=excluded.indexed_at,
            -- keep a known valid/invalid verdict unless a new one is given
            valid=COALESCE(excluded.valid, challenges.valid)
        """,
        {**row, "valid": valid, "indexed_at": time.time()},
    )


def index_generated(base_dir: str | Path = "generated", db_path: str | Path = DEFAULT_DB) -> int:
    """Scan the generated tree and upsert every challenge. Returns the count."""
    base = Path(base_dir)
    conn = connect(db_path)
    count = 0
    seen: list[str] = []
    try:
        if base.exists():
            for child in sorted(base.iterdir()):
                if not child.is_dir():
                    continue
                row = _row_from_dir(child)
                if row is None:
                    continue
                _upsert(conn, row, valid=None)
                seen.append(row["name"])
                count += 1
        # Prune rows whose challenge folder no longer exists on disk.
        existing = [r["name"] for r in conn.execute("SELECT name FROM challenges").fetchall()]
        stale = [n for n in existing if n not in seen]
        if stale:
            conn.executemany("DELETE FROM challenges WHERE name = ?", [(n,) for n in stale])
        conn.commit()
    finally:
        conn.close()
    return count


def record(challenge_dir: str | Path, valid: bool | None = None, db_path: str | Path = DEFAULT_DB) -> None:
    """Record (or refresh) a single challenge, e.g. right after generating it."""
    challenge_dir = Path(challenge_dir)
    row = _row_from_dir(challenge_dir)
    if row is None:
        return
    conn = connect(db_path)
    try:
        _upsert(conn, row, valid=None if valid is None else int(valid))
        conn.commit()
    finally:
        conn.close()


def query(
    category: str | None = None,
    difficulty: str | None = None,
    kind: str | None = None,
    valid: bool | None = None,
    search: str | None = None,
    db_path: str | Path = DEFAULT_DB,
) -> list[dict[str, Any]]:
    conn = connect(db_path)
    try:
        sql = "SELECT * FROM challenges WHERE 1=1"
        params: list[Any] = []
        if category:
            sql += " AND lower(category) = lower(?)"; params.append(category)
        if difficulty:
            sql += " AND lower(difficulty) = lower(?)"; params.append(difficulty)
        if kind:
            sql += " AND kind = ?"; params.append(kind)
        if valid is not None:
            sql += " AND valid = ?"; params.append(int(valid))
        if search:
            like = f"%{search.lower()}%"
            sql += " AND (lower(name) LIKE ? OR lower(title) LIKE ? OR lower(category) LIKE ?)"
            params += [like, like, like]
        sql += " ORDER BY created_at DESC"
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def facets(db_path: str | Path = DEFAULT_DB) -> dict[str, list[str]]:
    """Distinct categories / difficulties / kinds, for building filter menus."""
    conn = connect(db_path)
    try:
        def col(name: str) -> list[str]:
            rows = conn.execute(
                f"SELECT DISTINCT {name} AS v FROM challenges WHERE {name} <> '' ORDER BY v"
            ).fetchall()
            return [r["v"] for r in rows]
        return {"category": col("category"), "difficulty": col("difficulty"), "kind": col("kind")}
    finally:
        conn.close()
