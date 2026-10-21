"""A dependency-free web GUI to browse the challenge catalog.

Uses only the standard library (http.server). Run it with:

    python main.py web

It rescans the generated/ tree into the catalog, then serves a single page at
http://127.0.0.1:8000 that lists challenges with live filtering, and a detail
view per challenge: player-facing files to download, and (behind a reveal) the
author material showing how the flag is built — the C source for reverse
challenges, and later the log + extractor for log challenges.
"""
from __future__ import annotations

import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from slothctf import catalog

GENERATED = Path("generated")

# Files that reveal the solution / how the challenge is built. Everything else
# (the binary, the README, evidence the player receives) is player-facing.
_AUTHOR_PREFIXES = ("organizer/", "src/")
_AUTHOR_NAMES = ("solution.md", "challenge.json")
_TEXT_EXT = {".c", ".h", ".py", ".md", ".json", ".txt", ".log", ".csv"}


def _challenge_dir(name: str) -> Path | None:
    # name comes from the catalog; keep it a single path segment (no traversal).
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return None
    d = GENERATED / name
    return d if (d / "challenge.json").exists() else None


def _classify(rel: str) -> str:
    rel_posix = rel.replace("\\", "/")
    if rel_posix in _AUTHOR_NAMES or rel_posix.startswith(_AUTHOR_PREFIXES):
        return "author"
    # the compiled encoder is an organizer tool, not something the player gets
    if rel_posix.startswith("bin/") and rel_posix.rsplit("/", 1)[-1].startswith("encoder"):
        return "author"
    return "player"


def _list_files(challenge_dir: Path) -> list[dict]:
    out = []
    for p in sorted(challenge_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(challenge_dir).as_posix()
        out.append({
            "path": rel,
            "size": p.stat().st_size,
            "group": _classify(rel),
            "text": p.suffix.lower() in _TEXT_EXT,
        })
    return out


def _safe_file(name: str, rel: str) -> Path | None:
    """Resolve rel inside the named challenge dir, rejecting path traversal."""
    base = _challenge_dir(name)
    if base is None:
        return None
    base_res = base.resolve()
    target = (base / rel).resolve()
    if base_res != target and base_res not in target.parents:
        return None
    return target if target.is_file() else None


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SlothCTF Catalog</title>
<style>
  :root { color-scheme: light dark; --bg:#0f1115; --card:#1a1d24; --fg:#e8eaed;
          --muted:#9aa0aa; --accent:#8bd450; --line:#2a2e37; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 system-ui,Segoe UI,sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:20px 24px; border-bottom:1px solid var(--line); display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  h1 { margin:0; font-size:22px; } h1 .sloth { font-size:24px; }
  .count { color:var(--muted); }
  .controls { padding:16px 24px; display:flex; gap:10px; flex-wrap:wrap; align-items:center; border-bottom:1px solid var(--line); }
  select, input { background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:8px 10px; font:inherit; }
  input[type=search] { min-width:220px; flex:1; }
  main { padding:20px 24px; display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; cursor:pointer; }
  .card:hover { border-color:var(--accent); }
  .card h3 { margin:0 0 6px; font-size:16px; }
  .tags { display:flex; gap:6px; flex-wrap:wrap; margin:8px 0; }
  .tag { font-size:12px; padding:2px 8px; border-radius:999px; border:1px solid var(--line); color:var(--muted); }
  .tag.kind { color:var(--accent); border-color:var(--accent); }
  .desc { color:var(--muted); font-size:13px; }
  .meta { margin-top:10px; font-size:12px; color:var(--muted); }
  code { background:#000; padding:1px 5px; border-radius:5px; font-size:12px; }
  .empty { color:var(--muted); padding:40px 24px; }
  /* modal */
  .overlay { position:fixed; inset:0; background:rgba(0,0,0,.6); display:none; align-items:flex-start; justify-content:center; padding:40px 16px; overflow:auto; }
  .overlay.open { display:flex; }
  .modal { background:var(--bg); border:1px solid var(--line); border-radius:14px; max-width:860px; width:100%; padding:22px 24px; }
  .modal h2 { margin:0 0 4px; } .modal .sub { color:var(--muted); margin-bottom:14px; }
  .modal h4 { margin:18px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px; }
  .files a { color:var(--accent); display:inline-block; margin:2px 12px 2px 0; }
  .btn { background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:6px 12px; cursor:pointer; font:inherit; }
  .close { float:right; }
  pre { background:#000; border:1px solid var(--line); border-radius:8px; padding:12px; overflow:auto; max-height:360px; font-size:12.5px; }
  .filebtn { background:none; border:none; color:var(--accent); cursor:pointer; font:inherit; padding:0; margin:2px 12px 2px 0; }
</style>
</head>
<body>
<header>
  <h1><span class="sloth">🦥</span> SlothCTF Catalog</h1>
  <span class="count" id="count"></span>
</header>
<div class="controls">
  <select id="category"><option value="">All categories</option></select>
  <select id="difficulty"><option value="">All difficulties</option></select>
  <select id="kind"><option value="">All kinds</option></select>
  <input type="search" id="search" placeholder="Search name / title…">
</div>
<main id="grid"></main>

<div class="overlay" id="overlay">
  <div class="modal" id="modal"></div>
</div>

<script>
const grid = document.getElementById('grid'), count = document.getElementById('count');
const overlay = document.getElementById('overlay'), modal = document.getElementById('modal');
const sel = {category:document.getElementById('category'), difficulty:document.getElementById('difficulty'),
             kind:document.getElementById('kind'), search:document.getElementById('search')};

function esc(s){ return (s||'').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function fileUrl(name, path, dl){ return '/api/file?name='+encodeURIComponent(name)+'&path='+encodeURIComponent(path)+(dl?'&download=1':''); }

async function loadFacets(){
  const f = await (await fetch('/api/facets')).json();
  for (const key of ['category','difficulty','kind'])
    for (const v of f[key]) { const o=document.createElement('option'); o.value=v; o.textContent=v; sel[key].appendChild(o); }
}
async function refresh(){
  const q = new URLSearchParams();
  for (const k of ['category','difficulty','kind','search']) if (sel[k].value) q.set(k, sel[k].value);
  const rows = await (await fetch('/api/challenges?'+q)).json();
  count.textContent = rows.length + ' challenge' + (rows.length===1?'':'s');
  grid.innerHTML = rows.length ? '' : '<p class="empty">No challenges. Generate some with <code>python main.py generate</code>.</p>';
  for (const r of rows){
    const el = document.createElement('div'); el.className='card'; el.onclick=()=>openDetail(r.name);
    el.innerHTML = `<h3>${esc(r.title)}</h3>
      <div class="tags">
        <span class="tag kind">${esc(r.kind)}</span>
        <span class="tag">${esc(r.category)}</span>
        <span class="tag">${esc(r.difficulty)}</span>
        ${r.built? '<span class="tag">built</span>':''}
      </div>
      <div class="desc"><code>${esc(r.name)}</code></div>
      <div class="meta">flag: <code>${esc(r.flag)}</code></div>`;
    grid.appendChild(el);
  }
}

async function openDetail(name){
  const d = await (await fetch('/api/challenge/'+encodeURIComponent(name))).json();
  const player = d.files.filter(f=>f.group==='player');
  const author = d.files.filter(f=>f.group==='author');
  const dl = f => `<a href="${fileUrl(name,f.path,1)}">${esc(f.path)}</a>`;
  const view = f => f.text
      ? `<button class="filebtn" onclick="viewFile('${esc(name)}','${esc(f.path)}')">${esc(f.path)}</button>`
      : `<a href="${fileUrl(name,f.path,1)}">${esc(f.path)} (download)</a>`;
  modal.innerHTML = `
    <button class="btn close" onclick="closeDetail()">✕ close</button>
    <h2>${esc(d.title)}</h2>
    <div class="sub"><code>${esc(d.name)}</code> · ${esc(d.category)} · ${esc(d.difficulty)} · ${esc(d.kind)}</div>
    <p>${esc(d.description)}</p>
    <div class="meta">flag: <code>${esc(d.flag)}</code>${d.success_marker?` · success marker: <code>${esc(d.success_marker)}</code>`:''}</div>
    <h4>Player files (download)</h4>
    <div class="files">${player.length? player.map(dl).join('') : '<span class="desc">none</span>'}</div>
    <h4>How the flag is built <button class="btn" id="revealBtn" onclick="toggleReveal()">reveal ▸</button></h4>
    <div id="reveal" style="display:none">
      <div class="files">${author.map(view).join('')}</div>
      <pre id="viewer" style="display:none"></pre>
    </div>`;
  overlay.classList.add('open');
}
function toggleReveal(){ const r=document.getElementById('reveal'); const b=document.getElementById('revealBtn');
  const open = r.style.display==='none'; r.style.display=open?'block':'none'; b.textContent=open?'hide ▾':'reveal ▸'; }
async function viewFile(name, path){
  const txt = await (await fetch(fileUrl(name,path,0))).text();
  const v = document.getElementById('viewer'); v.style.display='block'; v.textContent = txt; v.scrollIntoView({behavior:'smooth',block:'nearest'});
}
function closeDetail(){ overlay.classList.remove('open'); }
overlay.addEventListener('click', e => { if (e.target===overlay) closeDetail(); });
document.addEventListener('keydown', e => { if (e.key==='Escape') closeDetail(); });

for (const k in sel) sel[k].addEventListener('input', refresh);
loadFacets().then(refresh);
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path, q = parsed.path, parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/facets":
            self._json(catalog.facets())
        elif path == "/api/challenges":
            self._json(catalog.query(
                category=(q.get("category") or [None])[0],
                difficulty=(q.get("difficulty") or [None])[0],
                kind=(q.get("kind") or [None])[0],
                search=(q.get("search") or [None])[0],
            ))
        elif path.startswith("/api/challenge/"):
            name = path[len("/api/challenge/"):]
            d = _challenge_dir(name)
            if d is None:
                self._send(404, b"not found", "text/plain"); return
            meta = json.loads((d / "challenge.json").read_text(encoding="utf-8"))
            self._json({
                "name": name,
                "title": meta.get("title", name),
                "category": meta.get("category", ""),
                "difficulty": meta.get("difficulty", ""),
                "kind": meta.get("challenge_kind", "decode"),
                "flag": meta.get("flag", ""),
                "success_marker": meta.get("success_marker", ""),
                "description": meta.get("description", ""),
                "files": _list_files(d),
            })
        elif path == "/api/file":
            name = (q.get("name") or [""])[0]
            rel = (q.get("path") or [""])[0]
            target = _safe_file(name, rel)
            if target is None:
                self._send(404, b"not found", "text/plain"); return
            data = target.read_bytes()
            download = (q.get("download") or ["0"])[0] == "1"
            if download:
                ctype = "application/octet-stream"
                extra = {"Content-Disposition": f'attachment; filename="{target.name}"'}
            else:
                ctype = mimetypes.guess_type(target.name)[0] or "text/plain; charset=utf-8"
                if ctype.startswith("text/") or target.suffix.lower() in _TEXT_EXT:
                    ctype = "text/plain; charset=utf-8"
                extra = None
            self._send(200, data, ctype, extra)
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *_args) -> None:  # keep the console quiet
        pass


def launch_web(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    catalog.index_generated()  # refresh the catalog from disk on startup
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    print(f"SlothCTF catalog GUI on {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        server.server_close()
