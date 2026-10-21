"""A dependency-free web GUI to browse the challenge catalog.

Uses only the standard library (http.server). Run it with:

    python main.py web

It rescans the generated/ tree into the catalog, then serves a single page at
http://127.0.0.1:8000 that lists challenges with live filtering by category,
difficulty and kind, plus free-text search.
"""
from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from slothctf import catalog

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
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
  .card h3 { margin:0 0 6px; font-size:16px; }
  .tags { display:flex; gap:6px; flex-wrap:wrap; margin:8px 0; }
  .tag { font-size:12px; padding:2px 8px; border-radius:999px; border:1px solid var(--line); color:var(--muted); }
  .tag.kind { color:var(--accent); border-color:var(--accent); }
  .desc { color:var(--muted); font-size:13px; }
  .meta { margin-top:10px; font-size:12px; color:var(--muted); display:flex; justify-content:space-between; gap:8px; }
  code { background:#000; padding:1px 5px; border-radius:5px; font-size:12px; }
  .empty { color:var(--muted); padding:40px 24px; }
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
<script>
const grid = document.getElementById('grid'), count = document.getElementById('count');
const sel = {category:document.getElementById('category'), difficulty:document.getElementById('difficulty'),
             kind:document.getElementById('kind'), search:document.getElementById('search')};

function esc(s){ return (s||'').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

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
    const el = document.createElement('div'); el.className='card';
    el.innerHTML = `<h3>${esc(r.title)}</h3>
      <div class="tags">
        <span class="tag kind">${esc(r.kind)}</span>
        <span class="tag">${esc(r.category)}</span>
        <span class="tag">${esc(r.difficulty)}</span>
        ${r.built? '<span class="tag">built</span>':''}
      </div>
      <div class="desc"><code>${esc(r.name)}</code></div>
      <div class="meta"><span>flag: <code>${esc(r.flag)}</code></span></div>`;
    grid.appendChild(el);
  }
}
for (const k in sel) sel[k].addEventListener('input', refresh);
loadFacets().then(refresh);
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/facets":
            self._json(catalog.facets())
        elif parsed.path == "/api/challenges":
            q = parse_qs(parsed.query)
            self._json(catalog.query(
                category=(q.get("category") or [None])[0],
                difficulty=(q.get("difficulty") or [None])[0],
                kind=(q.get("kind") or [None])[0],
                search=(q.get("search") or [None])[0],
            ))
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
