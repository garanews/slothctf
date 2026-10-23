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
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from slothctf import catalog
from slothctf.execution import (
    NativeBackend,
    docker_available,
    image_available,
    resolve_backend,
)
from slothctf.generator import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    generate_challenge,
)
from slothctf.targets import DEFAULT_TARGET
from slothctf.validator import validate_challenge

GENERATED = Path("generated")

# In-memory job store for browser-triggered generations (they take minutes, so
# the POST starts a worker thread and the page polls for the result).
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _safe_name(name: str) -> str | None:
    name = (name or "").strip()
    if not name or "/" in name or "\\" in name or name in (".", "..") or len(name) > 64:
        return None
    return name


def _build_generate_kwargs(payload: dict) -> dict:
    """Map a browser form payload to generate_challenge kwargs (validated)."""
    name = _safe_name(payload.get("name", ""))
    if name is None:
        raise ValueError("invalid or missing challenge name")
    genre = "log" if str(payload.get("genre")) == "log" else "reverse"
    difficulty = payload.get("difficulty") if payload.get("difficulty") in ("easy", "medium", "hard") else "easy"
    try:
        max_attempts = max(1, int(payload.get("max_attempts") or DEFAULT_MAX_ATTEMPTS))
    except (TypeError, ValueError):
        max_attempts = DEFAULT_MAX_ATTEMPTS
    target = "pe" if str(payload.get("target")) == "pe" else "elf"
    unsafe = bool(payload.get("unsafe_native"))
    # Resolve the backend up front so an unavailable combination surfaces as a clean
    # job error (no silent fallback). Isolated by default; native only on opt-in.
    backend = resolve_backend(target, "unsafe-native" if unsafe else "isolated", allow_native=unsafe)
    kwargs = {
        "name": name,
        "prompt": str(payload.get("prompt") or "").strip(),
        "genre": genre,
        "difficulty": difficulty,
        "model": str(payload.get("model") or DEFAULT_MODEL).strip(),
        "ollama_url": str(payload.get("ollama_url") or DEFAULT_OLLAMA_URL).strip(),
        "max_attempts": max_attempts,
        "surprise": bool(payload.get("surprise")),
        "roundtrip": bool(payload.get("roundtrip")) and genre == "reverse",
        "adversary": bool(payload.get("adversary")) and genre == "reverse",
        "target": target,
        "backend": backend,
    }
    if payload.get("category"):
        kwargs["challenge_type"] = str(payload["category"]).strip()
    return kwargs


def _report_dict(rep) -> dict:
    return {
        "name": rep.name,
        "success": rep.success,
        "directory": rep.directory,
        "attempts_used": rep.attempts_used,
        "attempts": [{"index": a.index, "ok": a.ok, "stage": a.stage, "detail": a.detail} for a in rep.attempts],
    }


def _run_generate_job(job_id: str, payload: dict) -> None:
    try:
        kwargs = _build_generate_kwargs(payload)
        rep = generate_challenge(return_report=True, **kwargs)
        try:
            catalog.record(rep.directory, valid=rep.success)
        except Exception:
            pass
        with _LOCK:
            _JOBS[job_id] = {"status": "done", "report": _report_dict(rep), "error": None}
    except Exception as exc:
        with _LOCK:
            _JOBS[job_id] = {"status": "error", "report": None, "error": str(exc)}

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


def _search_code(query: str, max_per: int = 6) -> list[dict]:
    """Full-text search inside every challenge's text files (e.g. find 0x26 in
    the sources). Returns per-challenge matches with file + line + snippet."""
    q = (query or "").lower().strip()
    out: list[dict] = []
    if not q or not GENERATED.exists():
        return out
    for d in sorted(GENERATED.iterdir()):
        meta_p = d / "challenge.json"
        if not d.is_dir() or not meta_p.exists():
            continue
        matches: list[dict] = []
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in _TEXT_EXT:
                continue
            rel = p.relative_to(d).as_posix()
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if q in line.lower():
                    matches.append({"path": rel, "line": i, "text": line.strip()[:200],
                                    "group": _classify(rel)})
                    if len(matches) >= max_per:
                        break
            if len(matches) >= max_per:
                break
        if matches:
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            out.append({"name": d.name, "title": meta.get("title", d.name),
                        "kind": meta.get("challenge_kind", "decode"),
                        "category": meta.get("category", ""), "difficulty": meta.get("difficulty", ""),
                        "matches": matches})
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


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SlothCTF Catalog</title>
<style>
  :root { color-scheme: light dark; --bg:#0f1115; --card:#1a1d24; --fg:#e8eaed;
          --muted:#9aa0aa; --accent:#8bd450; --line:#2a2e37; --codebg:#000; --codefg:#e8eaed; }
  :root[data-theme="light"] { color-scheme: light; --bg:#f5f7fa; --card:#ffffff; --fg:#1b1f24;
          --muted:#5b6673; --accent:#2e7d32; --line:#dfe3e8; --codebg:#0f172a; --codefg:#e8eaed; }
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
  .hit { font-size:12px; margin:4px 0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .hit .hitloc { color:var(--accent); margin-right:6px; }
  code { background:var(--codebg); color:var(--codefg); padding:1px 5px; border-radius:5px; font-size:12px; }
  .empty { color:var(--muted); padding:40px 24px; }
  /* modal */
  .overlay { position:fixed; inset:0; background:rgba(0,0,0,.6); display:none; align-items:flex-start; justify-content:center; padding:40px 16px; overflow:auto; }
  .overlay.open { display:flex; }
  .modal { background:var(--bg); border:1px solid var(--line); border-radius:14px; max-width:min(1200px,96vw); width:100%; padding:22px 24px; }
  .modal h2 { margin:0 0 4px; } .modal .sub { color:var(--muted); margin-bottom:14px; }
  .modal h4 { margin:18px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px; }
  .files a { color:var(--accent); display:inline-block; margin:2px 12px 2px 0; }
  .btn { background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:6px 12px; cursor:pointer; font:inherit; }
  .close { float:right; }
  pre { background:var(--codebg); color:var(--codefg); border:1px solid var(--line); border-radius:8px; padding:12px; overflow:auto; max-height:360px; font-size:12.5px; }
  .filebtn { background:none; border:none; color:var(--accent); cursor:pointer; font:inherit; padding:0; margin:2px 12px 2px 0; }
  .panels { display:grid; grid-template-columns:repeat(auto-fill,minmax(460px,1fr)); gap:10px; align-items:start; }
  details.fp { border:1px solid var(--line); border-radius:8px; background:var(--card); overflow:hidden; }
  details.fp > summary { cursor:pointer; padding:8px 12px; font-size:13px; color:var(--fg); user-select:none; list-style:none; }
  details.fp > summary::-webkit-details-marker { display:none; }
  details.fp > summary::before { content:'▸ '; color:var(--muted); }
  details.fp[open] > summary::before { content:'▾ '; }
  details.fp > summary a { color:var(--accent); font-size:12px; margin-left:8px; }
  details.fp .fpbody { margin:0; border:0; border-top:1px solid var(--line); border-radius:0; max-height:420px; }
  a.dlrow { color:var(--accent); display:block; padding:6px 0; }
  .btn.primary { background:var(--accent); color:#06240a; border-color:var(--accent); font-weight:600; }
  #newpanel { display:none; padding:16px 24px; border-bottom:1px solid var(--line); background:var(--card); }
  #newpanel.open { display:block; }
  #newpanel .row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:10px; }
  #newpanel label { display:flex; flex-direction:column; font-size:12px; color:var(--muted); gap:4px; }
  #newpanel textarea { width:100%; min-height:64px; background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:8px; font:inherit; }
  #newpanel input, #newpanel select { min-width:150px; }
  .genstatus { margin-top:8px; font-size:13px; color:var(--muted); white-space:pre-wrap; }
</style>
</head>
<body>
<header>
  <h1><span class="sloth">🦥</span> SlothCTF Catalog</h1>
  <span class="count" id="count"></span>
  <button class="btn" id="themeBtn" style="margin-left:auto" onclick="toggleTheme()">◐ theme</button>
  <button class="btn primary" id="newBtn" onclick="toggleNew()">＋ New challenge</button>
</header>

<div id="newpanel">
  <div class="row">
    <label>Name<input id="f_name" placeholder="my_challenge"></label>
    <label>Genre<select id="f_genre"></select></label>
    <label>Target<select id="f_target"></select></label>
    <label>Difficulty<select id="f_difficulty"></select></label>
    <label>Category (optional)<input id="f_category" placeholder="e.g. cryptography"></label>
    <label>Max attempts<input id="f_attempts" type="number" min="1" value="5" style="min-width:90px"></label>
  </div>
  <div class="row" style="flex-direction:column">
    <label style="width:100%">Prompt<textarea id="f_prompt" placeholder="Describe the challenge…"></textarea></label>
  </div>
  <div class="row">
    <label>Model<input id="f_model" style="min-width:220px"></label>
    <label>Ollama URL<input id="f_url" style="min-width:320px"></label>
    <label style="flex-direction:row; align-items:center; gap:6px; align-self:flex-end"><input type="checkbox" id="f_surprise"> surprise me</label>
    <label id="f_rt_wrap" style="flex-direction:row; align-items:center; gap:6px; align-self:flex-end"><input type="checkbox" id="f_roundtrip"> round-trip gate</label>
    <label id="f_adv_wrap" style="flex-direction:row; align-items:center; gap:6px; align-self:flex-end" title="reject medium/hard 'check' challenges a baseline attacker can crack"><input type="checkbox" id="f_adversary"> adversary gate</label>
    <label style="flex-direction:row; align-items:center; gap:6px; align-self:flex-end" title="run the model's code on the host with NO isolation (development only)"><input type="checkbox" id="f_unsafe"> unsafe native (no sandbox)</label>
  </div>
  <div class="row" style="align-items:center">
    <button class="btn primary" id="genBtn" onclick="submitGen()">Generate</button>
    <button class="btn" onclick="toggleNew()">Cancel</button>
    <span class="genstatus" id="genstatus"></span>
    <span id="sandboxnote" class="meta"></span>
  </div>
</div>

<div class="controls">
  <select id="category"><option value="">All categories</option></select>
  <select id="difficulty"><option value="">All difficulties</option></select>
  <select id="kind"><option value="">All kinds</option></select>
  <select id="model"><option value="">All models</option></select>
  <select id="sort" title="Sort order">
    <option value="created_desc">Newest first</option>
    <option value="created_asc">Oldest first</option>
    <option value="name">Name A→Z</option>
    <option value="title">Title A→Z</option>
    <option value="difficulty">Difficulty</option>
  </select>
  <input type="search" id="search" placeholder="Search name / title…">
  <label style="display:flex;align-items:center;gap:6px;color:var(--muted);font-size:13px"><input type="checkbox" id="incode"> in code</label>
</div>
<main id="grid"></main>

<div class="overlay" id="overlay">
  <div class="modal" id="modal"></div>
</div>

<script>
const grid = document.getElementById('grid'), count = document.getElementById('count');
const overlay = document.getElementById('overlay'), modal = document.getElementById('modal');
const sel = {category:document.getElementById('category'), difficulty:document.getElementById('difficulty'),
             kind:document.getElementById('kind'), model:document.getElementById('model'),
             search:document.getElementById('search')};
function toggleTheme(){ const r=document.documentElement;
  const t = r.dataset.theme==='light' ? 'dark' : 'light'; r.dataset.theme=t;
  try{ localStorage.setItem('sloththeme', t); }catch(e){} }
try{ const saved=localStorage.getItem('sloththeme'); if(saved) document.documentElement.dataset.theme=saved; }catch(e){}

function esc(s){ return (s||'').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function fileUrl(name, path, dl){ return '/api/file?name='+encodeURIComponent(name)+'&path='+encodeURIComponent(path)+(dl?'&download=1':''); }

// ---- New challenge form ----
const F = id => document.getElementById(id);
async function loadConfig(){
  const c = await (await fetch('/api/config')).json();
  for (const g of c.genres){ const o=document.createElement('option'); o.value=o.textContent=g; F('f_genre').appendChild(o); }
  for (const t of (c.targets||['elf'])){ const o=document.createElement('option'); o.value=o.textContent=t; F('f_target').appendChild(o); }
  F('f_target').value = c.default_target || 'elf';
  for (const d of c.difficulties){ const o=document.createElement('option'); o.value=o.textContent=d; F('f_difficulty').appendChild(o); }
  F('f_difficulty').value='easy';
  F('f_model').value=c.model; F('f_url').value=c.ollama_url; F('f_attempts').value=c.max_attempts;
  F('f_genre').addEventListener('change', ()=>{ const rev = F('f_genre').value==='reverse';
    F('f_rt_wrap').style.display = rev?'flex':'none'; F('f_adv_wrap').style.display = rev?'flex':'none'; });
  // Sandbox readiness note: isolated ELF needs Docker + the built image.
  const note = c.docker ? (c.sandbox_image ? 'Docker sandbox ready (isolated ELF).'
      : 'Docker is up but the sandbox image is missing — run: python main.py sandbox-build')
      : 'Docker not detected: isolated runs are unavailable. Tick "unsafe native" to build on the host (PE on Windows), or start Docker for ELF.';
  F('sandboxnote').textContent = note;
}
function toggleNew(){ F('newpanel').classList.toggle('open'); }
async function submitGen(){
  const body = {
    name:F('f_name').value, genre:F('f_genre').value, difficulty:F('f_difficulty').value,
    target:F('f_target').value, category:F('f_category').value, prompt:F('f_prompt').value,
    model:F('f_model').value, ollama_url:F('f_url').value, max_attempts:F('f_attempts').value,
    surprise:F('f_surprise').checked, roundtrip:F('f_roundtrip').checked,
    adversary:F('f_adversary').checked, unsafe_native:F('f_unsafe').checked,
  };
  if(!body.name.trim()){ F('genstatus').textContent='Please enter a name.'; return; }
  F('genBtn').disabled=true; F('genstatus').textContent='Starting…';
  let res;
  try { res = await (await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json(); }
  catch(e){ F('genBtn').disabled=false; F('genstatus').textContent='Request failed: '+e; return; }
  const jid = res.job_id; let t0=Date.now();
  const poll = setInterval(async ()=>{
    const j = await (await fetch('/api/job/'+jid)).json();
    const secs = Math.round((Date.now()-t0)/1000);
    if(j.status==='running'){ F('genstatus').textContent=`Generating… ${secs}s (the model writes code, we compile & verify)`; return; }
    clearInterval(poll); F('genBtn').disabled=false;
    if(j.status==='done' && j.report){
      const r=j.report;
      F('genstatus').textContent = (r.success?'✅ VALID':'❌ FAILED')+` after ${r.attempts_used} attempt(s). `+
        r.attempts.map(a=>`#${a.index}:${a.stage}`).join('  ');
      await loadFacetsRefresh();
      if(r.success) openDetail(r.name);
    } else { F('genstatus').textContent='Error: '+(j.error||'unknown'); }
  }, 1500);
}
async function loadFacetsRefresh(){ // rebuild filter options then refresh grid
  const labels={category:'All categories',difficulty:'All difficulties',kind:'All kinds',model:'All models'};
  for (const k of ['category','difficulty','kind','model']){ const v=sel[k].value; sel[k].innerHTML='<option value="">'+labels[k]+'</option>'; }
  await loadFacets(); await refresh();
}
async function validateChallenge(name){
  const el=document.getElementById('valresult'); if(el) el.textContent='Validating…';
  const unsafe = F('f_unsafe') ? F('f_unsafe').checked : false;  // reuse the form's opt-in
  const r = await (await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name, unsafe_native:unsafe})})).json();
  if(el) el.textContent = r.result || '(no result)';
}
async function deleteChallenge(name){
  if(!confirm('Delete challenge "'+name+'"? This removes its folder from disk and cannot be undone.')) return;
  const r = await fetch('/api/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  if(r.ok){ closeDetail(); await loadFacetsRefresh(); }
  else { const el=document.getElementById('valresult'); if(el) el.textContent='Delete failed.'; }
}

async function loadFacets(){
  const f = await (await fetch('/api/facets')).json();
  for (const key of ['category','difficulty','kind','model'])
    for (const v of (f[key]||[])) { const o=document.createElement('option'); o.value=v; o.textContent=v; sel[key].appendChild(o); }
}
function fmtDate(ts){ try{ return ts? new Date(ts*1000).toLocaleString() : ''; }catch(e){ return ''; } }

async function refresh(){
  const term = sel.search.value;
  if (document.getElementById('incode').checked && term.trim()){
    return refreshCode(term);
  }
  const q = new URLSearchParams();
  for (const k of ['category','difficulty','kind','model','search']) if (sel[k].value) q.set(k, sel[k].value);
  q.set('sort', document.getElementById('sort').value);
  const rows = await (await fetch('/api/challenges?'+q)).json();
  count.textContent = rows.length + ' challenge' + (rows.length===1?'':'s');
  grid.innerHTML = rows.length ? '' : '<p class="empty">No challenges. Generate one with the <b>＋ New challenge</b> button.</p>';
  for (const r of rows){
    const el = document.createElement('div'); el.className='card'; el.onclick=()=>openDetail(r.name);
    el.innerHTML = `<h3>${esc(r.title)}</h3>
      <div class="tags">
        <span class="tag kind">${esc(r.kind)}</span>
        <span class="tag">${esc(r.category)}</span>
        <span class="tag">${esc(r.difficulty)}</span>
        ${r.model? '<span class="tag" title="model">'+esc(r.model)+'</span>':''}
        ${r.built? '<span class="tag">built</span>':''}
      </div>
      <div class="desc"><code>${esc(r.name)}</code></div>
      <div class="meta">flag: <code>${esc(r.flag)}</code> · <span title="created">${fmtDate(r.created_at)}</span></div>`;
    grid.appendChild(el);
  }
}
async function refreshCode(term){
  const rows = await (await fetch('/api/search_code?q='+encodeURIComponent(term))).json();
  const n = rows.reduce((a,r)=>a+r.matches.length,0);
  count.textContent = `${n} match${n===1?'':'es'} in ${rows.length} challenge${rows.length===1?'':'s'}`;
  grid.innerHTML = rows.length ? '' : `<p class="empty">No file contains <code>${esc(term)}</code>.</p>`;
  for (const r of rows){
    const el = document.createElement('div'); el.className='card'; el.onclick=()=>openDetail(r.name);
    el.innerHTML = `<h3>${esc(r.title)}</h3>
      <div class="tags"><span class="tag kind">${esc(r.kind)}</span><span class="tag">${esc(r.difficulty)}</span><code>${esc(r.name)}</code></div>
      ${r.matches.map(m=>`<div class="hit"><span class="hitloc">${esc(m.path)}:${m.line}</span> <code>${esc(m.text)}</code></div>`).join('')}`;
    grid.appendChild(el);
  }
}

// An embedded file viewer: text files open inline in their own panel (lazy
// loaded), binaries show a download link.
function filePanel(name, f, open){
  if(!f.text) return `<a class="dlrow" href="${fileUrl(name,f.path,1)}">⬇ ${esc(f.path)} (${f.size} B)</a>`;
  return `<details class="fp" data-name="${esc(name)}" data-path="${esc(f.path)}" ${open?'open':''}>
    <summary>${esc(f.path)} <a href="${fileUrl(name,f.path,1)}" onclick="event.stopPropagation()">download</a></summary>
    <pre class="fpbody">loading…</pre></details>`;
}
async function loadPanel(p){
  if(p.dataset.loaded) return; p.dataset.loaded='1';
  const body=p.querySelector('.fpbody');
  try { body.textContent = await (await fetch(fileUrl(p.dataset.name,p.dataset.path,0))).text(); }
  catch(e){ body.textContent='(failed to load)'; }
}
async function openDetail(name){
  const d = await (await fetch('/api/challenge/'+encodeURIComponent(name))).json();
  const player = d.files.filter(f=>f.group==='player');
  const author = d.files.filter(f=>f.group==='author');
  // open the main artifacts by default: README and the evidence log / binary note
  const preferOpen = f => /(^|\/)(README\.md)$/i.test(f.path) || /\.log$/i.test(f.path);
  modal.innerHTML = `
    <button class="btn close" onclick="closeDetail()">✕ close</button>
    <h2>${esc(d.title)}</h2>
    <div class="sub"><code>${esc(d.name)}</code> · ${esc(d.category)} · ${esc(d.difficulty)} · ${esc(d.kind)}</div>
    <p>${esc(d.description)}</p>
    <div class="meta">flag: <code>${esc(d.flag)}</code>${d.success_marker?` · success marker: <code>${esc(d.success_marker)}</code>`:''}</div>
    <div style="margin-top:8px"><button class="btn" onclick="validateChallenge('${esc(name)}')">Re-validate</button>
      <button class="btn" style="border-color:#b4433a;color:#e07a72" onclick="deleteChallenge('${esc(name)}')">🗑 Delete</button>
      <span class="genstatus" id="valresult"></span></div>
    ${(d.generation&&d.generation.prompt!==undefined)?`<h4>Generated from</h4>
      <pre style="max-height:160px">${esc(d.generation.prompt||'(none)')}</pre>
      <div class="meta">genre: <code>${esc(d.generation.genre||'')}</code> · model: <code>${esc(d.generation.model||'')}</code> · difficulty: <code>${esc(d.generation.difficulty||'')}</code>${d.generation.roundtrip?' · round-trip':''}${d.generation.surprise?' · surprise':''}${d.generation.input?` · input: <code>${esc(d.generation.input)}</code>`:''}</div>`:''}
    ${(d.tasks&&d.tasks.length)?`<h4>Questions</h4><ol>${d.tasks.map(t=>`<li>${esc(t.question)} <span class="tag">${esc(t.type||'')}</span></li>`).join('')}</ol>`:''}
    <h4>Player files</h4>
    <div class="panels">${player.length? player.map(f=>filePanel(name,f,preferOpen(f))).join('') : '<span class="desc">none</span>'}</div>
    <h4>How the flag is built <button class="btn" id="revealBtn" onclick="toggleReveal()">reveal ▸</button></h4>
    <div id="reveal" style="display:none">
      ${(d.tasks&&d.tasks.length)?`<p>Answers:</p><ul>${d.tasks.map((t,i)=>`<li><b>${i+1}.</b> <code>${esc(t.answer)}</code> <span class="tag">${esc(t.type||'')}</span>${t.evidence_line?` · line ${esc(String(t.evidence_line))}`:''}</li>`).join('')}</ul>`:''}
      <div class="panels">${author.map(f=>filePanel(name,f,false)).join('')}</div>
    </div>`;
  // wire lazy loading + load the ones opened by default
  modal.querySelectorAll('details.fp').forEach(p=>{
    p.addEventListener('toggle',()=>{ if(p.open) loadPanel(p); });
    if(p.open) loadPanel(p);
  });
  overlay.classList.add('open');
}
function toggleReveal(){ const r=document.getElementById('reveal'); const b=document.getElementById('revealBtn');
  const open = r.style.display==='none'; r.style.display=open?'block':'none'; b.textContent=open?'hide ▾':'reveal ▸'; }
function closeDetail(){ overlay.classList.remove('open'); }
overlay.addEventListener('click', e => { if (e.target===overlay) closeDetail(); });
document.addEventListener('keydown', e => { if (e.key==='Escape') closeDetail(); });

for (const k in sel) sel[k].addEventListener('input', refresh);
document.getElementById('sort').addEventListener('change', refresh);
document.getElementById('incode').addEventListener('change', refresh);
loadConfig();
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
        elif path == "/api/config":
            docker_up = docker_available()
            self._json({
                "model": DEFAULT_MODEL,
                "ollama_url": DEFAULT_OLLAMA_URL,
                "genres": ["reverse", "log"],
                "targets": ["elf", "pe"],
                "default_target": DEFAULT_TARGET,
                "difficulties": ["easy", "medium", "hard"],
                "max_attempts": DEFAULT_MAX_ATTEMPTS,
                "docker": docker_up,
                "sandbox_image": docker_up and image_available(),
            })
        elif path.startswith("/api/job/"):
            with _LOCK:
                job = _JOBS.get(path[len("/api/job/"):])
            self._json(job or {"status": "unknown"})
        elif path == "/api/facets":
            self._json(catalog.facets())
        elif path == "/api/challenges":
            self._json(catalog.query(
                category=(q.get("category") or [None])[0],
                difficulty=(q.get("difficulty") or [None])[0],
                kind=(q.get("kind") or [None])[0],
                search=(q.get("search") or [None])[0],
                sort=(q.get("sort") or [None])[0],
                model=(q.get("model") or [None])[0],
            ))
        elif path == "/api/search_code":
            self._json(_search_code((q.get("q") or [""])[0]))
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
                "tasks": meta.get("tasks", []),
                "generation": meta.get("generation", {}),
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

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw or b"{}")
        except Exception:
            self._send(400, b"bad json", "text/plain"); return

        if parsed.path == "/api/generate":
            job_id = uuid4().hex
            with _LOCK:
                _JOBS[job_id] = {"status": "running", "report": None, "error": None}
            threading.Thread(target=_run_generate_job, args=(job_id, payload), daemon=True).start()
            self._json({"job_id": job_id})
        elif parsed.path == "/api/validate":
            d = _challenge_dir(_safe_name(payload.get("name", "")) or "")
            if d is None:
                self._send(404, b"not found", "text/plain"); return
            # Isolated by default; opt into the unsafe host backend explicitly.
            vb = NativeBackend() if bool(payload.get("unsafe_native")) else None
            res = validate_challenge(d, backend=vb)
            self._json({"result": str(res), "success": res.success, "stage": res.stage})
        elif parsed.path == "/api/delete":
            d = _challenge_dir(_safe_name(payload.get("name", "")) or "")
            if d is None:
                self._send(404, b"not found", "text/plain"); return
            try:
                shutil.rmtree(d)
                catalog.index_generated()  # prune the row
                self._json({"ok": True})
            except Exception as exc:
                self._send(500, str(exc).encode(), "text/plain")
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
