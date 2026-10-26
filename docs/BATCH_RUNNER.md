# Batch Campaign Runner

Runs a configured campaign of CTF generations sequentially, with crash-safe resume and
automatic statistics. It replaces the throwaway campaign scripts: one config file, one
command, resumable, reproducible. It orchestrates the existing generators and changes
none of them.

## The three commands

```bash
# 1) start a new campaign from a config file
PYTHONPATH=. python -m slothctf.batch run    --config examples/campaigns/agentic_recover_elf_easy_15.json

# 2) resume an interrupted campaign (no run is ever repeated); the tunnel may have moved
PYTHONPATH=. python -m slothctf.batch resume --dir generated/agentic_recover_elf_easy_15 --ollama-url http://NEW-TUNNEL:11434

# 3) (re)generate the Markdown + CSV report from the collected data
PYTHONPATH=. python -m slothctf.batch report --dir generated/agentic_recover_elf_easy_15
```

`--ollama-url` on `resume` is optional — pass it only when the endpoint has changed.

## Config file

```jsonc
{
  "model": "qwen2.5-coder:32b",          // Ollama model (its digest is pinned in the fingerprint)
  "ollama_url": "http://127.0.0.1:11434",// transport only — NOT part of the fingerprint
  "prompt": "Design an original ...",     // one neutral prompt, identical for every run
  "matrix": {
    "genres":       ["recover"],          // recover | reverse | log
    "difficulties": ["easy"],             // easy | medium | hard
    "targets":      ["elf"],              // elf | pe
    "workflows":    ["agentic"]           // agentic (recover only) | monolithic
  },
  "repeats": 15,                          // generations per matrix cell
  "inference": { "temperature": 0.7, "num_ctx": 4096 },
  "budget": { "max_llm_calls": 24, "max_repairs_per_stage": 4, "max_wall_seconds": 1800 },
  "isolation": "isolated",                // ONLY 'isolated' is accepted (no native fallback)
  "dest": "agentic_recover_elf_easy_15"   // campaign folder name under generated/
}
```

The plan is the Cartesian product of the matrix × `repeats`, each with a stable id
`"{genre}_{difficulty}_{target}_{workflow}_{NN}"`, executed in order. Each generation
gets an independent random flag from the generator.

Caveats (the generators are frozen, so these are honoured, not overridden):
- **agentic** applies `inference.temperature`; `num_ctx` is the transport's fixed 4096 (recorded in the fingerprint for the record).
- **monolithic** uses the generator's own transport; its `budget.max_attempts` is honoured, other inference params are recorded but applied via the generator defaults.

## Fingerprint & reproducibility

Each campaign writes `manifest.json` with a **fingerprint** = sha256 of
`commit + model + digest + prompt + inference + budget + matrix + repeats + isolation`.
The **Ollama URL is excluded on purpose**: the tunnel can change between a run and its
resume as long as the model digest is unchanged. On `resume` the fingerprint is
recomputed from the current commit and the live digest and must match — a changed
commit, model, config or digest refuses the resume; a changed URL does not (it is logged
as an `endpoint_change` in the operational log).

## Resume & interruptions

- **Completed** model generations (`valid` / `model-failure`) are never repeated.
- A caught transport fault is recorded as **`infra`**; the campaign stops (no auto-retry), everything is preserved.
- A process **killed** mid-generation leaves an `attempt_start` with no result. On the next
  `resume` it is classified as **`incomplete`**, preserved, and a fresh attempt is created —
  it is *not* counted as a model failure, and the earlier attempt is not overwritten.
- `campaign.jsonl` is **append-only** with per-line `fsync`; a truncated trailing line from
  an interrupted write is tolerated on read.

## Outputs (all local, under `generated/<dest>/`)

- `manifest.json` — config, fingerprint, commit, model, digest, current URL, plan.
- `campaign.jsonl` — append-only operational log and per-attempt metrics (no flags, no transcripts).
- `report.md` / `report.csv` — statistics (see below).
- `<run-id>/attempt_NN/` — the full generation artifacts (including the organizer-only
  `agentic_state.json`, transcript, solver, solution, flag — kept here, never copied out).
- `_valid_playtest/<run-id>/` — for each fully valid challenge, the **player-only** files
  (binary + README) for a blind playtest.

## Statistics

`report.md` / `report.csv` separate, deliberately:
- **Completed** model generations — the only success-rate denominator.
- **Model** successes / failures, with failure-by-stage.
- **Infra-interrupted** and **incomplete/killed** attempts — reported separately, excluded from the denominator.
- Calls / tokens / duration (median + range), effective repairs, no-progress events; cost per valid challenge.
- A **heuristic** mechanic-family breakdown, clearly marked as not a verified property.
- Unavailable token counts are left blank — never shown as zero.

## Safety

Isolated (Docker) execution only — no non-isolated native fallback. Nothing is published.
Confidential data (flags, transcripts, solver, solution) stays inside each run directory;
it is never written into `campaign.jsonl` or the reports. The model digest is verified
before a run and before a resume.
