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

Inference parameters are actually applied, so the fingerprint matches what is sent:
- **agentic** applies both `inference.temperature` and `inference.num_ctx`.
- **monolithic** uses the configured `ollama_url` and `budget.max_attempts`; it does **not**
  support `temperature` / `num_ctx` overrides, so specifying them for a monolithic workflow
  is **rejected at config time** rather than silently ignored.

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
- Attempts are classified distinctly, and all preserve their artifacts:
  - **`infra`** — a real transport/sandbox fault (connection, Docker, `InfraError`). Stops the campaign (no auto-retry).
  - **`runner-error`** — an unexpected runner/verifier fault. Stops the campaign; **not** an infra problem and **not** a model failure.
  - **`interrupted`** — an explicit Ctrl-C / `SystemExit`. Recorded, then propagated.
  - **`incomplete`** — a process killed mid-generation (an `attempt_start` with no result). Detected on the next `resume`, preserved, a fresh attempt is created; **not** a model failure.
  None of these count in the model success-rate denominator.

### JSONL integrity (`campaign.jsonl`)

Append-only, per-line `fsync`. Before a run/resume/report the dataset is made safe:
- A **truncated final line** (an interrupted append with no terminating newline) is
  recoverable: the full damaged bytes are first copied to `campaign.jsonl.corrupt-<ts>`,
  then the file is rewritten with only the complete valid records (so a later append never
  concatenates onto damaged bytes).
- A malformed **complete / intermediate** record is **not** silently dropped: the original
  is preserved to a `.corrupt-<ts>` copy and the command **refuses** with an error.
- No valid event is ever silently lost.

## Exit codes (`run` and `resume`)

| code | meaning |
|---|---|
| `0` | campaign completed — even if some challenges failed for model-attributable reasons |
| `2` | usage / config / fingerprint error |
| `3` | stopped by a real infrastructure fault |
| `4` | stopped by an unexpected runner/verifier error |
| `130` | interrupted by the user (Ctrl-C) — data preserved, `resume` to continue |

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
- **Infra**, **runner/verifier-error**, **user-interrupted** and **incomplete/killed** attempts — each reported separately and excluded from the denominator.
- Calls / tokens / duration (median + range), effective repairs, no-progress events; cost per valid challenge.
- A **heuristic** mechanic-family breakdown, clearly marked as not a verified property.
- Unavailable token counts are left blank — never shown as zero.

## Safety

Isolated (Docker) execution only — no non-isolated native fallback. Nothing is published.
Confidential data (flags, transcripts, solver, solution) stays inside each run directory;
it is never written into `campaign.jsonl` or the reports. The model digest is verified
before a run and before a resume.
