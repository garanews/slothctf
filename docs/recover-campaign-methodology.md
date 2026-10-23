# recover campaign — methodology

A measurement campaign for the `recover` genre with **real Qwen generations** (not
the pre-recorded responses used in tests). It exists to answer three questions and
nothing more — it is not another development cycle.

1. **Does it work?** The technical success rate of autonomous generations.
2. **Does it produce different challenges?** The variety of obfuscation techniques.
3. **Are they interesting?** Difficulty and obvious shortcuts — a challenge solved
   in 30 seconds with `strings` is not the same as one that needs 15 minutes of
   static analysis, even if both pass the automated checks.

The tool (`tools/run_recover_campaign.py`) drives the existing generator,
verifiers and backends **unchanged**.

## Reproducibility

- Scenarios are frozen in `tools/recover_campaign/scenarios.json`: both targets
  (ELF, PE) × three difficulties (easy, medium, hard), with **neutral prompts** that
  do not steer the obfuscation technique (we measure what Qwen chooses). Pilot
  sample: **18 runs** (6 scenarios × 3 repetitions).
- Every record stores provenance: tool version, git commit, model name and (online)
  its Ollama digest, inference parameters, platform and Python version, plus the
  backend and target. Do not edit prompts/model/reps mid-campaign — bump `version`
  and start a new dataset.
- The dataset is an **append-only JSONL**; the runner is **resumable** (it skips
  run_ids already present), so an interrupted campaign neither duplicates nor loses
  results. Records are written only after a run completes; an interruption mid-run
  simply re-runs that one on resume. Before resuming, the runner **repairs** the
  dataset: a torn trailing fragment (an incomplete final line from a hard kill) is
  dropped via a temp-file + atomic replace, so the next append starts on a clean
  line and no complete record is lost.
- Each record carries a **campaign fingerprint** (a hash of scenarios + prompts +
  model + reps + max_attempts + inference params + tool version + git commit). On
  resume, if the current fingerprint does not match the dataset's, the runner
  **refuses with an explicit error** rather than mixing configurations/code into one
  dataset — bump `version` and start a new JSONL instead.
- Provenance also records the **backend** actually used and, when available, the
  **Docker image id**; the model digest is read from Ollama `/api/show` with a
  fallback to `/api/tags`.

## What is recorded per run

Final outcome and (on failure) the original stage; number of attempts and total
time; the per-attempt stages; and a **cause category** assigned conservatively:

- **ok** — success.
- **infra** — a genuine infrastructure error we caught (a backend exception such as
  `SandboxUnavailable` or a Docker error).
- **undetermined** — the cause cannot be attributed with certainty: `solver-error`,
  `swap-solver-error`, `challenge-timeout`, or a failed model call (`model-error`).
  These may be the generated code's fault OR the environment's, so they are **not**
  counted as infrastructure.
- **verification-failure** — the generated challenge failed a concrete check
  (compile, schema, functional accept/reject, leak, `solver-mismatch`,
  `solver-hardcoded`, `too-trivial`, ...), attributable to the content.

The original stage is always preserved alongside the category.

## Heuristic quality measures (labelled as heuristics)

On each **successful** challenge the tool also records, clearly flagged as
heuristics, never as ground truth:

- **Transform family** — a coarse tag set from the organizer's encoder/solver
  source (xor, add/sub, rotate, s-box/table, named cipher, position-dependent), and
  a `family_signature`. Variety is the number of distinct signatures over the
  successful set.
- **Obvious shortcuts** — whether the plaintext flag is recoverable by a `strings`
  scan (UTF-8/UTF-16LE), or by a **single-byte XOR/ADD sweep** of the whole binary.
  A sweep counts as a shortcut ONLY when it recovers the **exact expected flag**; a
  stray `sloth{...}` that is not the real flag is recorded as a `candidate`, not a
  shortcut (false positives are not counted). These catch the laziest attacks only.
- **Structural** — binary size and byte entropy.

**Important:** a challenge is **not** labelled hard because the naive attacker
failed — that only rules out the laziest attack. Real difficulty comes from the
human playtest, below.

## Gameplay quality — human, not LLM

No LLM assigns a final difficulty score. A human fills in
`tools/recover_campaign/playtest_scorecard.md` per challenge: time to solve, tools
used, hints needed, perceived difficulty, unintended shortcuts, and interest.

### Note from the first (pre-campaign) playtest

The two hand-made example binaries were solved from the distributed binaries alone
using **static-analysis / disassembly tooling** (not necessarily the Ghidra GUI).
Their transform was **position-dependent addition combined with XOR** — not a
XOR-free scheme. Both are fine introductory challenges but use essentially the same
mechanic, so they do not yet demonstrate variety or advanced difficulty. Measuring
exactly that variety across real generations is the point of this campaign.

## Limits

- Technical success ≠ good gameplay. The automated checks prove a recovery path
  exists and the program is a working validator; they do not prove the challenge is
  interesting or appropriately hard.
- Heuristics can mislabel (e.g. a custom transform tagged only "position_dependent").
  Treat the family histogram as a signal, not a taxonomy.
- The naive attacker is intentionally weak; absence of a shortcut is weak evidence.
- Timeouts are classified as infrastructure, but a model that writes a slow program
  can also cause them — inspect the stage/detail when it matters.

## Outputs and data handling

- **Raw dataset** (JSONL) is kept **locally** and is the source of truth; it contains
  the plaintext flags and local paths and is not published.
- **Aggregate report** (Markdown) and **CSV** are produced by `aggregate`. The report
  keeps **technical metrics** (success per scenario — all six target × difficulty —
  by target, by difficulty, failure distribution by stage, attempts used) clearly
  separated from the **heuristics** (variety, shortcuts).
- **Publication** uses `anonymize`, which writes a separate copy with the flag hashed
  and the flag (plus its inner secret) and any local Windows path **redacted from
  every string field** — `detail`, `quality_error`, the naive-attacker candidate,
  everything — not just the top-level `flag`. Host details are dropped. It **never
  alters the original dataset**. Scrubbed, publishable datasets live under `campaign/`.
