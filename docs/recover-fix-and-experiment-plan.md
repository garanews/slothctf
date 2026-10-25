# `recover` fix (P1 + P0) and A/B experiment plan

Follow-up to `docs/recover-failure-analysis.md` (baseline `43e3619`, 0/18). Two
distinct commits on `fix/recover-verifier-and-prompt`:

- **P1** `2a07306` — verifier: contiguity is diagnostic-only.
- **P0** `a9c0130` — prompt: demand robust C storage/buffers/headers and a Python solver.

No model params (temperature, `num_ctx`), backends, scenarios, prompts of
`decode`/`check`/`log`, or the frozen baseline were changed.

## P1 — verifier (commit `2a07306`)

`_verify_recover_challenge()` no longer rejects on `data-missing`. The old gate
required the encoder's exact byte sequence to appear **contiguously** in the binary:

```python
want = _tokens_to_bytes(body)
if len(want) >= 4 and want not in challenge_bin.read_bytes():
    return AttemptResult(..., "data-missing", ...)   # removed
```

Contiguity is **not necessary** for recoverability (gcc fragments an automatic array
into immediate stores at every `-O`). The check is now a non-fatal diagnostic
(`AttemptResult.diagnostics["encoded_bytes_contiguous"]`), and recoverability is
proven authoritatively by the pre-existing independent checks, all unchanged:

1. the program accepts the correct flag and rejects wrong input (no crash/timeout/trivial);
2. the reference **solver recovers the real flag** from the distributed binary alone;
3. the **same unmodified solver recovers a second, harness-chosen flag** from a
   recompiled binary (anti-hardcode swap);
4. leak / exit-code / truncation checks.

So a working-but-non-contiguous challenge is no longer wrongly rejected, while a
challenge whose data is genuinely unrecoverable still fails — now at the *solver*
step, with an accurate reason, instead of a misleading `data-missing`.

### Important measured caveat — P1 alone will not lift the success rate

For the **simple byte-level solvers** Qwen writes, "`want` contiguous" and
"recoverable" are, in practice, **equivalent**: a sweep/`re.search` solver can only
find the flag where the encoded bytes sit contiguously. Reconstruction of a
fragmented automatic array was tested and does **not** work with a simple static
solver (fails at both `-O0` and `-O2`; `capstone` is not in the sandbox image).
Therefore P1 mostly **reclassifies** the campaign's `data-missing` failures into
`solver-mismatch`/`solver-error` — it does not, by itself, turn them into successes.
The lever that makes challenges both contiguous *and* recoverable is **P0**.

## P0 — prompt (commit `a9c0130`)

The recover schema now additionally requires (correctness only; the obfuscation is
still the model's free choice — `xor/add/sbox/rolling-key` remain examples and "you
invent EVERYTHING" is unchanged):

| Requirement | Fixes (campaign evidence) |
|---|---|
| embedded array declared `static const unsigned char NAME[] = {…}` at **file scope**, and used | `data-missing` 10/18 — automatic arrays fragment; `static const` lands contiguously in `.rodata` |
| correct buffer sizing — `memcmp(…,len)` or NUL-terminate before `strcmp` | `recover-rejects-correct` 3/18 — exact-size buffer + `strcmp` is UB |
| only headers available on the stated BUILD TARGET (no `<windows.h>` on ELF, no POSIX-only on PE) | `challenge-compile` — `windows.h` on an ELF target |
| `solver_code` valid Python 3, `#` comments only, no C/other syntax | `solver-error` — `/* solver.py */` in a Python file |

Prompt size grows to ~1674 tokens — still well within `num_ctx=4096` (Phase 4 of the
analysis showed context is not the bottleneck; all 54 campaign attempts produced
complete JSON).

## Regression tests

- **Unit (no Docker)** `tests/test_recover.py`: the schema carries every P0
  requirement **and** preserves transform freedom.
- **Gated (Docker ELF)** `tests/test_docker_backend.py`:
  - contiguous recoverable challenge **passes** and reports `encoded_bytes_contiguous=True`;
  - **non-contiguous automatic-array** challenge (the `elf-easy#rep3` shape) is **not**
    rejected for contiguity — it proceeds to the solver and, since a naive sweep
    cannot recover fragmented data, fails at `solver-mismatch` (not blessed);
  - a validator that **rejects the correct flag** still fails (`recover-rejects-correct`);
  - **hardcoded** solver (flag-swap), **external-file** solver, and **exec** solver
    negatives still hold.

Full suite incl. ELF and PE gated tests must be green before merge (no `main`
change, no auto-merge).

## A/B experiment plan (pilot first — do NOT auto-run)

Goal: determine whether the fixes let Qwen generate **valid** `recover` challenges
without weakening the checks — not to hit 100%. Three points of comparison, each a
separate dataset (the `campaign_fp` embeds the git commit, so they cannot mix):

| Config | Code commit | Prompt | Verifier | Purpose |
|---|---|---|---|---|
| **Baseline** | `43e3619` | original | old gate | reference (0/18, already have) |
| **A** | `2a07306` (P1) | original | fixed | isolates the verifier fix |
| **B** | `a9c0130` (P0) | improved | fixed | isolates the prompt fix (on top of A) |

- **Pilot size:** `tools/recover_campaign/scenarios_pilot.json` (version 2, reps 1) —
  6 runs per config, one per ELF/PE × easy/medium/hard. Cheap and spread; expand to
  the full 3-reps matrix only after reading the pilot.
- **Isolation:** distinct `--out-dir` and `--jsonl` per config
  (`campaign_recover/pilot_A`, `campaign_recover/pilot_B`); baseline `run1` untouched.
- **Same frozen inputs:** model `qwen2.5-coder:32b`, temperature, `num_ctx=4096`,
  `max_attempts=3`; the pilot scenarios file is identical for A and B.

Per config, when the A100/Ollama is available and **after your go-ahead**:

```bash
git checkout <commit>          # 2a07306 for A, a9c0130 for B
python tools/run_recover_campaign.py run \
  --scenarios tools/recover_campaign/scenarios_pilot.json \
  --out-dir campaign_recover/pilot_<A|B> --jsonl campaign_recover/pilot_<A|B>/raw.jsonl \
  --ollama-url $SLOTHCTF_OLLAMA_URL --limit 1     # smoke first, then resume without --limit
python tools/run_recover_campaign.py aggregate \
  --jsonl campaign_recover/pilot_<A|B>/raw.jsonl \
  --md campaign_recover/pilot_<A|B>/report.md --csv campaign_recover/pilot_<A|B>/report.csv
```

**Read-outs to compare:** overall success n/N; per target/difficulty; and the
failure-stage mix — expect `data-missing` to vanish in A (reclassified to solver
stages), and expect B to convert those into successes if P0 lands. Decide on the
full campaign and any further changes only after the pilot.

**Expected, to be falsified by data:** A ≈ baseline on success count but with
`data-missing` → solver-stage (cleaner attribution); B materially higher success,
driven by `static const` arrays becoming contiguous *and* recoverable. If B does not
improve, the next suspect is the model itself (not prompt or verifier), to be weighed
before any architectural change.
