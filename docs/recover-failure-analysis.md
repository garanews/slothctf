# `recover` campaign v1 — failure root-cause analysis

**Baseline:** commit `43e3619` (frozen) · model `qwen2.5-coder:32b` (digest `b92d6a0b…ef41`, Q4_K_M) · temp 0.7 · `num_ctx` 4096 · max_attempts 3 · 18 runs (ELF+PE × easy/medium/hard × 3 reps) = **54 attempts** · **0/18 success**.
Dataset: `campaign_recover/run1/raw.jsonl` (untouched). This document is offline forensics only — **no new generations were run, `main` was not modified, nothing was merged.**

## Verdict first: model / prompt / compiler / verifier

| Layer | Role in the 0/18 | Confidence |
|---|---|---|
| **Model** | **Primary cause.** Qwen produces C with concrete correctness bugs (non-nul-terminated buffer compared with `strcmp`, wrong-target `#include`, unbraced array initialisers) and at least one Python solver written with C comment syntax. | Confirmed from preserved sources |
| **Prompt** | **Contributory, cheapest lever.** It does not pin storage class, nul-termination, brace discipline, or hard-forbid wrong-target APIs. It is *not* too large: measured ~1378 tokens, and all 54 outputs were complete valid JSON. | Confirmed (prompt), measured |
| **Compiler `-O2`** | **Framing only, not root cause.** Automatic local arrays fragment at **every** `-O` level (even `-O0`); `-O2` only changes granularity. | Confirmed by recompilation |
| **Verifier** | **One gate is too strict.** `data-missing` (the biggest bucket) rejected at least one fully working challenge — a confirmed false negative. The check is sound in intent but its contiguity assumption only holds for static/global storage. | Confirmed on 1 case; hypothesised for the other 9 |

**Short answer:** this is mostly a *model-quality* result, amplified by a *prompt* that leaves too much unspecified and a *verifier gate* that over-rejects a legitimate code pattern. `num_ctx` and the compiler are **not** the bottleneck.

---

## Phase 1 — forensic distribution (all 54 attempts)

Reconstructed from each run's `stages` array, not just the final stage.

| stage | per-attempt (/54) | final (/18) |
|---|---|---|
| `data-missing` | **26** | **10** |
| `challenge-compile` | 9 | 4 |
| `recover-rejects-correct` | 6 | 3 |
| `solver-error` | 4 | 1 |
| `challenge-crash` | 4 | — |
| `flag-leak` | 2 | — |
| `challenge-bad-exit` | 2 | — |
| `missing-placeholder` | 1 | — |
| **total** | **54** | **18** |

Categories (final): `verification-failure` 17/18, `undetermined` 1/18 (the all-`solver-error` run). **`infra` 0/18, timeouts 0/18** — the A100 tunnel and Docker held throughout; every failure is attributable to generated content, never the environment.

### Artifact inventory & diagnostic limits

Each of the 18 run directories preserved **one** workspace: `organizer/encoder.c`, `src/challenge.c` (post-injection), and the compiled `bin/challenge[.exe]` + `bin/encoder[.exe]`.

- **Inspectable:** the *last* attempt of every run. For the 10 runs whose final stage is `data-missing`, the preserved `challenge.c`/binary **are** a `data-missing` artifact — directly inspectable.
- **Not recoverable:** earlier attempts (the workspace is overwritten in place each attempt), **`solver.py`** (only materialised in an ephemeral isolated solver workspace), **`challenge.json`** (written on success only), and per-attempt compiler/solver **stderr** beyond the 300-char `detail` snippet in the final record.
- **Unreliable:** for the 4 runs whose *final* stage is `challenge-compile`, the preserved binary is **stale** (from an earlier attempt that did compile), so contiguity numbers for those four are not meaningful.

> Cheap future improvement: retain per-attempt artifacts (incl. `solver.py`) and log Ollama's `prompt_eval_count`/`eval_count`. Both would remove the limits above. (Logged as a proposal, not done.)

---

## Phase 2 — `data-missing` (P0, 26/54 attempts, 10/18 final)

The gate in `_verify_recover_challenge()`:

```python
want = _tokens_to_bytes(body)
if len(want) >= 4 and want not in challenge_bin.read_bytes():
    return AttemptResult(0, False, "data-missing", …)
```

This requires the **entire** encoded sequence to appear **contiguously** in the compiled binary.

### What actually happens (confirmed)

Example `elf-easy#rep3` (`src/challenge.c`):

```c
unsigned char flag[] = {0xd9,0xc6,0xc5,0xde,0xc2,0xd1,0x9d,0x9d,0xda,
                        0xcc,0x9e,0x99,0xd9,0xda,0x9a,0x9d,0x9c,0xcb,0xd7};
for (int i=0;i<len;i++) flag[i]^=0xAA;         // decodes to sloth{77pf43sp076a}
if (strcmp(input,(char*)flag)==0) puts("Access Granted"); else puts("Denied");
```

- The array **is** present and **is** used — not dropped.
- In the compiled `-O2` binary the **full 19-byte sequence is absent**; the longest contiguous run is **16/19 bytes**. → `data-missing` fires.
- **Recompiling the same source** in an isolated workspace: full sequence is non-contiguous at `-O0`, `-O1`, `-O2`, `-Os` alike (longest runs 8 / 8 / 16 / 16). So **`-O2` is not the cause** — an *automatic (stack) array* is materialised as inline immediate stores (`movabs` in 8-byte groups) at every level; `-O2` merely merges them into 16-byte runs.
- A control with `static const unsigned char flag[]` at **file scope** is **fully contiguous** at both `-O0` and `-O2` (it lands in `.rodata`). `static` *local* arrays are contiguous too (static storage duration).

**The decisive test:** fed its correct flag on stdin, the `elf-easy#rep3` binary prints **`Access Granted`**. It is a **fully working recover challenge that the gate rejected** — a confirmed **false negative**. The downstream checks (solver-recovers-from-artifact, second-flag swap) were never reached because step 4 fires first.

### Scope survey across all 10 `data-missing`-final runs (confirmed)

All 10 declare the encoded bytes as a **non-static automatic local array**, and all 10 show the data **present but fragmented** (contiguous runs of 8–17 of 19 bytes). The two reliably-contiguous preserved artifacts in the whole campaign both used `static`/`static const`.

- **Confirmed:** the mechanism (automatic local → fragmented; static/global → contiguous), on reproduced recompiles, and ≥1 working challenge wrongly rejected.
- **Hypothesis (not individually proven):** that the *other 9* `data-missing` challenges are likewise functionally correct. They show the same scope and the same present-but-fragmented data, but I could not offline-derive the correct flag for the non-XOR transforms to run each one. This does not lower confidence in the *mechanism*, only in the per-challenge correctness of those 9.

### On the gate itself

Do **not** silently drop or bypass it: a recover challenge must guarantee the flag is recoverable **from the distributed artifact**, including under the second-flag swap. But note the verifier *already* proves recoverability downstream (step 6 solver-recovers-real-flag, step 7 swap-recovers-second-flag). The contiguity test is a **cheap early proxy** whose assumption ("the encoded blob sits contiguously in the file") only holds for static/global storage. It is a strong candidate for replacement — see P1 — but that change needs your sign-off, so it is proposed, not made.

---

## Phase 3 — the other buckets

### `challenge-compile` (9/54, 4/18 final) — confirmed from stderr

Two distinct model errors:

1. **Wrong-target API.** `elf-easy#rep2`: `#include <windows.h>` on a **Linux/ELF** target — even though the prompt's BUILD TARGET line explicitly said *"compiled and run on: Linux ELF (x86-64, gcc)"*. The model ignored an explicit instruction.
2. **Unbraced injected array (the majority).** `pe-easy#rep2`, `pe-hard#rep3`: *"expected identifier or '(' before numeric constant"* with a bare `0x31, 0x2e, …` line; `pe-medium#rep2`: *"invalid initializer"* from `static const unsigned char encoded[] = 0x9c, 0x44, …`. Cause: the model put the `/*__ENCODED_ARRAY__*/` placeholder **outside** the `{ }` of the initialiser, so injecting the comma-separated body yields a syntactically invalid declaration. The injector substitutes a raw body with no braces by design, so the braces must come from the model.

### `recover-rejects-correct` (6/54, 3/18 final) — confirmed from source

`elf-easy#rep1`:

```c
unsigned char decoded[len];                 // len = 19, exact size, NO nul terminator
for (int i=0;i<len;i++) decoded[i]=encoded[i]^KEY;
if (strcmp(input,(char*)decoded)==0) …       // strcmp reads past decoded[18] → UB
```

The decoded buffer is sized to exactly the byte count with **no `'\0'`**, then compared with `strcmp`, which reads until a nul — undefined behaviour reading adjacent stack memory. It rejects the correct flag depending on stack layout. This is **flaky UB**: `elf-easy#rep3` has the same pattern yet printed `Access Granted` (its trailing stack byte happened to be `0x00`). So the same defect can surface as `recover-rejects-correct` *or* pass by luck — a correctness time-bomb either way.

### `solver-error` (4/54, 1/18 final) — confirmed, **not infra**

`pe-easy#rep3` solver, line 1: `/* solver.py */` — a **C-style comment atop a Python file** → `SyntaxError`. All three attempts repeated it. A model output defect; the conservative `undetermined` classification was appropriate, but the evidence shows it is model-side, not infrastructure.

### Minor (per-attempt only, recovered by a later attempt or re-failed otherwise)

`challenge-crash` (4) binary crashed on run; `challenge-bad-exit` (2) non-zero exit; `flag-leak` (2) plaintext flag readable in the binary (`elf-medium#rep1` attempts 1–2); `missing-placeholder` (1) `challenge_code` lacked the placeholder token entirely.

---

## Phase 4 — is `num_ctx=4096` a bottleneck? (measured: **no**)

| measurement | value |
|---|---|
| Attempt-1 recover prompt (ELF, no feedback) | **4959 chars ≈ ~1378 tokens** |
| Generated `encoder.c`+`challenge.c` (preserved) | 863 / 1292 / 2009 chars (min/median/max) |
| Repair feedback appended per retry | the error `detail`, < 300 chars (code is **not** re-sent) |
| Attempts that produced incomplete / unparseable JSON | **0 / 54** |

The prompt is ~1378 tokens; a full output (three programs + JSON metadata) is on the order of 600–1000 tokens; the attempt-1 total is comfortably inside 4096, and repair prompts grow by only a few hundred chars. **Every one of the 54 generations returned complete, schema-valid JSON** — no truncation, no `bad-schema`, no parse retry anywhere in the stage data. So 4096 was sufficient for this workload.

- **Measured:** prompt size; that all 54 outputs parsed fully.
- **Not measured (limit):** exact *output* token counts — Ollama's `eval_count`/`prompt_eval_count` were not logged. Recommend logging them before any future `num_ctx` change so the decision stays data-driven.
- **Recommendation:** do **not** raise `num_ctx` (or temperature, or attempts) now. There is no evidence it would help; the failures are downstream of generation capacity.

---

## Phase 5 — targeted fixes (≤3, prioritised)

| # | Problem | Freq (final) | Cause — verified / hypothesised | Evidence | Proposed change | Regression risk | Acceptance test |
|---|---|---|---|---|---|---|---|
| **P0** | `data-missing` + `recover-rejects-correct` | 13/18 | **Verified:** automatic local array → fragmented bytes; exact-size buffer + `strcmp` → UB | `elf-easy#rep3` works despite gate; `elf-easy#rep1` source; recompile O0/O2; static-vs-local control | **Prompt only:** require the embedded array be declared `static const` at file scope, and require the decoded buffer be nul-terminated (or compared with explicit length, `memcmp(len)`), not `strcmp` on an exact-size buffer | Low — prompt wording; model/verifier/backends unchanged | Re-run the comparative experiment (below); expect `data-missing` and `recover-rejects-correct` to drop sharply |
| **P1** | `data-missing` gate over-rejects working challenges | 10/18 | **Verified** false negative (1 case); gate assumption holds only for static storage | Phase 2 | **Verifier (needs your OK):** demote the strict byte-contiguity test to a non-fatal signal and rely on the existing *solver-recovers-real-flag* + *second-flag swap* checks to guarantee recoverability from the artifact. Keeps all correctness/anti-hardcode guarantees; removes the false-negative class | Medium — changes pass/fail semantics; must keep swap + solver gates intact | New regression tests: (a) a working non-contiguous challenge now passes; (b) a challenge whose solver cannot recover still fails; (c) the flag-swap still catches a hardcoded solver |
| **P2** | `challenge-compile` + `solver-error` | 5/18 | **Verified:** unbraced placeholder; wrong-target `#include`; C comment in Python solver | Phase 3 stderr | **Prompt/schema:** show the array declaration *with* `{ }` around the placeholder; hard-forbid `windows.h`/Win32 on ELF and POSIX-only headers on PE; state the solver must be pure Python 3 (no C syntax) | Low — prompt wording | Expect zero unbraced-initialiser and wrong-header compile failures; no C-comment solvers |

**Not recommended now:** raising `num_ctx`, temperature, or `max_attempts` (Phase 4 shows no support); any change to `decode`/`check`/`log` genres or the Docker backends (out of scope, no demonstrated need).

No deterministic *harness* bug was found that warrants a unilateral fix: the injector, encoder run, and gates behave as designed. P0/P2 are prompt changes and P1 is a verifier change you asked to discuss first — so nothing is being changed in this document.

---

## Next step — small comparative experiment (proposed, not started)

Isolate completely from the baseline so `43e3619`'s 0/18 stays intact for comparison:

- **Separate everything:** new `tools/recover_campaign/scenarios.json` **bumped to `version: 2`** (new `campaign_fp`), new `--out-dir campaign_recover/run2`, new `--jsonl campaign_recover/run2/raw.jsonl`.
- **Change exactly one variable:** the **P0 prompt wording** only. Keep model, digest, temperature, `num_ctx=4096`, `max_attempts=3`, and the 18-cell matrix frozen.
- **Procedure:** `--limit 1` smoke run → verify the record → resume the remaining 17 (same crash-safe, fingerprinted runner).
- **Compare:** success rate overall and per target/difficulty, and especially the `data-missing` / `recover-rejects-correct` / `challenge-compile` bucket sizes, v2 vs the v1 baseline.
- **Decide P1 after** seeing whether the P0 prompt alone clears the gate, or whether the verifier change is also needed.

**The baseline `43e3619` (0/18) remains the untouched reference for all future comparisons.**
