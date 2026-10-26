# Agentic / test-driven `recover` generation — design proposal

**Status: design only.** No code, no real generations, no deterministic templates,
no `main` change, no merge. This proposes an experimental `--workflow agentic` path
that keeps SlothCTF fully LLM-driven: Qwen invents the mechanic and writes the C
encoder, the C challenge and the Python solver — the harness only runs deterministic
checks and relays their real output back to the model so it can fix its own work
*during* a single generation.

The pilot showed the monolithic path makes the model emit three mutually-consistent
programs in one shot; B (P1+P0) fixed storage/compile/solver-syntax but the bottleneck
moved to *transform round-trip* (`solver-mismatch`) and *validator* (`recover-rejects-correct`)
correctness — exactly the errors an incremental, verify-as-you-go loop can catch early.

---

## 0. Guiding invariant

The **final acceptance gate is unchanged and model-independent**: a challenge is valid
iff it passes the existing `_verify_recover_challenge()` end-to-end (Stage E below),
run exactly as today. The agentic stages only change *how the model arrives* at the
three programs; they never lower or replace the bar. So an agentic challenge is "valid"
by the identical standard as a monolithic one — which is what makes the A/B comparison
meaningful.

---

## 1. Existing functions we reuse (no rewrite)

Everything deterministic already exists and is reused verbatim — the agentic layer is
an orchestrator around these, not a replacement:

| Need | Reused as-is |
|---|---|
| Compile C in sandbox | `backend.compile_c()` (DockerLinux / DockerWinCompat) |
| Run binary / Python | `backend.run_binary()`, `backend.run_python()` |
| Isolated solver run (no flag/sources/manifest) | `backend.run_solver()` + `_run_solver_isolated()` |
| Encoder output → bytes | `_parse_encoder_output()`, `_tokens_to_bytes()` |
| Inject bytes at placeholder | `ENCODED_PLACEHOLDER` replace (same substitution) |
| Functional contract (accept-correct / reject-wrong / crash / timeout / truncate / too-trivial) | `_recover_execution_problem()` |
| Plaintext leak check (UTF-8 + UTF-16LE) | `_flag_leaks_recover()` |
| Anti-hardcode second flag | `_recover_flag_swap()`, `_distinct_random_flag()` |
| Full final gate | **`_verify_recover_challenge()` unchanged** |
| Backend/target selection, isolation, no native fallback | `resolve_backend()`, `get_target()`, `ResourceLimits` |
| LLM transport + test injection | `_call_ollama()`, the `caller` hook |
| Result/record types | `AttemptResult`, `GenerationReport` |
| Output + catalog | `_normalize_challenge()`, `_write_challenge_files()`, `_render_readme/solution()` |

New code is confined to: a stage orchestrator, a `ProjectState`, per-stage prompt
builders, a budget/ledger, and a `--workflow agentic` switch. The monolithic
`generate_challenge()` is untouched; `decode`/`check`/`log`/old `recover` keep working.

The staged gates are the SAME primitives `_verify_recover_challenge()` already calls —
we just call them one component at a time and feed the real error back, then run the
whole `_verify_recover_challenge()` once more at the end as the independent gate.

### One small reuse gap
`_call_ollama()` currently discards Ollama's `eval_count` / `prompt_eval_count`. The
agentic budget (§3) and the fair comparison (§6) need per-call tokens, so we surface
those counts from the transport. That is instrumentation, not a template.

---

## 2. Stages A–E and the state machine

```
A Design ──► B Encoder ──► C Challenge ──► D Solver ──► E Final verify (unchanged)
   (text)      (compile+run)  (inject+compile+        (isolated run +      (full
               → want bytes    functional+leak)        mini flag-swap)      _verify_recover_challenge)
```

- **A — Design.** One LLM call: free-form technical description (mechanic, transform &
  recovery, program structure, obfuscation, difficulty rationale). Stored as prose; it
  is context for later stages, **never** parsed into a fixed algorithm set. No gate
  (nothing to compile) beyond "non-empty".
- **B — Encoder.** Model writes `encoder_code`. Gate = `compile_c` + `run_binary(flag)`
  + `_parse_encoder_output` → the exact `want` bytes, cached in state. Loop here (within
  budget) until it compiles and emits parseable bytes.
- **C — Challenge.** Model writes `challenge_code` (must contain `ENCODED_PLACEHOLDER`).
  Gate = inject `want` → `compile_c` → `_flag_leaks_recover` → `_recover_execution_problem`
  (accept-correct / reject-wrong / no crash/timeout/trivial). Contiguity stays a
  non-fatal diagnostic (P1). Loop until the binary is a well-behaved validator.
- **D — Solver.** Only once the binary works. Model writes `solver_code`. Gate =
  `_run_solver_isolated` on the real binary → must equal the flag, **plus a mini
  flag-swap** (see §4) so hardcoding is caught here with targeted feedback, not only at E.
- **E — Final verification.** Run `_verify_recover_challenge()` **unchanged** as the
  authoritative, model-independent gate. If it passes, `_write_challenge_files()` and
  record success exactly as the monolithic path. If E's flag-swap fails, feed the
  diagnostic back (bounded repairs) — the gate itself never consults the model.

---

## 3. Persisting project state between calls

A `ProjectState` (one per generation) is the single source of truth, held in memory and
mirrored to `base_dir/agentic_state.json` after every step (crash-safe, inspectable,
resumable):

```
ProjectState:
  name, target, difficulty, flag, success_marker
  design: str
  components: { encoder: Component, challenge: Component, solver: Component }
  transcript: [ {stage, role, prompt_tokens, resp_tokens, text_ref, ts} ]
  ledger: Budget counters (§3)
Component:
  code: str                 # current source
  last_good: str | None     # last version that passed its gate
  status: pending|verified|dirty|failed
  gate_detail: str          # last real deterministic error
  artifacts: { want_bytes, bin_path, ... }   # e.g. cached encoder output
```

- `want` bytes from B are cached so C/E don't recompile the encoder needlessly.
- The transcript stores each call's stage, token counts and a reference to the saved
  prompt/response text (for the comparison and for debugging), not inline blobs.
- Writing the state file reuses the same atomic temp-file + `os.replace` + `fsync`
  pattern as the campaign runner's JSONL writer.

---

## 4. Targeted repair without losing correct parts; dependency invalidation

**Repairs are produced by the LLM; the harness only verifies.** No rule-based patching.
On a gate failure the harness hands the model the *real* deterministic output (compiler
stderr, "fed the correct flag, got 'Denied'", "solver returned X, expected the flag",
"on a different binary the same solver returned Y") and asks it to fix the responsible
component — in isolation, with the verified others as read-only context.

Map of failure → component the model is asked to fix (all model-authored fixes):

| Signal | Component to repair |
|---|---|
| encoder compile / bad output | encoder |
| challenge compile | challenge |
| correct flag rejected (`recover-rejects-correct`) | encoder↔challenge consistency |
| leak (`flag-leak`) | challenge |
| solver syntax/crash (`solver-error`) | solver only |
| solver wrong output (`solver-mismatch`) | model reasons about the inverse; fix encoder or solver |
| flag-swap fail (`solver-hardcoded`/`swap-*`) | surface the hardcode; fix solver (or challenge if the flag isn't really embedded) |

**Preserve good work:** each `Component.last_good` holds the last version that passed
its gate; a failed repair never overwrites it (we keep the best-known, and only promote
a new version when it passes). **Dependency invalidation:** encoder → challenge → solver.
If a repair changes the encoder, mark challenge+solver `dirty` and re-run their gates
(re-derive `want`); if it changes the challenge, mark solver `dirty`. We never declare
success on stale downstream verification. A per-stage repair cap (§3) bounds each loop.

---

## 5. Preventing an indirectly-hardcoded solver

This is the real risk of an agentic flow: the model sees the flag and its own design
while writing the solver, so it could embed the flag or flag-specific constants. Four
layers, three of them already in the codebase:

1. **Runtime isolation (existing).** `run_solver()` mounts only the script + the binary,
   read-only, no network, noexec `/tmp`, no `challenge.json`/sources/encoder — so the
   solver cannot *read* the flag at run time regardless of what the model knows.
2. **Mini flag-swap inside Stage D (new use of existing `_recover_flag_swap`).** Before
   leaving D, rebuild the SAME challenge with a second harness-chosen flag and run the
   SAME solver; require it to recover the new flag. This catches a hardcoded flag/constant
   *with targeted feedback during construction*, not only at the end.
3. **Authoritative Stage E flag-swap (unchanged).** The final gate's swap is identical to
   today and model-independent; it is the guarantee, not the prompt.
4. **Prompt hygiene (defense in depth, not a guarantee).** Stage D's prompt frames the
   task as "recover from arbitrary bytes produced by your transform", and we can omit the
   literal flag from the solver prompt where the design allows — but we rely on layers
   1–3, never on hygiene alone.

---

## 6. Budget, timeouts, failures

A `Budget` configures hard caps; a central dispatcher enforces them and the ledger
records every call:

- `max_llm_calls` (whole generation), `max_repairs_per_stage`, `max_wall_seconds`,
  optional `max_tokens` (sum of `eval_count`+`prompt_eval_count`).
- Per-call LLM timeout (already in `_call_ollama`) and per-compile/run limits
  (`ResourceLimits`) stay; a global deadline aborts cleanly.
- **Failure policy:** infra errors (`SandboxUnavailable`, Docker down) abort preserving
  state (category `infra`), never consuming the model. Model/verification failures
  consume per-stage repair budget. On any cap hit → stop, record `budget-exhausted`
  with the partial state. **A challenge is valid only if Stage E passes — never because
  the budget was large.** No infinite loops: every loop is bounded by a cap.

---

## 7. Comparing agentic vs monolithic fairly

Call-count is not comparable (agentic = many small calls, monolithic = few big ones).
The comparable unit is **total compute = summed tokens** (and wall time) across all LLM
calls, which both paths now log. Plan:

- Same model/digest, same scenarios/flags, both run through the staged ELF/PE backends.
- Separate datasets; the `campaign_fp` gains a `workflow` field so agentic and
  monolithic can never mix.
- Report **valid challenges per 1k tokens** and **per wall-second**, plus raw success
  rate, mean tokens/calls/time per *successful* challenge, and the failure-stage mix.
- Give the monolithic baseline a token budget in the same ballpark as the agentic run's
  measured consumption, and **document both budgets explicitly** so the comparison is
  "same compute, which workflow converges?" — not "which got more calls".

The hypothesis to test (not assume): the agentic path converts the pilot's late-stage
failures (`solver-mismatch`, `recover-rejects-correct`) into successes at acceptable
extra token cost. If it does not, that is itself a finding about the model, not the flow.

---

## 8. Compatibility, isolation, creative freedom

- ELF→DockerLinux, PE→DockerWinCompat unchanged; solver always on a Python-capable
  Linux backend (as today). No native fallback.
- `decode`, `check`, `log`, and the existing monolithic `recover` are untouched and
  remain the default. The new path is opt-in: `--workflow agentic` (default `monolithic`).
- **Creative freedom is a hard requirement of the design:** no stage enumerates or
  restricts transforms. Prompts ask for *correctness properties* (compiles on the target,
  uses the embedded data, buffers sized/terminated, solver is valid Python that reads the
  binary) — never *which algorithm*. Composed transforms, position-dependent checks,
  state machines, distributed checks, control-flow obfuscation and original mechanics are
  all allowed; the harness verifies they round-trip, it does not vet the algorithm. (The
  P0 guidance is correctness-only and carries this same line.)
- No prior dataset is modified; agentic runs write to their own dirs.

---

## 9. Build & validation plan (after you approve this design)

1. **Offline prototype first.** Implement the orchestrator with the `caller` hook driven
   by **pre-recorded** staged LLM responses (including deliberately-broken ones to
   exercise each repair path and dependency invalidation). Gated Docker ELF **and** PE
   tests drive real compile/run/solver through the reused primitives — no A100.
2. Only after the offline tests pass do we ask you to authorize a **small real pilot**
   on the A100 with `qwen2.5-coder:32b`, measured against the monolithic baseline per §7.

---

## 9b. Approved clarifications (incorporated)

**Backtracking repair (not strictly sequential).** A solver failure may be caused by the
solver *or* by the challenge, the encoder, or the original design. The orchestrator lets
the agent identify the responsible stage and go back to edit it (within budget). For an
unambiguous gate error (e.g. a compile error) the responsible component is obvious; for
an ambiguous one (`solver-mismatch`, a failed flag-swap) the model is given the full
picture (design + current encoder/challenge/solver + the real gate output) and asked to
diagnose which component is wrong and fix it. Whenever a component changes, every
downstream result that depends on it is invalidated and its gate re-run — so we never
keep a "verified" status that belongs to an older version of an upstream artifact.

**ProjectState keeps, atomically:** the LLM design; current encoder/challenge/solver
code; previous versions of each component; the test feedback; which stages are verified
vs invalidated; the reason for each repair; and the LLM calls with budget consumption.
Atomic save + crash recovery (temp file + `os.replace` + `fsync`). A stage is **never**
treated as done from the saved status alone: on resume, a `verified` flag is honoured
only if it was recorded against the *current* bytes of that component (and its
dependencies); otherwise its gate is re-run. The final solver is still verified in
isolation, with no access to ProjectState, the expected flag, or any private material.

**Budget & experimental cost.** Record per generation: LLM call count; input/output
tokens when available; inference time and total wall time; repairs per stage; failures
with their cause; and total resource consumption per *valid* challenge. This lets us
compare agentic vs monolithic not only on success rate but on computational cost.

## 10. PR #6 (P1/P0) — land it separately

P1/P0 are validated and independent of this workflow. Proposal: **merge PR #6 on its own**
onto `main` first (it improves the existing monolithic path and the shared
`_verify_recover_challenge`), then build the agentic workflow on a **separate branch** on
top of it. That keeps the two efforts, and their histories, cleanly separable — and the
agentic Stage E reuses the P1-fixed verifier. (No merge is performed now; this is the
recommended sequence for your decision.)
