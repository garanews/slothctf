# Round-trip gate: what the follow-up found

**The point of this study is not a success rate. It is that SlothCTF, run against
its own frozen code, caught a real crash in its own verification gate — and that a
larger sample corrected a conclusion the first campaign seemed to support.**

## The question

The first 18-run campaign showed reverse/easy `decode` challenges converging
**3/3 with the round-trip gate OFF** but only **1/3 with it ON**, which looked like
"`--roundtrip` badly hurts convergence". Three runs per arm is far too few to
draw that conclusion, so this follow-up added 10 runs per arm.

## Reproduce

```bash
python tools/run_campaign.py --report campaign/baseline.jsonl
python tools/run_campaign.py --report campaign/roundtrip_followup.jsonl
```

## Primary comparison — same (fixed) code, N=10 per arm

Both arms below were generated with the **same corrected code**, so they are a
clean OFF/ON comparison:

| arm | success | 1st-try | gate fired (any attempt) | attempts to converge | seconds |
|---|---|---|---|---|---|
| **OFF** `rev_decode_xor` | **9/10** | 5/9 | 0 | 1.67 ± 0.82 | 21.2 ± 10.2 |
| **ON** `rev_decode_xor_rt` | **9/10** | 6/9 | 2 | 1.67 ± 1.05 | 24.7 ± 12.8 |

**What we can say at N=10:** the large OFF/ON gap seen in the first campaign
**did not reproduce** — here both arms succeed at the same rate, and convergence on
successful runs is similar (ON is a few seconds slower per run because it also
re-compiles and re-runs the pipeline on random probes). **What we cannot yet say:**
whether the gate has *no* real cost. N=10 per arm is too small to rule out a modest
effect; it only shows the initial 3-vs-1 difference was not stable.

## Baseline (kept separate — different code version)

The first campaign's reverse/easy runs were **OFF 3/3** and **ON 1/3**. We keep
these separate and do **not** pool them into the primary comparison, because the
ON arm ran on the *pre-fix* code whose round-trip path could abort (see
`METHODOLOGY.md → The crash the follow-up found`). The two ON baseline failures
were `encoder-crash` (the organizer `encoder.c` segfaulting), a failure mode
unrelated to the gate that happened to land in that 3-run arm.

**Descriptive only:** pooling the OFF arm across both datasets (the fix does not
touch the OFF path) gives **12/13 OFF**. This mixes two code versions and is listed
only as a descriptive figure, not as the primary result.

## The round-trip gate did its job

The gate fired on **2 of 10** ON runs — both cases where the model wrote a
transform that inverts the chosen flag but is **not a true inverse for a sampled
random input** (it appended a stray byte, e.g. `decode(encode(x)) = '…}6' != '…}'`):

- **rep10** — the gate rejected the attempt, the feedback went back to the model,
  and it produced a fully-invertible transform and **converged** on a later attempt.
- **rep3** — the gate rejected the attempt, but the model did not reach a true
  inverse within the 5-attempt budget, so the run **failed** instead of shipping.

This shows, concretely, that the round-trip check **intercepts transforms that are
not invertible for the sampled inputs** — the single-chosen-flag check alone would
have accepted the chosen flag in both runs. We did **not** separately run these two
candidates with the gate OFF, so we don't claim they would definitely have shipped
"valid" without it — only that the extra sampled probes are what exposed the
non-invertibility.

## Recurring failure signature

Across both arms the dominant repair signal is the decode corrupting the **final
byte**: `printed 'sloth{…?'` or `'sloth{…}B'` vs the real `'sloth{…}'`. Single-byte
XOR/add transforms with a signed-char or overflow edge get the last byte wrong. The
self-verifying loop usually repairs it in a second attempt; when it can't, the run
fails rather than shipping a broken challenge.

## `encoder-crash`

Appears in both arms (OFF follow-up rep6; ON baseline) and is independent of the
gate: the model's organizer-only `encoder.c` segfaults on the secret. It is the
main "hard" failure for this prompt and the next thing worth reducing.

## Takeaways

1. **Trust sample size, not one striking run.** A 3-vs-3 split invited the story
   "the gate halves success"; 10-vs-10 on fixed code did not reproduce it. We report
   what N=10 supports and no more.
2. **Keep `--roundtrip` as an opt-in correctness gate.** On this evidence it costs
   little in convergence and it demonstrably catches transforms that are not true
   inverses for sampled inputs.
3. **An honest harness audits itself — that is the real result.** Running on the
   frozen code surfaced a latent crash in one of the gates (now fixed, with a
   regression test). For security tooling, "it caught a flaw in its own verifier" is
   a stronger claim than any headline pass rate.
