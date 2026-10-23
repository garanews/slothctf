# SlothCTF reliability study — methodology

This directory holds the raw data behind the reliability numbers in the write-up,
so anyone can re-aggregate them instead of taking a screenshot on trust.

## What was measured

Whether the self-verifying generation pipeline produces a **valid** challenge, how
often, how many attempts/seconds it takes to converge, and *why* it fails when it
fails — split by challenge kind and by whether an optional gate was on. "Valid"
means `generate_challenge(...)` returned `success=True`: the model-written
artifacts compiled, ran, and passed the independent behaviour check (decode:
stdout == flag; check: correct secret accepted and a wrong one rejected; log: at
least one task's extractor recovered its answer from the real log and did **not**
reproduce it on the decoy). It is **not** a human judgement of challenge quality
or reversing difficulty.

Every run — including failures — is appended as one JSON record to a `.jsonl`, so
the rates are **not** subject to survivorship bias (unlike `main.py stats`, which
only counts challenges that reached disk).

## Frozen setup

| | |
|---|---|
| Code (git), baseline | `6e4e49a` (tag `campaign-baseline`) |
| Code (git), follow-up | `6e4e49a` + a round-trip decode-robustness fix (see *The crash the follow-up found* below) |
| Harness | [`tools/run_campaign.py`](../tools/run_campaign.py), frozen `SCENARIOS` list |
| Model | `qwen2.5-coder:32b` — qwen2, 32.8B params, **Q4_K_M**, digest `b92d6a0bd47e…`, ~18.9 GB |
| Inference | Ollama on a cloud **NVIDIA A100** (Google Colab), reached over a cloudflared tunnel |
| Compile + run | **local**, `gcc (MSYS2) 13.2.0` on the host — only inference is remote |
| Host | Windows 11 (10.0.26200), CPython 3.14.2 |
| `--max-attempts` | 5 (the model gets up to 5 tries, each fed the previous failure) |
| Sampling | Ollama defaults (`format: json`, `num_ctx: 4096`); log-from-prompt uses the model's default temperature |

## Scenarios

Six frozen prompts, grouped so each optional gate is a clean OFF/ON comparison on
the **same** prompt and difficulty:

| id | genre | difficulty | gate |
|---|---|---|---|
| `rev_decode_xor` | reverse (decode) | easy | — |
| `rev_decode_xor_rt` | reverse (decode) | easy | `--roundtrip` |
| `rev_check_pwd` | reverse (check) | hard | — |
| `rev_check_pwd_adv` | reverse (check) | hard | `--adversary` |
| `log_ssh` | log | medium | — |
| `log_cloudtrail` | log | medium | — |

## Runs in this directory

- **`baseline.jsonl`** — the first campaign: all six scenarios × 3 repeats = **18 runs**.
- **`roundtrip_followup.jsonl`** — a targeted follow-up on the most interesting
  result (round-trip OFF vs ON): `rev_decode_xor` and `rev_decode_xor_rt` only,
  × 10 repeats = **20 runs**. Combined with the baseline's 3+3, that is **13 vs 13**.

Each follow-up run used an **isolated working directory** so its `camp_*` artifacts
did not overwrite the baseline's — the two datasets are independent.

## The crash the follow-up found

The first follow-up attempt ran on the *exact* frozen `6e4e49a` and aborted mid-way
(16 of 20 runs) with an `AttributeError` inside the `--roundtrip` gate. Root cause:
the harness ran the compiled challenge with `subprocess(..., text=True)` and the
host's **default cp1252 codec**, so when a round-trip input drove a *non-invertible*
transform to print an un-mappable byte (`0x9d`), decoding threw in a reader thread,
`stdout` came back `None`, and `_roundtrip_check` crashed on `None.strip()`.

That is precisely the case the round-trip gate exists to catch — a transform that
only inverts the chosen flag and emits garbage for other inputs — so it must be
reported as a round-trip failure, not crash the run. The fix decodes binary output
with `encoding="utf-8", errors="replace"` (plus `None` guards), turning the abort
into the intended `FAIL(roundtrip)` verdict. All 63 unit tests still pass.

The 20-run follow-up in `roundtrip_followup.jsonl` was then re-run **cleanly** on the
fixed code. Treat this as a finding of the campaign, not a footnote: an honest
reliability harness caught a latent crash in one of its own optional gates.

Re-aggregate either file (or both together) with:

```bash
python tools/run_campaign.py --report campaign/baseline.jsonl
python tools/run_campaign.py --report campaign/roundtrip_followup.jsonl
```

## Scrubbing

Both files were passed through [`tools/scrub_runs.py`](../tools/scrub_runs.py),
which edits only the human-readable `detail`/`error` fields and leaves all metrics
fields untouched:

- **`sloth{...}` flag bodies → `sloth{REDACTED}`.** The flags were random per run,
  but a published dataset should not ship working secrets. The surrounding
  diagnostic text is kept, so you can still see *why* a run failed.
- **Home paths → `<HOME>`.** A traceback from a model-written generator embedded an
  absolute path (`C:\Users\<name>\...`) that leaked the local username; the home
  prefix is collapsed to `<HOME>` while the useful relative part (`generated\…\
  generator.py`) survives. Covers `C:\Users\<name>`, `/home/<name>`, `/Users/<name>`.

**Intentionally *not* scrubbed — these are challenge content, not secrets:** the
synthetic log answers that appear in `detail`, namely RFC1918 IPs (e.g.
`192.168.1.100`) and the AWS-documentation dummy access keys
(`AKIA…EXAMPLE`). They are the answers to the generated forensics questions and are
not real credentials. A final scan confirmed no username, home path, email, URL, or
non-`sloth{}` secret remains in either file.

## Honest caveats

- **`duration_s` is wall-clock for the whole run** (prompt building + remote
  inference + local compile + verification + any retries), **not** pure inference
  time. It is useful for comparing settings on identical hardware, not as a model
  benchmark, and it says nothing yet about human effort saved — that needs a
  measured manual-authoring comparison, which this study does not include.
- **Disabling a gate means it was not evaluated**, not that the run would have
  passed it. Gate OFF/ON rows are different challenge kinds and are never merged.
- Numbers are small by design (this is a first study, not a benchmark); every rate
  is reported as numerator/denominator with the spread across repeats so the
  sample size is visible.
- The A100 was a shared cloud instance; absolute seconds will differ on other
  hardware.
