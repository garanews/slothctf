# SlothCTF 🦥

**A local, AI-driven factory for reverse-engineering CTF challenges.** Give it a
prompt; a local open-weight model writes the whole challenge in C, and SlothCTF
compiles it, runs it, and proves it actually works — before you ever see it.

Built for the dev.to **"Slow and Steady"** Hacktoberfest challenge: the model is
allowed to take its time and several tries, because a machine does the boring,
error-prone work while you do something else.

## The core idea

Early versions tried to make the model do everything, including the byte-level
arithmetic of hiding a flag. Small and large models alike got that wrong almost
every time (an LLM cannot reliably compute `flag[i] ^ key` across 20 bytes).

So SlothCTF flips it around:

1. The model invents the challenge and writes **two** C programs:
   - an **encoder** (organizer-only) that transforms the secret, and
   - the **challenge** the player gets, with a placeholder where the hidden
     bytes go.
2. SlothCTF **compiles and runs the model's encoder** on the secret to get the
   exact bytes — *the machine does the arithmetic the model is bad at* — and
   injects them into the challenge.
3. It **compiles and runs the challenge** and verifies it end-to-end, feeding any
   failure back to the model for another attempt.

There are **no templates and no per-scheme `if` ladders**: the model is free to
invent any obfuscation (XOR, add, S-box, RC4, TEA, FNV, a bytecode VM, …). The
harness only guarantees correctness.

## Challenge kinds

Reverse / crypto (`--genre reverse`, the default) come in two kinds:

- **`decode`** — the binary decodes an embedded array at runtime and prints the
  flag. Verified by: *run it → stdout equals the flag*.
- **`check`** — the binary reads a line from stdin and accepts only the correct
  secret (password / hash / cipher checks, state machines, anti-debug gates, …).
  Verified by: *feed the secret → success marker printed; feed a wrong input →
  not printed*.

Both are also checked so the plaintext secret is **not** readable via `strings`.
That check means the plaintext is not present as readable bytes in the binary —
it is **not** a proof that a reverse engineer can't recover the secret, and not a
measure of how strong the obfuscation is.

Log analysis (`--genre log`) produces a forensics challenge: one log with
**several questions**. The model writes a Python **generator** (builds a
realistic synthetic log) and, per question, an **extractor**. Each question is
either **extractive** (the answer appears verbatim in the log → verified by a
cited line) or **computed** (a count, distinct total, max… → verified purely by
running the extractor). Every extractor must recover its answer from the log and
must NOT reproduce it on an unrelated log (anti-hardcode), and ambiguous
questions (ties) are rejected.

```bash
python main.py generate --name ssh_case --genre log \
  --prompt "An SSH brute-force where one IP finally logs in as root; the answer is that IP" \
  --model qwen2.5-coder:32b
```

You can also point it at a **real log you already have** and let it build a set
of questions from the evidence. Here the model runs at temperature 0 and must
cite the exact line for extractive answers; we check each answer appears at its
line (or, for computed answers, that the extractor reproduces it) and that no
extractor hardcodes its answer:

```bash
python main.py generate --name my_case --genre log --input /path/to/server.log \
  --model qwen2.5-coder:32b
```

## Requirements

- **[Ollama](https://ollama.com)** running a capable **coding** model. A small
  model cannot follow the two-program protocol; the sweet spot is
  **`qwen2.5-coder:32b`** on a GPU with enough VRAM (A100-class). Smaller coder
  models work for simple cases but need more attempts.
- A C compiler on `PATH` (**MinGW / MSYS2 `gcc`** on Windows, `gcc`/`clang` on
  Linux/macOS). The compiler and the challenge run **locally**; only model
  inference can be remote.
- Python 3.9+ (standard library only — no pip dependencies).

## Quick start

```bash
# with a local Ollama on the default localhost:11434
python main.py generate --name xor_demo \
  --prompt "Create an easy reverse engineering challenge with XOR encryption" \
  --model qwen2.5-coder:32b --max-attempts 4

python main.py validate --dir generated/xor_demo
```

Point at a **remote** Ollama (e.g. a workstation over WireGuard, or a cloud GPU
behind a tunnel — see [docs/remote-ollama.md](docs/remote-ollama.md)):

```bash
python main.py generate --name pwd_demo \
  --prompt "Ask for a password on stdin; print Access Granted only if correct. Obfuscate the password so strings can't reveal it." \
  --model qwen2.5-coder:32b \
  --ollama-url https://your-ollama-host
```

Other flags: `--category <hint>`, `--difficulty easy|medium|hard`,
`--surprise` (model invents the scenario), `--max-attempts N`
(raise it on a bigger machine). `SLOTHCTF_OLLAMA_URL` sets the endpoint for the
whole shell.

### Round-trip correctness gate (`--roundtrip`)

For reverse challenges, after the chosen flag verifies, `--roundtrip` re-runs the
real pipeline (encode → inject → compile → decode) on a handful of **random**
inputs and requires `decode(encode(x)) == x` for each:

```bash
python main.py generate --name xor_demo --roundtrip \
  --prompt "Create an easy reverse engineering challenge with XOR encryption" \
  --model qwen2.5-coder:32b
```

This gives **additional confidence that the transform is invertible across
different inputs** — not just the chosen flag — so it catches overflow /
signed-unsigned / modulo-edge bugs that happen to work for one flag. It is a
property-based check over a few random samples (currently 4, short ASCII), **not
a proof** that the transform inverts every possible input. It reuses the model's
actual encoder and challenge (no model-written test), so it checks the real code.
([Idea from a reader on the write-up.](https://dev.to/garanews/i-was-too-lazy-to-write-ctf-challenges-so-i-built-a-local-ai-that-writes-compiles-and-checks-its-1927))

### Reference-attacker gate (`--adversary`)

Correctness is not difficulty. `--adversary` runs a budgeted reference attacker
against `check` challenges — a dictionary of common secrets plus a short
brute-force — and **rejects a medium/hard challenge that it cracks** (it's too
easy for its label). `decode` challenges solve by being run, so the gate doesn't
apply to them.

The budget **scales with the declared difficulty** (a `hard` challenge must
resist a stronger attacker than a `medium` one — defaults: medium 800 tries/8s,
hard 2000/20s) and is tunable with `--adversary-budget ATTEMPTS,SECONDS`. The
dictionary is kept deliberately small — a lazy first-guess attacker, not a
determined reverser — so the gate flags *trivially guessable* secrets without
rejecting challenges that are genuinely fine for humans.

This is a **first automatic filter against trivially guessable secrets, not a
global difficulty assessment**: it only measures resistance to *this* attacker.
A challenge with a random 20-char secret passes the gate yet could still be
reversed in seconds by reading the binary — judging true reversing difficulty is
out of scope. ([Idea from a reader on the write-up.](https://dev.to/garanews/i-was-too-lazy-to-write-ctf-challenges-so-i-built-a-local-ai-that-writes-compiles-and-checks-its-1927))

**Both gates are opt-in and run only at generation time** — `python main.py
validate` re-checks compile / behaviour / strings-leak, but does **not** re-run
`--roundtrip` or `--adversary`.

## What gets generated

```text
generated/<name>/
  challenge.json        # title, category, difficulty, description, flag, kind
  README.md             # player-facing brief
  solution.md           # organizer walkthrough (+ how the secret was embedded)
  src/challenge.c       # the player's challenge (encoded bytes already injected)
  organizer/encoder.c   # organizer-only: how the bytes were produced
  bin/challenge[.exe]   # compiled binary
```

See [examples/](examples/) for a `decode` and a `check` sample.

## Generate a whole set at once

[`tools/batch_logs.py`](tools/batch_logs.py) runs a broad set of log-analysis
scenarios in one go (SSH, Nginx/Apache, WAF, DNS tunneling/DGA, firewall scans,
Zeek, mail, Docker/Kubernetes, AWS/GCP/Okta, Sysmon, PowerShell, auditd, DB),
printing a VALID/FAIL summary:

```bash
python tools/batch_logs.py --model qwen2.5-coder:32b --ollama-url https://your-ollama
python tools/batch_logs.py --model qwen2.5-coder:32b --only dns_tunnel,cloudtrail_key
```

## Browse your challenges

Every generated challenge is indexed in a small SQLite catalog
(`generated/catalog.db`). List them from the CLI:

```bash
python main.py list                          # all challenges, as a table
python main.py list --kind check --difficulty easy
python main.py list --search xor --json
```

`python main.py stats` prints an offline metrics snapshot of the corpus — counts
by genre / difficulty / kind / model, how many have a compiled binary, log
richness (questions per log, extractive vs computed), and which optional gates
were used at generation (`--json` for the raw numbers). Note it only counts
challenges that made it to disk (survivors), not failed generation runs.

For an unbiased measurement, [`tools/run_campaign.py`](tools/run_campaign.py)
runs a **frozen** set of scenarios (repeated N times) and logs one JSON record
per run — **including failures** — to `runs.jsonl`, then aggregates honest
metrics:

```bash
python tools/run_campaign.py --model qwen2.5-coder:32b --ollama-url URL --repeats 3
python tools/run_campaign.py --report runs.jsonl
```

It reports, per `genre/difficulty/gates` group: success rate, first-attempt
success, attempts and seconds to converge (on successful runs only), failure
frequency by stage, and gate-rejection count — always as numerator/denominator.
Gates are separate OFF/ON comparisons per challenge kind; disabling a gate means
it was *not evaluated*, not that the run would have passed it.

Or open the **web GUI** (standard library only — no extra install):

```bash
python main.py web            # serves http://127.0.0.1:8000 and opens it
```

From the browser you can: browse and filter by category / difficulty / kind;
open a challenge to see its flag(s), download the player files, and reveal how
it is built (C source / encoder / extractors); re-validate it; and **generate a
new challenge** with the *＋ New challenge* form (pick genre, difficulty, prompt,
model and Ollama URL — it runs the same self-verifying pipeline and streams the
attempt status). The server binds to `127.0.0.1` and runs the model-written code
locally, just like the CLI.

## How a challenge is verified

```
model → encoder.c + challenge.c (with placeholder)
      → compile & run encoder on the secret            → exact bytes
      → inject bytes into challenge.c, compile & run    → behaviour check
      → decode: stdout == flag   |   check: secret accepted, wrong rejected
      → strings(binary) must NOT contain the plaintext secret
      ↺ on any failure, the exact error is fed back for another attempt
```

`python main.py validate --dir <dir>` re-runs this check independently: it
recompiles, runs, and re-checks the `strings` leak for reverse challenges, and
re-runs each extractor against the log for log challenges. It does **not** re-run
the generation-only gates (`--roundtrip`, `--adversary`) or the log anti-hardcode
decoy test — those run when the challenge is generated. `validate` exits non-zero
on failure (usable in CI); the Python API returns a structured result
(`success`, `stage`, `message`).

## Missing compiler

```bash
python main.py install-toolchain --dry-run   # print MinGW/MSYS2 steps
python main.py install-toolchain             # attempt the install (Windows)
```

## Security note

SlothCTF **compiles and runs code written by the model** (C programs, and Python
generators/extractors for the log genre), and the web GUI can trigger that from
the browser. There is no sandbox: steps run as your user with ordinary process
timeouts, not resource/syscall isolation. That's fine for local, single-user use
with a trusted local model, which is the intended setup. If you expose it to
others or run untrusted prompts/models, run it inside a container or VM with CPU,
memory, filesystem and network limits. The web server binds to `127.0.0.1` by
default; don't expose it on an untrusted network.

## Roadmap

- ✅ A small **database** of generated challenges + a **web GUI** to browse them
  by category, difficulty and kind (`python main.py list` / `web`).
- Support for the other classic CTF genres beyond reverse/crypto, each with its
  own deterministic verifier: **log analysis** (next), then **stego**, **pcap /
  network**, and later **memory dumps**, **disk forensics**, **OSINT**.

## Tests

```bash
python -m unittest discover -s tests
```

## License

See [LICENSE](LICENSE).
