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

`--roundtrip` (reverse genre) adds a stronger correctness gate: after the chosen
flag verifies, the pipeline re-runs encode→inject→compile→decode on several
random inputs and requires `decode(encode(x)) == x` for each. This catches a
transform that only inverts the chosen flag but isn't a true inverse in general
(byte overflow, signed/unsigned or modulo edges) — the classic "perfect-looking
function, constant off by one" bug.

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

Or open the **web GUI** (standard library only — no extra install) to browse
and filter by category, difficulty and kind in your browser:

```bash
python main.py web            # serves http://127.0.0.1:8000 and opens it
```

## How a challenge is verified

```
model → encoder.c + challenge.c (with placeholder)
      → compile & run encoder on the secret            → exact bytes
      → inject bytes into challenge.c, compile & run    → behaviour check
      → decode: stdout == flag   |   check: secret accepted, wrong rejected
      → strings(binary) must NOT contain the plaintext secret
      ↺ on any failure, the exact error is fed back for another attempt
```

`python main.py validate --dir <dir>` re-runs this check independently.

## Missing compiler

```bash
python main.py install-toolchain --dry-run   # print MinGW/MSYS2 steps
python main.py install-toolchain             # attempt the install (Windows)
```

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
