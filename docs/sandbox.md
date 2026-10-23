# Using the execution sandbox

SlothCTF runs model-written code. By default it runs it **isolated** in Docker; there
is no automatic fallback to running it on your host. See
[sandbox-architecture.md](sandbox-architecture.md) for the design and its limits.

## One-time setup (for isolated ELF)

Install Docker (Docker Desktop + WSL2 on Windows, or native Docker on Linux), then
build the sandbox image:

```bash
python main.py sandbox-build          # builds slothctf-sandbox:latest from docker/
python main.py sandbox-check          # reports daemon + image readiness
```

## Generating

```bash
# Linux ELF, isolated (the default target + default isolation)
python main.py generate --name elf_demo --target elf \
  --prompt "easy XOR reverse challenge" --model qwen2.5-coder:32b

# Windows PE .exe — isolated PE is Phase B, so for now build it on a Windows host
# with an explicit, deliberate opt-in to the UNSAFE native backend:
python main.py generate --name pe_demo --target pe --unsafe-native \
  --prompt "easy XOR reverse challenge" --model qwen2.5-coder:32b
```

If a combination is unavailable you get an explicit error (never a silent downgrade):
`--target elf` without Docker, or `--target pe` without `--unsafe-native`, both stop
with a message telling you what to do.

## Validating

```bash
python main.py validate --dir generated/elf_demo                 # isolated (Docker)
python main.py validate --dir generated/pe_demo --unsafe-native  # host, required for .exe in Phase A
```

Existing challenges generated before the `target` field are recognised from the
binary's magic bytes, so they keep validating (use `--unsafe-native` for `.exe`).

## Environment variables

| Var | Effect |
|---|---|
| `SLOTHCTF_EXECUTION` | `docker` (default) or `native` for the plain `get_backend()` factory |
| `SLOTHCTF_ALLOW_NATIVE=1` | opt into the unsafe native backend without the CLI flag |
| `SLOTHCTF_DOCKER_IMAGE` | override the sandbox image tag (default `slothctf-sandbox:latest`) |

## Safety

Docker is a strong boundary, **not** a VM — it shares the host kernel. For hostile
inputs, run SlothCTF inside a dedicated VM. The web GUI can trigger generation; it
binds `127.0.0.1` and obeys the same isolation rules (isolated by default; the
"unsafe native" checkbox is the deliberate opt-in).
