# SlothCTF execution sandbox — architecture

SlothCTF compiles and runs code an LLM wrote (C programs; Python generators and
extractors). That is untrusted code, so this document describes *where* and *how* it
runs, and how the two challenge **targets** stay independent of the host OS.

## Three independent axes

| Axis | What it is | Examples |
|---|---|---|
| **Host OS** | where SlothCTF itself runs | Windows, Linux |
| **Target** | the artifact format a challenge is built for | `pe` (Windows PE x86-64 `.exe`), `elf` (Linux ELF x86-64) |
| **ExecutionBackend** | where/how the model's code is compiled and run, and the isolation level | NativeBackend, DockerLinuxBackend, … |

The target is a property of the **challenge** (recorded in `challenge.json`), not of
the host. You can build an ELF challenge on Windows and (from Phase B) a PE challenge
on Linux. The user picks a target and an isolation level; SlothCTF resolves a backend
that can satisfy both, or raises — it **never** silently falls back to an unsandboxed
run, and **never** converts one target into another.

## Backend × target matrix

| Backend | Target | Isolation | Fidelity | Needs | Phase |
|---|---|---|---|---|---|
| **NativeBackend** (host toolchain) | host-native only: `pe` on Windows, `elf` on Linux | **none — UNSAFE** | real OS | a host compiler | A (legacy/dev, explicit `--unsafe-native`) |
| **DockerLinuxBackend** (`gcc` in a container) | `elf` | strong (container) | real Linux | Docker + `slothctf-sandbox` image | **A (default for elf)** |
| **DockerWinCompat** (`x86_64-w64-mingw32-gcc` + **Wine**) | `pe` | strong (container) | Wine ≈ (not real Windows) | Docker + a win image | **B (next)** |
| **VMWindows** (real Windows VM) | `pe` | strong (VM) | real Windows (anti-debug, WinAPI) | a Windows VM | C (later, maybe) |

## Resolution rules (no silent fallback)

`resolve_backend(target, isolation)`:

- `elf` + isolated → **DockerLinuxBackend** (requires Docker + the image, else an
  explicit `SandboxUnavailable`).
- `pe` + isolated → **not available in Phase A** → explicit error naming Phase B.
  SlothCTF does not build the `.exe` unsandboxed on its own, and does not downgrade
  it to ELF.
- any target + `unsafe-native` → **NativeBackend**, but only after an explicit opt-in
  (`--unsafe-native` / `SLOTHCTF_ALLOW_NATIVE=1`) **and** only if the host toolchain
  emits that target (e.g. a native `elf` build on Windows is refused).

The library default (`generate_challenge`/`validate_challenge` without a `backend`)
is the isolated backend for the target. Because that now requires isolation, omitting
`backend` is **not** behaviourally identical to the pre-sandbox API: a PE challenge or
a host without Docker raises instead of running. Pass an explicit `NativeBackend()` to
keep the old host behaviour on purpose.

## Isolation applied by the Docker backend

Every compile and run is one throwaway `docker run` with: `--network none`, a non-root
user (`--user 65534`), `--cap-drop ALL`, `--security-opt no-new-privileges`, a
read-only root filesystem with a small `--tmpfs /tmp`, `--pids-limit`, `--memory`,
`--cpus`, a wall-clock timeout (compile and run), and captured output truncated to a
byte cap. A timed-out container is force-removed. The workspace is bind-mounted so the
ELF artifact is readable on the host (for the byte-level strings-leak check).

## Limitations — a container is not a VM

Docker shares the **host kernel**. It is a strong boundary against the challenge code
reading your files, reaching the network, or exhausting resources, but a kernel-level
escape is in scope in a way it is not for a hypervisor. For hostile (not merely
untrusted) inputs, run SlothCTF itself inside a dedicated VM. Wine (Phase B) is a
compatibility layer, not Windows: anti-debugging, specific Windows APIs, and some
syscalls behave differently, so a PE verified under Wine is "likely correct", not
"behaves exactly as on Windows" — hence the eventual VMWindows option.

## Roadmap

- **Phase A (this change):** Target concept (pe/elf), DockerLinuxBackend for isolated
  ELF, NativeBackend preserved as explicit unsafe legacy (keeps existing Windows `.exe`
  generation and validation working), CLI/GUI selectors, no-fallback resolver, tests.
- **Phase B (next):** DockerWinCompat (MinGW + Wine) for *isolated* PE, with the Wine
  caveat surfaced in metadata/solution. This is the immediate priority — Phase A alone
  does **not** count as isolated Windows support.
- **Phase C (later, only if needed):** VMWindows for faithful isolated PE. Deferred:
  first measure how many real PE challenges behave correctly under Wine; that decides
  whether C is urgent or whether effort goes to new challenge genres instead.
