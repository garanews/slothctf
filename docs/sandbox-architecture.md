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
| **DockerWinCompatBackend** (`x86_64-w64-mingw32-gcc` + **Wine**) | `pe` | strong (container) | Wine ≈ (not real Windows) | Docker + `slothctf-sandbox-win` image | **B (implemented)** |
| **VMWindows** (real Windows VM) | `pe` | strong (VM) | real Windows (anti-debug, WinAPI) | a Windows VM | C (later, maybe) |

## Resolution rules (no silent fallback)

`resolve_backend(target, isolation)`:

- `elf` + isolated → **DockerLinuxBackend** (requires Docker + the image, else an
  explicit `SandboxUnavailable`).
- `pe` + isolated → **DockerWinCompatBackend** (requires Docker + the
  `slothctf-sandbox-win` image, else an explicit `SandboxUnavailable` naming
  `sandbox-build --win`). It cross-compiles the `.exe` with MinGW-w64 and runs it
  under Wine. SlothCTF does not build the `.exe` unsandboxed on its own, and does
  not downgrade it to ELF.
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

### Extra isolation for the Windows-compat (Wine) backend

`DockerWinCompatBackend` keeps every flag above and tightens two things for the run
step. (1) It bind-mounts **only the single `.exe`** under test, read-only, instead of
the whole workspace — so model code under Wine cannot read or modify the sources, the
flag or other binaries. (2) Wine needs a writable `WINEPREFIX`, which would clash with
the read-only root, so each run stages a private prefix in the per-run `/tmp` tmpfs:
the 687 MB static DLL tree (`drive_c/windows`) is **shared read-only** by symlink to a
prefix baked at build time, and only the few MB of mutable state (registry, user dir)
are copied. tmpfs pages count against `--memory`, so this backend runs with a larger
memory floor. Compile uses the normal read-write workspace mount (the `.exe` must land
back on the host). Container isolation (this section) is **independent of** Wine's
emulation fidelity (next section); do not conflate the two.

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
- **Phase B (implemented):** DockerWinCompatBackend (MinGW-w64 + Wine) for *isolated*
  PE, resolver wired (`pe` + isolated → this backend), hardened PE detection (PE
  signature + x86-64 machine, not just `MZ`/extension), `sandbox-build --win`, gated
  live tests, and a Wine **compatibility report** over the existing PE challenges
  (`tools/wine_compat_report.py`) — the result-driven measure of how faithfully Wine
  runs them, especially the WinAPI / anti-debug / anti-VM ones.
- **Phase C (later, only if needed):** VMWindows for faithful isolated PE. The decision
  is driven by the compatibility report: if Wine covers most PE challenges, C stays
  deferred and effort goes elsewhere; the report names the cases that would need it.
