# Example `recover` challenges

Two worked examples of the `recover` genre — the player must **recover the flag by
analysing the binary**, not by running it. Running the program only validates input
(`Correct` / `Denied`); it never prints the flag.

Sources only (no compiled binaries — those are build artifacts and gitignored). The
`src/challenge.c` here already has the obfuscated byte array injected, so it rebuilds
to the exact distributed binary.

| Example | Target | Transform | Flag |
|---|---|---|---|
| `recover_xor_elf` | Linux ELF | single-byte XOR (0x5a) | `sloth{r3covered_from_elf}` |
| `recover_xor_pe` | Windows PE | single-byte XOR (0x5a) | `sloth{r3covered_from_pe}` |

Each directory has:
- `README.md` — the player-facing brief
- `challenge.json` — metadata (`challenge_kind: "recover"`, `target`)
- `src/challenge.c` — the player binary's source (obfuscated bytes embedded)
- `organizer/encoder.c` — how the flag bytes were produced (organizer only)
- `organizer/solver.py` — the reference solver (organizer only; **not** shipped to players)

## Build and solve (reproducible)

Build in the isolated sandbox (see `docs/sandbox-architecture.md`):

```
# ELF
docker run --rm -v "$PWD/examples/recover/recover_xor_elf:/work" -w /work slothctf-sandbox \
  gcc -O2 -s -o bin/challenge src/challenge.c
# PE (MinGW + Wine image)
docker run --rm -v "$PWD/examples/recover/recover_xor_pe:/work" -w /work slothctf-sandbox-win \
  x86_64-w64-mingw32-gcc -O2 -s -static -pthread -o bin/challenge.exe src/challenge.c
```

The reference solver recovers the flag from the compiled binary alone (it reads the
bytes, never other files, and never executes the binary):

```
python organizer/solver.py bin/challenge        # ELF
python organizer/solver.py bin/challenge.exe     # PE
```

These are intentionally *easy* (single XOR) v1 examples that exercise the recover
pipeline end to end. The point is the gameplay shift: the flag is not handed to you by
running the program — you recover it from the artifact (e.g. open it in Ghidra, find
the embedded array, invert the transform).
