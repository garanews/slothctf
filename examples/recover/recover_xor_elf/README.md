# XOR Vault (ELF)

**Category:** Reverse Engineering  
**Difficulty:** easy

The binary validates input but never prints the flag. Recover it by analysing the binary.

## Files

- `bin/challenge` — the binary to reverse (`.exe` on Windows)

## Goal

Recover the flag `sloth{...}` by ANALYSING the binary — static analysis, a disassembler, extracting and inverting the embedded data. Simply running it does **not** reveal the flag.

## Run (won't hand you the flag)

```
./bin/challenge
```
