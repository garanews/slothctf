# Sloth's Password Vault

**Category:** Reverse Engineering  
**Difficulty:** easy

Enter the correct password to gain access to Sloth's secrets.

## Files

- `bin/challenge` — the binary to reverse (`.exe` on Windows)

## Goal

Reverse the binary to find the secret input it accepts. Run it, type a candidate on standard input, and keep going until it accepts you.

## Run

```
./bin/challenge
```
