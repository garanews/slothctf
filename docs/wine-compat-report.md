# Wine compatibility report — SlothCTF PE challenges

Challenges examined: **32**  ·  native ground truth: **yes (Windows host)**

## Summary
- **ok**: 29
- **wine-incompat**: 1
- **not-runnable**: 2

## Matrix
| Challenge | Kind | WinAPI | anti-dbg | anti-VM | compile | native | wine | verdict | cause |
|---|---|:--:|:--:|:--:|:--:|---|---|---|---|
| camp_rev_decode_xor_rt_2 |  |  |  |  | — | — | — | **not-runnable** | missing challenge.json or bin/challenge.exe |
| camp_rev_decode_xor_rt_3 |  |  |  |  | — | — | — | **not-runnable** | missing challenge.json or bin/challenge.exe |
| anti_debug_demo | check | ✓ | ✓ |  | ok | correct | correct | **ok** |  |
| anti_vm_demo | check |  |  | ✓ | ok | correct | correct | **ok** |  |
| camp_rev_check_pwd_1 | check | ✓ | ✓ |  | ok | correct | correct | **ok** |  |
| camp_rev_check_pwd_2 | check | ✓ | ✓ |  | ok | correct | correct | **ok** |  |
| camp_rev_check_pwd_3 | check | ✓ | ✓ |  | ok | correct | correct | **ok** |  |
| camp_rev_check_pwd_adv_1 | check | ✓ | ✓ |  | ok | correct | correct | **ok** |  |
| camp_rev_check_pwd_adv_2 | check | ✓ | ✓ |  | ok | correct | correct | **ok** |  |
| camp_rev_check_pwd_adv_3 | check | ✓ | ✓ |  | ok | correct | correct | **ok** |  |
| camp_rev_decode_xor_1 | decode |  |  |  | ok | correct | correct | **ok** |  |
| camp_rev_decode_xor_2 | decode |  |  |  | ok | correct | correct | **ok** |  |
| camp_rev_decode_xor_3 | decode |  |  |  | ok | correct | correct | **ok** |  |
| camp_rev_decode_xor_rt_1 | decode |  |  |  | ok | correct | correct | **ok** |  |
| constructor_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| custom_b64_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| custom_vm_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| dynamic_imports_demo | check | ✓ |  |  | ok | correct | correct | **ok** |  |
| fnv1a_hash_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| func_pointers_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| junk_code_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| math_equations_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| obfuscation_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| rc4_cipher_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| sbox_crypto_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| seh_exception_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| self_modifying_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| state_machine_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| tea_crypto_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| timing_attack_demo | check |  |  |  | ok | correct | correct | **ok** |  |
| xor_demo | decode |  |  |  | ok | correct | correct | **ok** |  |
| multithread_check_demo | check |  |  |  | FAIL | correct | wrong (correct secret did not print marker 'Acc) | **wine-incompat** | native correct but wine: correct secret did not print marker |

## Focus: WinAPI / anti-debug / anti-VM challenges
- **anti_debug_demo** (WinAPI, anti-debug): ok
- **anti_vm_demo** (anti-VM): ok
- **camp_rev_check_pwd_1** (WinAPI, anti-debug): ok
- **camp_rev_check_pwd_2** (WinAPI, anti-debug): ok
- **camp_rev_check_pwd_3** (WinAPI, anti-debug): ok
- **camp_rev_check_pwd_adv_1** (WinAPI, anti-debug): ok
- **camp_rev_check_pwd_adv_2** (WinAPI, anti-debug): ok
- **camp_rev_check_pwd_adv_3** (WinAPI, anti-debug): ok
- **dynamic_imports_demo** (WinAPI): ok

## Diagnosis of the one wine-incompat: `multithread_check_demo`

Targeted follow-up before merge (the original binary was never modified):

1. **Imports of the original PE** (`objdump -p`): `KERNEL32.dll`, the UCRT
   `api-ms-win-crt-*` set, and **`libwinpthread-1.dll`** — i.e. it was built
   elsewhere with a *dynamic* GCC pthread runtime.
2. **Under Wine with diagnostics** (`WINEDEBUG=err+module`):
   `err:module:import_dll Library libwinpthread-1.dll (which is needed by
   challenge.exe) not found` → `LdrInitializeThunk ... failed, status c0000135`.
   So it is a **DLL-load failure** (the prefix has no `libwinpthread-1.dll`), **not
   a thread-behaviour problem** — the process never starts. Wine's threading is
   not implicated.
3. **MinGW-w64 recompile with correct thread linking** (`-static -pthread`): the
   rebuilt PE imports only `KERNEL32.dll` + `msvcrt.dll` (no `libwinpthread-1.dll`)
   and under Wine prints `Access Granted` for the correct secret and `Denied` for a
   wrong one — fully correct.

**Fix applied (simple, generalizable):** `DockerWinCompatBackend.compile_c` now
links PE with `-static -pthread`, so every PE SlothCTF *generates/validates* is
self-contained and needs no MinGW runtime DLLs under Wine (or on bare Windows).
Covered by a regression test (`test_threaded_pe_runs_under_wine`). This changes
what the backend *produces*; the pre-existing, externally-built original binary is
left untouched and still depends on the missing DLL, which is why its row above
stays `wine-incompat`.

The two `not-runnable` challenges (`camp_rev_decode_xor_rt_2/3`) have **no
`challenge.json`** — a data gap, **not** a Wine issue — and are left as such.

**Scope note:** these results show functional correctness of the challenges
examined under Wine; they are **not** a general proof of Wine's fidelity to real
Windows. VMWindows (Phase C) remains deferred, to be revisited only if a future
challenge needs behaviour Wine does not reproduce.
