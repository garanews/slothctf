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
