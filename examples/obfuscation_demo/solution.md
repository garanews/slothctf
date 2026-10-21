# Solution — Sloth's Password Vault

**Secret input:** `sloth{securepass}`

This is an input-validation challenge. It was verified automatically: feeding the secret above on stdin makes the binary print `Access Granted`, while a wrong input does not. A plain `strings` run does not reveal the secret.

## Recover the secret

```
echo <secret> | ./bin/challenge
```

## How the secret was embedded (`organizer/encoder.c`)

```c
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc != 2) return 1;
    char *secret = argv[1];
    int len = strlen(secret);
    for (int i = 0; i < len; i++) {
        secret[i] ^= 0x42;
    }
    for (int i = 0; i < len; i++) {
        printf("0x%02X", (unsigned char)secret[i]);
        if (i < len - 1) printf(",");
    }
    return 0;
}
```
