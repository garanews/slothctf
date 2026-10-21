# Solution — XORed Flag

**Flag:** `sloth{easy_xor}`

This challenge was verified automatically: the encoder below was compiled and run on the flag to produce the embedded array, the challenge binary was compiled and run, and its output was confirmed to equal the flag. A plain `strings` run does not reveal the flag.

## Recover the flag

The challenge decodes and prints the flag when run:

```
./bin/challenge
```

## How the secret was embedded (`organizer/encoder.c`)

```c
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    unsigned char *flag = (unsigned char *)argv[1];
    size_t len = strlen(flag);
    unsigned char key = 0x42;
    for (size_t i = 0; i < len; i++) {
        flag[i] ^= key;
        printf("0x%02X", flag[i]);
        if (i < len - 1) printf(",");
    }
    printf("");
    return 0;
}
```
