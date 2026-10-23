# Solution — XOR Vault (PE)

**Flag:** `sloth{r3covered_from_pe}`

This is a RECOVER challenge: running the binary does not reveal the flag. It was verified automatically — the obfuscated bytes are embedded in the binary, a plain run does not print the flag (UTF-8 or UTF-16LE strings do not leak it either), and the reference solver below recovers the flag from the distributed artifact alone, in an isolated workspace with no access to this file or the sources. The same solver also recovered an independent, harness-chosen second flag from a freshly built binary, proving it is not hardcoded.

## Recover the flag (organizer reference solver)

```
python organizer/solver.py bin/challenge      # .exe on Windows
```

## How the flag was hidden (`organizer/encoder.c`)

```c
#include <stdio.h>
int main(int c, char **v) {
    for (char *p = v[1]; *p; ++p) printf("0x%02x,", (unsigned char)(*p ^ 0x5a));
    return 0;
}
```

## Reference solver (`organizer/solver.py`)

```python
import sys, re
b = open(sys.argv[1], 'rb').read()
dec = bytes(x ^ 0x5a for x in b)
m = re.search(rb'sloth\{[^}]{1,64}\}', dec)
print(m.group(0).decode() if m else '')
```
