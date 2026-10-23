#include <stdio.h>
int main(int c, char **v) {
    for (char *p = v[1]; *p; ++p) printf("0x%02x,", (unsigned char)(*p ^ 0x5a));
    return 0;
}