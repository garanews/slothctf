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