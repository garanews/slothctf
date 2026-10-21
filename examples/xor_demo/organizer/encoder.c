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