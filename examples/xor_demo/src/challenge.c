#include <stdio.h>

static unsigned char data[] = { 0x31, 0x2e, 0x2d, 0x36, 0x2a, 0x39, 0x27, 0x23, 0x31, 0x3b, 0x1d, 0x3a, 0x2d, 0x30, 0x3f };

int main() {
    size_t len = sizeof(data) / sizeof(data[0]);
    unsigned char key = 0x42;
    for (size_t i = 0; i < len; i++) {
        data[i] ^= key;
    }
    printf("%s\n", data);
    return 0;
}