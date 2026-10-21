#include <stdio.h>
#include <string.h>

static unsigned char data[] = { 0x31, 0x2e, 0x2d, 0x36, 0x2a, 0x39, 0x31, 0x27, 0x21, 0x37, 0x30, 0x27, 0x32, 0x23, 0x31, 0x31, 0x3f };

int main() {
    char input[256];
    fgets(input, sizeof(input), stdin);
    input[strcspn(input, "\n")] = '\0';
    int len = strlen(input);
    if (len != sizeof(data)) {
        printf("Denied\n");
        return 1;
    }
    for (int i = 0; i < len; i++) {
        input[i] ^= 0x42;
    }
    if (memcmp(input, data, len) == 0) {
        printf("Access Granted\n");
    } else {
        printf("Denied\n");
    }
    return 0;
}