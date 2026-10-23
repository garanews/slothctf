#include <stdio.h>
#include <string.h>
static unsigned char data[] = { 0x29, 0x36, 0x35, 0x2e, 0x32, 0x21, 0x28, 0x69, 0x39, 0x35, 0x2c, 0x3f, 0x28, 0x3f, 0x3e, 0x05, 0x3c, 0x28, 0x35, 0x37, 0x05, 0x2a, 0x3f, 0x27 };
int main(void) {
    char b[256];
    if (!fgets(b, sizeof b, stdin)) { printf("no input\n"); return 0; }
    size_t n = strlen(b);
    if (n && b[n-1] == '\n') b[--n] = 0;
    if (n != sizeof data) { printf("Denied\n"); return 0; }
    for (size_t i = 0; i < n; i++)
        if (((unsigned char)b[i] ^ 0x5a) != data[i]) { printf("Denied\n"); return 0; }
    printf("Correct\n");
    return 0;
}