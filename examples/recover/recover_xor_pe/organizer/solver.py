import sys, re
b = open(sys.argv[1], 'rb').read()
dec = bytes(x ^ 0x5a for x in b)
m = re.search(rb'sloth\{[^}]{1,64}\}', dec)
print(m.group(0).decode() if m else '')