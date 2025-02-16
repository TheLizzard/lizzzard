class A:
    X = 0

import sys
n = int(sys.argv[1])

while A.X < n:
    A.X += 1

print(A.X)