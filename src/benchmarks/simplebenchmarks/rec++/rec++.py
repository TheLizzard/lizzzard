def f(x):
    if x > 0:
        return f(x-1)+1
    return 0

import sys
n = int(sys.argv[1])
sys.setrecursionlimit(n<<3)
print(f(n))