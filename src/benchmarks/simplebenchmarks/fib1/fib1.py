def fib(x):
    if x < 1:
        return 1
    return fib(x-2) + fib(x-1)

import sys
n = int(sys.argv[1])
print(f"fib({n})={fib(n)}")