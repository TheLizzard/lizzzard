# Base:
#   https://github.com/hanabi1224/Programming-Language-Benchmarks/blob/main/bench/algorithm/coro-prime-sieve/1.py
# Modified by TheLizzard

def filter(generator, prime):
    for i in generator:
        if i % prime != 0:
            yield i

def base_generator():
    i = 2
    while True:
        yield i
        i += 1

def main(n):
    assert n > 0, "ValueError"
    generator = base_generator()
    for i in range(n):
        prime = next(generator)
        generator = filter(generator, prime)
    print(prime)

import sys
sys.setrecursionlimit(100_000)
n = int(sys.argv[1])
main(n)