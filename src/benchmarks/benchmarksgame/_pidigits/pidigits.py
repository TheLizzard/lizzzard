# The Computer Language Benchmarks Game
# https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
#
# Translated from Mr Ledrug's C program by Jeremy Zerfas.
# Transliterated from GMP to built-in by Isaac Gouy

# Simplified and removed most IO operations by TheLizzard


def main(n):
    i = k = acc = 0
    den = num = 1

    while i < n:
        k += 1

        # next_term:
        k2 = 2*k + 1
        acc = k2 * (acc + 2*num)
        den *= k2
        num *= k

        if num > acc:
            continue

        # extract_digit
        d = (3*num + acc) // den
        print(k2, acc, den, num, d)
        if d != (4*num + acc) // den:
            continue

        # eliminate_digit:
        acc -= den * d
        acc *= 10
        num *= 10

        i += 1
        if i%1000 == 0:
            print(d, ":", str(i))


import sys
# n = int(sys.argv[1])
n = 10000
main(n)