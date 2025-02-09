# The Computer Language Benchmarks Game
# https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
#
# modified by Ian Osgood
# modified again by Heinrich Acker
# modified by Justin Peel
# 2to3

# Modified to keep only randomFasta without most of the io operations

import sys, bisect

alu = (
   'GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG'
   'GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA'
   'CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT'
   'ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA'
   'GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG'
   'AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC'
   'AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA')

iub = list(zip('acgtBDHKMNRSVWY', [0.27, 0.12, 0.12, 0.27] + [0.02]*11))

homosapiens = [
    ('a', 0.3029549426680),
    ('c', 0.1979883004921),
    ('g', 0.1975473066391),
    ('t', 0.3015094502008),
]


def genRandom(ia = 3877, ic = 29573, im = 139968):
    seed = 42
    while 1:
        seed = (seed * ia + ic) % im
        yield seed / im

Random = genRandom()

def makeCumulative(table):
    P = []
    C = []
    prob = 0.
    for char, p in table:
        prob += p
        P += [prob]
        C += [char]
    return (P, C)

def randomFasta(table, n):
    width = 60
    r = range(width)
    gR = Random.__next__
    bb = bisect.bisect
    jn = ''.join
    probs, chars = makeCumulative(table)
    for j in range(n // width):
        x = jn(chars[bb(probs, gR())] for i in r)
        if j % 1000 == 0:
            print(x)
    if n % width:
        print(jn(chars[bb(probs, gR())] for i in range(n % width)))

def main():
    n = int(sys.argv[1])

    print('>TWO IUB ambiguity codes')
    randomFasta(iub, n*3)

    print('>THREE Homo sapiens frequency')
    randomFasta(homosapiens, n*5)

main()