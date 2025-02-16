# The Computer Language Benchmarks Game
# https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
#
# Contributed by Sebastien Loisel
# Fixed by Isaac Gouy
# Sped up by Josh Goldfoot
# Dirtily sped up by Simon Descarpentries
# Used list comprehension by Vadim Zelenin
# 2to3

# Simplified by TheLizzard (added testing of global variable loading speed)

import math


def eval_A(i, j):
    ij = i+j
    return 1 / (ij*(ij+1)/2 + i + 1)

def eval_A_times_u(u):
    return [sum(eval_A(i,j)*u_j for j, u_j in enumerate(u))
            for i in range(len(u))]

def eval_At_times_u(u):
    return [sum(eval_A(j,i)*u_j for j, u_j in enumerate(u))
            for i in range(len(u))]

def eval_AtA_times_u(u):
    return eval_At_times_u(eval_A_times_u(u))

def main(n):
    u = [1] * n
    for _ in range(10):
        v = eval_AtA_times_u(u)
        u = eval_AtA_times_u(v)
    vBv = vv = 0
    for ue, ve in zip(u, v):
        vBv += ue * ve
        vv  += ve * ve
    print(f"{math.sqrt(vBv/vv):.9f}")

import sys
n = int(sys.argv[1])
main(n)