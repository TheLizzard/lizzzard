# The Computer Language Benchmarks Game
# https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
#
# contributed by Jacob Lee, Steven Bethard, et al
# 2to3, fixed by Daniele Varrazzo
# modified by Daniel Nanz

# Converted to use str instead of bytes


import sys


TABLE = ["ACBDGHKMNSRUTWVYacbdghkmnsrutwvy",
         "TGVHCDMKNSYAAWBRTGVHCDMKNSYAAWBR"]

def str_translate(string):
    output = []
    for char in string:
        output.append(TABLE[1][TABLE[0].index(char)])
    return "".join(output)

def show(seq):
    header, seq = seq.split("\n", 1)
    new_seq = str_translate(seq.replace("\n", ""))[::-1]
    print(">" + header)
    for i in range(0, len(new_seq), 60):
        if i%100_000 == 0:
            print(new_seq[i:i+60])

def main(file):
    for seq in file.read().split(">")[1:]:
        show(seq)

with open(sys.argv[1], "r") as file:
    main(file)