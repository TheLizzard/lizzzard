# The Computer Language Benchmarks Game
# https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
#
# contributed by Isaac Gouy

class Tree:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def with_(depth):
        if depth == 0:
            return Tree(None, None)
        else:
            return Tree(Tree.with_(depth-1), Tree.with_(depth-1))

    def node_count(self):
        if self.left is None:
            return 1
        else:
            return 1 + self.left.node_count() + self.right.node_count()

    def clear(self):
        if self.left is None:
            return
        self.left.clear()
        self.right.clear()
        self.left = None
        self.right = None

def main(n):
    MIN_DEPTH = 4
    max_depth = (MIN_DEPTH + 2) if (MIN_DEPTH + 2 > n) else n
    stretch_depth = max_depth + 1

    stretch(stretch_depth)
    long_lived_tree = Tree.with_(max_depth)

    for depth in range(MIN_DEPTH, stretch_depth, 2):
        iterations = 1 << (max_depth - depth + MIN_DEPTH)
        sum = 0
        for _ in range(iterations):
            sum += count(depth)
    print("%i\t trees of depth %i\t check:" % (iterations, depth), sum)

    c = long_lived_tree.node_count()
    long_lived_tree.clear()
    print("long lived tree of depth %i\t check:" % max_depth, c)

def stretch(depth):
    print("stretch tree of depth %i\t check:" % depth, count(depth))

def count(depth):
    t = Tree.with_(depth)
    c = t.node_count()
    t.clear()
    return c

import sys
n = int(sys.argv[1])
main(n)