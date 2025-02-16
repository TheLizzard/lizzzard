# The Computer Language Benchmarks Game
# https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
#
#     line-by-line from Greg Buchholz's C program

# Simplified and removed most IO - TheLizzard


def main(n):
    x = y = limit = 2.0
    Zr = Zi = Cr = Ci = Tr = Ti = 0.0
    img = [0]*n*n

    for y in range(n):
        for x in range(n):
            Zr = Zi = Tr = Ti = 0.0
            Cr = (2.0 * x / n - 1.5)
            Ci = (2.0 * y / n - 1.0)

            i = 0
            while i < 50:
                Zi = 2*Zr*Zi + Ci
                Zr = Tr - Ti + Cr
                Tr = Zr * Zr
                Ti = Zi * Zi
                if Tr+Ti > limit*limit:
                    break
                i += 1
            img[y*n+x] = i
    return hash_img(img, n, n)


def hash_img(img, width, height):
    hash = 0
    for y in range(height):
        for x in range(width):
            x_weight = 1 - (x-width/2)**2 / width
            y_weight = 1 - (y-height/2)**2 / height
            hash += img[y*width+x] * x_weight * y_weight
    return hash/(width*height)**2


import sys
n = int(sys.argv[1])
print(f"{main(n):.8f}")