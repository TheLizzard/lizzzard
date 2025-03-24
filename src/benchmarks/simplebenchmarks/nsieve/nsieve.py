def main(n):
    flags = [True]*n
    count = 0
    for i in range(2, n):
        if flags[i]:
            j = i << 1
            while j < n:
                flags[j] = False
                j += i
            count += 1
    print("Primes up to " + str(n) + " " + str(count))


import sys
n = int(sys.argv[1])
main(n)