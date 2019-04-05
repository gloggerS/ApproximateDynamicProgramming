# Enter your code here. Read input from STDIN. Print output to STDOUT
from statistics import median

n = int(input())
a = [int(x) for x in "10 40 30 50 20 10 40 30 50 20 1 2 3 4 5 6 7 8 9 10 20 10 40 30 50 20 10 40 30 50".split()]
b = [int(x) for x in "1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10 10 40 30 50 20 10 40 30 50 20".split()]

c = [[x] * y for (x,y) in zip(a, b)]
c = [item for sublist in c for item in sublist]
c.sort()

t = int(len(c)/2)
if len(c)%2 == 0:
    L = c[:t]
    U = c[t:]
else:
    L = c[:t]
    U = c[(t+1):]

print(round((median(U) - median(L))*1.0, 1))

