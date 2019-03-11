def factors_set():
    factors_set = ((i, j, k, l) for i in [-1, 0, 1]
                   for j in [-1, 0, 1]
                   for k in [-1, 0, 1]
                   for l in [-1, 0, 1])
    for factor in factors_set:
        yield factor


def memoize(f):
    results = {}

    def helper(n):
        if n not in results:
            results[n] = f(n)
        return results[n]

    return helper


@memoize
def linear_combination(n):
    """ returns the tuple (i,j,k,l) satisfying
        n = i*1 + j*3 + k*9 + l*27      """
    weighs = (1, 3, 9, 27)

    for factors in factors_set():
        sum = 0
        for i in range(len(factors)):
            sum += factors[i] * weighs[i]
        if sum == n:
            return factors

def weigh(pounds):
    weights = (1, 3, 9, 27)
    scalars = linear_combination(pounds)
    left = ""
    right = ""
    for i in range(len(scalars)):
        if scalars[i] == -1:
            left += str(weights[i]) + " "
	elif scalars[i] == 1:
            right += str(weights[i]) + " "
    return (left,right)

for i in [2, 3, 4, 7, 8, 9, 20, 40]:
	pans = weigh(i)
	print("Left  pan: " + str(i) + " plus " + pans[0])
	print("Right pan: " + pans[1] + "\n")
