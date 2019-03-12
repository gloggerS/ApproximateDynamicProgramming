import functools
import numpy as np

def memoize(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        print(id(func))
        key = str(args) + str(kwargs)
        if key not in cache:
            print("cache missed: ", key)
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized_func


def fibonacci(n, a):
    if n == 0: return 0
    if n == 1: return 1
    else: return fibonacci(n-1, a) + fibonacci(n-2, a)

print(id(fibonacci))
fibonacci = memoize(fibonacci)
print(id(fibonacci))

a = np.array([2,3])
fibonacci(2, a)