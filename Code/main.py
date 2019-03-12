import  functools

def memoize(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        print(key)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized_func

def fibonacci(n):
    if n == 0:return 0
    if n == 1:return 1
    else: return fibonacci(n-1) + fibonacci(n-2)

fibonacci = memoize(fibonacci)

fibonacci(10)
lines = inspect.getsource(fibonacci)
print(lines)