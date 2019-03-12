from simulationBasic import *



# %% working here
import functools

import inspect
lines = inspect.getsource(value_expected)
print(lines)

id(customer_choice_segments)
#%%
def memoize(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            print("cache missed: ", key)
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized_func


customer_choice_segments = memoize(customer_choice_segments)
value_expected = memoize(value_expected)


# %%
probs = customer_choice_segments(1, preference_no_purchase, offer_set, preference_weights)
probs = customer_choice_segments(1, preference_no_purchase, offer_set, preference_weights)

# %%
start_time = time.time()
value_expected(2, 2, products, revenues, preference_weights, preference_no_purchase)
print(time.time() - start_time)
