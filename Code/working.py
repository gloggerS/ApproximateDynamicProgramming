from simulationBasic import *


#
# # %% working here
# import functools
#
# # import inspect
# # lines = inspect.getsource(value_expected)
# # print(lines)
#
# #%%
#
# def memoize(obj):
#     cache = obj.cache = {}
#
#     @functools.wraps(obj)
#     def memoizer(*args, **kwargs):
#         key = str(args) + str(kwargs)
#         if key not in cache:
#             cache[key] = obj(*args, **kwargs)
#         return cache[key]
#     return memoizer
#
# #%%
# def customer_choice_individual_tuple(offer_set_tuple):
#     """
#     For one customer of one customer segment, determine its purchase probabilities given one offer set.
#
#     :param preference_weights: vector indicating the preference for each product
#     :param preference_no_purchase: preference for no purchase
#     :param offer_set: vector with offered products indicated by 1=product offered
#     :return: vector of purchase probabilities starting with no purchase
#     """
#     offer_set = np.asarray(offer_set_tuple)
#     ret = preference_weights * offer_set
#     ret = np.array(ret / (preference_no_purchase + sum(ret)))
#     ret = np.insert(ret, 0, 1 - sum(ret))
#     return ret
#
# customer_choice_individual_tuple = memoize(customer_choice_individual_tuple)
# %%
offer_set_tuple = (1,1,0,1)
offer_set = np.asarray(offer_set_tuple)

print(customer_choice_individual(offer_set))
print(customer_choice_individual_tuple(offer_set_tuple))

# %%
start_time = time.time()
value_expected(3, 3)
print(time.time() - start_time)
