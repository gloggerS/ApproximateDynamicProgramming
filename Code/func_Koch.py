# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:00:31 2019

@author: Stefan
"""

#  PACKAGES
# Data
import numpy as np
import pandas as pd

# Calculation and Counting
import itertools
import math

# Memoization
import functools

from dat_Koch import get_data_for_table1
from dat_Koch import get_variations

# %% HELPER-FUNCTIONS
def memoize(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoizer


def offer_sets(products):
    """
    Generates all possible offer sets.

    :param products:
    :return:
    """
    n = len(products)
    offer_sets_all = np.array(list(map(list, itertools.product([0, 1], repeat=n))))
    offer_sets_all = offer_sets_all[1:]  # always at least one product to offer
    return offer_sets_all


# %% FUNCTIONS
@memoize
def customer_choice_individual(offer_set_tuple, preference_weights, preference_no_purchase):
    """
    For one customer of one customer segment, determine its purchase probabilities given one offer set.

    Tuple needed for memoization.

    :param offer_set_tuple: tuple with offered products indicated by 1=product offered
    :param preference_weights: preference weights of one customer
    :param preference_no_purchase: no purchase preference of one customer
    :return: array of purchase probabilities ending with no purchase
    """

    if offer_set_tuple is None:
        ret = np.zeros_like(preference_weights)
        return np.append(ret, 1 - np.sum(ret))

    offer_set = np.asarray(offer_set_tuple)
    ret = preference_weights * offer_set
    ret = np.array(ret / (preference_no_purchase + np.sum(ret)))
    ret = np.append(ret, 1 - np.sum(ret))
    return ret


@memoize
def customer_choice_vector(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities):
    """
    From perspective of retailer: With which probability can he expect to sell each product (respectively non-purchase)

    :param offer_set_tuple: tuple with offered products indicated by 1=product offered
    :param preference_weights: preference weights of all customers
    :param preference_no_purchase: preference for no purchase for all customers
    :param arrival_probabilities: vector with arrival probabilities of all customer segments
    :return: array with probabilities of purchase ending with no purchase
    TODO probabilities don't have to sum up to one?
    """
    probs = np.zeros(len(offer_set_tuple) + 1)
    for l in np.arange(len(preference_weights)):
        probs += arrival_probabilities[l] * customer_choice_individual(offer_set_tuple, preference_weights[l, :],
                                                                       preference_no_purchase[l])
    return probs


def delta_value_j(j, capacities, t, A, preference_no_purchase):
    """
    For one product j, what is the difference in the value function if we sell one product.

    :param j:
    :param capacities:
    :param t:
    :param preference_no_purchase:
    :return:
    """
    return value_expected(capacities, t, preference_no_purchase)[0] - \
        value_expected(capacities - A[:, j], t, preference_no_purchase)[0]


@memoize
def value_expected(capacities, t, preference_no_purchase):
    """
    Recursive implementation of the value function, i.e. dynamic program (DP) as described on p. 241.

    :param capacities:
    :param t: time to go (last chance for revenue is t=0)
    :param preference_no_purchase:
    :return: value to be expected and optimal policy
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_for_table1()
    T = len(times)

    offer_sets_to_test = offer_sets(products)

    max_index = 0
    max_val = 0

    if all(capacities == 0):
        return 0, None
    if any(capacities < 0):
        return -math.inf, None
    if t == T:
        return 0, None

    for offer_set_index in range(len(offer_sets_to_test)):
        offer_set = offer_sets_to_test[offer_set_index, :]
        probs = customer_choice_vector(tuple(offer_set), preference_weights, preference_no_purchase,
                                       arrival_probabilities)

        val = float(value_expected(capacities, t + 1, preference_no_purchase)[0])  # ohne "float" würde ein numpy array
        #  zurückgegeben, und das später (max_val = val) direkt auf val verknüpft (call by reference)
        for j in products:  # nett, da Nichtkaufalternative danach kommt und nicht betrachtet wird
            p = float(probs[j])
            if p > 0.0:
                value_delta_j = delta_value_j(j, capacities, t + 1, A, preference_no_purchase)
                val += p * (revenues[j] - value_delta_j)

        if val > max_val:
            max_index = offer_set_index
            max_val = val
    return max_val, tuple(offer_sets_to_test[max_index, :])


# %%
var_capacities, var_no_purchase_preferences = get_variations()

num_rows = len(var_capacities)*len(var_no_purchase_preferences)
df = pd.DataFrame(index=np.arange(num_rows), columns=['c', 'u', 'DP'])
i = 0
for capacity in var_capacities:
    for preference_no_purchase in var_no_purchase_preferences:
        print(capacity)
        print(preference_no_purchase)

        df.loc[i] = [capacity, preference_no_purchase, value_expected(capacities=capacity, t=0,
                                                                      preference_no_purchase=preference_no_purchase)]
        i += 1

df.to_pickle("table1_DP.pkl")

#%%
df2 = pd.read_pickle("table1_DP.pkl")

# %%
customer_choice_individual(offer_set_tuple=tuple(np.array([0, 0, 0, 1])),
                           preference_weights=np.array([0.4, 0.8, 1.2, 1.6]),
                           preference_no_purchase=np.array([1]))
