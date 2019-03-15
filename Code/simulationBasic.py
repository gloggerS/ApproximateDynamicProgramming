# -*- coding: utf-8 -*-
"""
Created on Mon Mar  11 14:46:31 2019

@author: Stefan
"""

# from simulationBasic_data_flight_singleLeg import *
from bront_data import *

# %% PACKAGES

# # Plot iPython
# %matplotlib notebook
# import matplotlib.pyplot as plt

# Plot Python
import matplotlib as mpl

mpl.use("module://backend_interagg")
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

# Data
import numpy as np
import pandas as pd

# Calculation and Counting
import itertools
import math

# Distributions
from scipy.stats import bernoulli

# Timing
import time

# Memoization
import functools


# Genaueres inspizieren von Funktionen
# import inspect
# lines = inspect.getsource(value_expected)
# print(lines)


# %% FUNCTIONS
def memoize(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoizer


@memoize
def customer_choice_individual(offer_set_tuple, pw = preference_weights, pnp = preference_no_purchase):
    """
    For one customer of one customer segment, determine its purchase probabilities given one offer set.

    Tuple needed for memoization.

    :param offer_set_tuple: tuple with offered products indicated by 1=product offered
    :return: array of purchase probabilities starting with no purchase
    """

    if offer_set_tuple is None:
        ret = np.zeros_like(pw)
        return np.insert(ret, 0, 1)

    offer_set = np.asarray(offer_set_tuple)
    ret = pw * offer_set
    ret = np.array(ret / (pnp + sum(ret)))
    ret = np.insert(ret, 0, 1 - sum(ret))
    return ret


@memoize
def customer_choice_all(offer_set_tuple):
    probs = np.zeros(len(offer_set_tuple) + 1)
    for l in np.arange(len(preference_weights)):
        probs += arrival_probability[l]*customer_choice_individual(offer_set_tuple, preference_weights[l, :], preference_no_purchase[l])
        print(probs)
    return probs


def arrival(num_periods, arrival_probability):
    """
    Calculates the sample path.

    :param num_periods:
    :param arrival_probability:
    :return: Vector with arrival of customers.
    """
    return bernoulli.rvs(size=num_periods, p=arrival_probability)


def sample_path(num_periods, arrival_probability, capacity, revenues):
    """

    Over one complete booking horizon with *num_period* periods and a total *capacity*, the selling sample_path is
    recorded. A customer comes with *arrival_probability* and has given *preferenceWeights* and *noPurchasePreferences*.
    RETURN: data frame with columns (time, capacity (at start), customer arrived, product sold, revenue)
    *customerArrived*: ID of
    *customerPreferences*: for each customer segment stores the preferences to determine which product will be bought

    Helpers
    ----
    customerArrived :
        the ID of the customer segment that has arrived (used for customer preferences later on)

    :param num_periods:
    :param arrival_probability:
    :param capacity:
    :param revenues:
    :return:
    """

    index = np.arange(num_periods + 1)  # first row is a dummy (for nice for loop)
    columns = ['capacityStart', 'customerArrived', 'productSold', 'revenue', 'capacityEnd']

    df_sample_path = pd.DataFrame(index=index, columns=columns)
    df_sample_path = df_sample_path.fillna(0)

    df_sample_path.loc[0, 'capacityStart'] = df_sample_path.loc[0, 'capacityEnd'] = capacity
    df_sample_path.loc[1:num_periods, 'customerArrived'] = arrival(num_periods, arrival_probability)

    revenues_with_no_purchase = np.insert(revenues, 0, 0)
    products_with_no_purchase = np.arange(revenues_with_no_purchase.size)

    for t in np.delete(index, 0):  # start in second row (without actually deleting row)
        if df_sample_path.loc[t, 'customerArrived'] == 1:
            if df_sample_path.loc[t - 1, 'capacityEnd'] == 0:
                break
            # A customer has arrived and we have capacity.

            df_sample_path.loc[t, 'capacityStart'] = df_sample_path.loc[t - 1, 'capacityEnd']

            offer_set_tuple = value_expected(df_sample_path.loc[t, 'capacityStart'], t)[1]
            customer_probabilities = customer_choice_individual(offer_set_tuple)

            df_sample_path.loc[t, 'productSold'] = np.random.choice(products_with_no_purchase, size=1,
                                                                    p=customer_probabilities)

            df_sample_path.loc[t, 'revenue'] = revenues_with_no_purchase[df_sample_path.loc[t, 'productSold']]

            if df_sample_path.loc[t, 'productSold'] != 0:
                df_sample_path.loc[t, 'capacityEnd'] = df_sample_path.loc[t, 'capacityStart'] - 1
            else:
                df_sample_path.loc[t, 'capacityEnd'] = df_sample_path.loc[t, 'capacityStart']
        else:
            # no customer arrived
            df_sample_path.loc[t, 'capacityEnd'] = df_sample_path.loc[t, 'capacityStart'] = \
                df_sample_path.loc[t - 1, 'capacityEnd']

    return df_sample_path


@memoize
def value_expected(capacity, t):
    """
    Recursive implementation of the value function, i.e. dynamic program (DP)

    :param capacity:
    :param t: time to go (last chance for revenue is t=0)
    :return: value to be expected and optimal policy
    """
    offer_sets_to_test = list(map(list, itertools.product([0, 1], repeat=len(products))))
    offer_sets_max = 0
    offer_sets_max_val = 0

    if capacity == 0:
        return 0, None
    if capacity < 0:
        return -math.inf, None
    if t == T + 1:
        return 0, None

    for offer_set_index in range(len(offer_sets_to_test)):
        offer_set = offer_sets_to_test[offer_set_index]
        probs = customer_choice_individual(tuple(offer_set))

        val = value_expected(capacity, t + 1)[0]
        for j in products:
            p = float(probs[j])
            if p > 0.0:
                value_delta = value_expected(capacity, t + 1)[0] - \
                              value_expected(capacity - 1, t + 1)[0]
                val += arrival_probability * p * (revenues[j - 1] - value_delta)  # j-1 shifts to right product

        if val > offer_sets_max_val:
            offer_sets_max_val = val
            offer_sets_max = offer_set_index

    return offer_sets_max_val, tuple(offer_sets_to_test[offer_sets_max])
