# -*- coding: utf-8 -*-
"""
Created on Mon Mar  11 14:46:31 2019

@author: Stefan
"""

# %% PACKAGES

# Data
import numpy as np
import pandas as pd

# Calculation and Counting
import itertools
import math

# Distributions
from scipy.stats import bernoulli

# Plot
import matplotlib as mpl

mpl.use('module://backend_interagg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

# Timing
import time

# Memoization
import functools

# %% OVERALL PARAMETERS
numProducts = 4
products = np.arange(numProducts) + 1  # only real products (starting from 1)
revenues = np.array([1000, 800, 600, 400])  # only real products

numPeriods = 10

customer_segments_num = 1
arrivalProbability = 0.8
preference_weights = np.array([0.4, 0.8, 1.2, 1.6])

varNoPurchasePreferences = np.array([1, 2, 3])
varCapacity = np.arange(40, 120, 20)
# %% LOCAL PARAMETERS


# %% JUST TEMPORARY
preference_no_purchase = 2
capacity = 6
offer_set = np.array([1, 0, 1, 1])


# %% FUNCTIONS
def customer_choice_individual(preference_weights, preference_no_purchase, offer_set):
    """
    For one customer of one customer segment, determine its purchase probabilities given one offer set.

    :param preference_weights: vector indicating the preference for each product
    :param preference_no_purchase: preference for no purchase
    :param offer_set: vector with offered products indicated by 1=product offered
    :return: vector of purchase probabilities starting with no purchase
    """
    ret = preference_weights * offer_set
    ret = np.array(ret / (preference_no_purchase + sum(ret)))
    ret = np.insert(ret, 0, 1 - sum(ret))
    return ret


def customer_choice_segments(customer_segments_num, preference_no_purchase, offer_set, preference_weights):
    """
    Generiert die Wahrscheinlichkeitsverteilung gemäß der Präferenzen der einzelnen Kundensegmente. x als Tupel, um Memoisierung zu nutzen

    :param x: customer_segments_num, preference_no_purchase, offer_set, preference_weights, products
    :return:
    """

    products_with_no_purchase = np.arange(len(offer_set)+1)
    df_customer = pd.DataFrame(index=[customer_segments_num], columns=products_with_no_purchase)
    df_customer = df_customer.fillna(0)
    df_customer.loc[1, :] = customer_choice_individual(preference_weights, preference_no_purchase, offer_set)
    return df_customer


def sample_path(num_periods, arrival_probability):
    """
    Calculates the sample path.

    :param num_periods:
    :param arrival_probability:
    :return: Vector with arrival of customers.
    """
    return bernoulli.rvs(size=num_periods, p=arrival_probability)


def history(numPeriods, arrivalProbability, preferenceWeights, noPurchasePreference, capacity, offerSet, products,
            revenues, customer_segments_num):
    """

    Over one complete booking horizon with *numPeriod* periods and a total *capacity*, the selling history is recorded. A customer comes with *arrivalProbability* and has given *preferenceWeights* and *noPurchasePreferences*.
    TODO: calculate *offerSet* over time.
    RETURN: data frame with columns (time, capacity (at start), customer arrived, product sold, revenue)
    *customerArrived*: ID of
    *customerPreferences*: for each customer segment stores the preferences to determine which product will be bought

    Helpers
    ----
    customerArrived :
        the ID of the customer segment that has arrived (used for customer preferences later on)

    :param numPeriods:
    :param arrivalProbability:
    :param preferenceWeights:
    :param noPurchasePreference:
    :param capacity:
    :param offerSet:
    :param products:
    :param revenues:
    :return:
    """

    index = np.arange(numPeriods + 1)[::-1]  # first row is a dummy (for nice for loop)
    columns = ['capacityStart', 'customerArrived', 'productSold', 'revenue', 'capacityEnd']

    df_history = pd.DataFrame(index=index, columns=columns)
    df_history = df_history.fillna(0)

    df_history.loc[numPeriods, 'capacityStart'] = df_history.loc[numPeriods, 'capacityEnd'] = capacity
    df_history.loc[(numPeriods - 1):0, 'customerArrived'] = sample_path(numPeriods, arrivalProbability)

    revenues_with_no_purchase = np.insert(revenues, 0, 0)

    df_customer = customer_choice_segments(customer_segments_num, noPurchasePreference, offerSet, preferenceWeights)

    for i in np.delete(index, 0):  # start in second row (without actually deleting row)
        if df_history.loc[i, 'customerArrived'] == 1:
            if df_history.loc[i + 1, 'capacityEnd'] == 0:
                break
            # A customer has arrived and we have capacity.

            df_history.loc[i, 'capacityStart'] = df_history.loc[i + 1, 'capacityEnd']

            products_with_no_purchase = np.insert(products, 0, 0)
            df_history.loc[i, 'productSold'] = np.random.choice(products_with_no_purchase, size=1,
                                                                p=df_customer.loc[1, :])

            df_history.loc[i, 'revenue'] = revenues_with_no_purchase[df_history.loc[i, 'productSold']]

            if df_history.loc[i, 'productSold'] != 0:
                df_history.loc[i, 'capacityEnd'] = df_history.loc[i, 'capacityStart'] - 1
            else:
                df_history.loc[i, 'capacityEnd'] = df_history.loc[i, 'capacityStart']
        else:
            # no customer arrived
            df_history.loc[i, 'capacityEnd'] = df_history.loc[i, 'capacityStart'] = df_history.loc[i + 1, 'capacityEnd']

    return df_history


def value_expected(capacity, time, products, revenues, preference_weights, preference_no_purchase):
    """
    Recursive implementation of the value function, i.e. dynamic program (DP)

    :param capacity:
    :param products:
    :param revenues:
    :param preferenceWeights:
    :param noPurchasePreference:
    :param time:
    :return: Dictionary with value function
    """
    offer_sets_to_test = list(map(list, itertools.product([0, 1], repeat=len(products))))
    offer_sets_max = 0
    offer_sets_max_val = 0

    if capacity == 0:
        return 0
    if capacity < 0:
        return -math.inf
    if time == 0:
        return 0

    for offer_set_index in range(len(offer_sets_to_test)):
        offer_set = offer_sets_to_test[offer_set_index]
        probs = customer_choice_segments([1], preference_no_purchase, offer_set, preference_weights)

        val = value_expected(capacity, time - 1, products, revenues, preference_weights, preference_no_purchase)
        for j in products:
            p = float(probs.loc[1, j])
            if p > 0.0:
                value_delta = value_expected(capacity, time - 1, products, revenues, preference_weights,
                                             preference_no_purchase) - \
                              value_expected(capacity - 1, time - 1, products, revenues, preference_weights,
                                             preference_no_purchase)
                val += p * (revenues[j - 1] - value_delta)  # j-1 shifts to right product

        if val > offer_sets_max_val:
            offer_sets_max_val = val
            offer_sets_max = offer_set_index

    return offer_sets_max_val


#%% Test - history
dfResult = history(numPeriods, arrivalProbability, preference_weights, preference_no_purchase, capacity,
                   offer_set, products, revenues, customer_segments_num)

x = -dfResult.index
y = np.cumsum(dfResult['revenue'])
plt.plot(x, y)

#%% Test - customer weight
probs = customer_choice_segments([1], preference_no_purchase, offer_set, preference_weights)

#%% Test - value expected
value_expected(1, 1, products, revenues, preference_weights, preference_no_purchase)