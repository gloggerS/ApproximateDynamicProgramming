# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:36:31 2019

@author: Stefan
"""
# %% PACKAGES

# Data
import numpy as np
import pandas as pd

# Distributions
from scipy.stats import bernoulli

# Plot
import matplotlib as mpl

mpl.use('module://backend_interagg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

# %% OVERALL PARAMETERS
numProducts = 4
products = np.arange(numProducts + 1)  # product number 0 is no purchase
revenues = np.array([0, 1000, 800, 600, 400])  # no purchase gives 0 revenue

numPeriods = 5

numCustomerSegments = 1
arrivalProbability = 1
preferenceWeights = np.array([0.4, 0.8, 1.2, 1.6])

varNoPurchasePreferences = np.array([1, 2, 3])
varCapacity = np.arange(40, 120, 20)

# %% LOCAL PARAMETERS


# %% JUST TEMPORARY
noPurchasePreference = 2
capacity = 4
offerSet = np.array([1, 0, 1, 1])


# %% ACTUAL CODE


def customer_choice(preference_weights, no_purchase_preference, offer_set):
    """
    For one customer of one customer segment, determine its purchase probabilities given one offer set.

    :param preference_weights: vector indicating the preference for each product
    :param no_purchase_preference: preference for no purchase
    :param offer_set: vector with offered products indicated by 1=product offered
    :return: vector of purchase probabilities starting with no purchase
    """
    ret = preference_weights * offer_set
    ret = np.array(ret / (no_purchase_preference + sum(ret)))
    ret = np.insert(ret, 0, 1 - sum(ret))
    return ret


def sample_path(num_periods, arrival_probability):
    """
    Calculates the sample path.

    :param num_periods:
    :param arrival_probability:
    :return: Vector with arrival of customers.
    """
    return bernoulli.rvs(size=num_periods, p=arrival_probability)


def history(numPeriods, arrivalProbability, preferenceWeights, noPurchasePreference, capacity, offerSet, products,
            revenues):
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

    index_customer = np.array([1])  # num customer segments

    df_customer = pd.DataFrame(index=index_customer, columns=products)
    df_customer = df_customer.fillna(0)

    df_customer.loc[1, :] = customer_choice(preferenceWeights, noPurchasePreference, offerSet)

    for i in np.delete(index, 0):  # start in second row (without actually deleting row)
        if df_history.loc[i, 'customerArrived'] == 1:
            if df_history.loc[i + 1, 'capacityEnd'] == 0:
                break
            # A customer has arrived and we have capacity.

            df_history.loc[i, 'capacityStart'] = df_history.loc[i + 1, 'capacityEnd']

            df_history.loc[i, 'productSold'] = np.random.choice(products, size=1, p=df_customer.loc[1, :])

            df_history.loc[i, 'revenue'] = revenues[df_history.loc[i, 'productSold']]

            if df_history.loc[i, 'productSold'] != 0:
                df_history.loc[i, 'capacityEnd'] = df_history.loc[i, 'capacityStart'] - 1
            else:
                df_history.loc[i, 'capacityEnd'] = df_history.loc[i, 'capacityStart']
        else:
            # no customer arrived
            df_history.loc[i, 'capacityEnd'] = df_history.loc[i, 'capacityStart'] = df_history.loc[i + 1, 'capacityEnd']

    return df_history



# %%
dfResult = history(numPeriods, arrivalProbability, preferenceWeights, noPurchasePreference, capacity, offerSet, products, revenues)

x = -dfResult.index
y = np.cumsum(dfResult['revenue'])
plt.plot(x, y)
plt.show()
