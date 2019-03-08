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
revenues = np.array([1000, 800, 600, 400])

numPeriods = 5

numCustomerSegments = 1
arrivalProbability = 0.5
preferenceWeights = np.array([0.4, 0.8, 1.2, 1.6])

varNoPurchasePreferences = np.array([1, 2, 3])
varCapacity = np.arange(40, 120, 20)

# %% LOCAL PARAMETERS


# %% JUST TEMPORARY
noPurchasePreference = varNoPurchasePreferences[1]
capacity = 2
offerSet = np.array([1, 0, 1, 1])


# %% ACTUAL CODE


# =============================================================================
# For one customer of one customer segment, determine its purchase probabilities given one offer set.
# Output: vector of purchase probabilities starting with no purchase
# =============================================================================
def customer_choice(preferenceWeights, noPurchasePreference, offerSet):
    ret = preferenceWeights * offerSet
    ret = np.array(ret / (noPurchasePreference + sum(ret)))
    ret = np.insert(ret, 0, 1 - sum(ret))
    return ret


def sample_path(numPeriods, arrivalProbability):
    return bernoulli.rvs(size=numPeriods, p=arrivalProbability)


def history(numPeriods, arrivalProbability, preferenceWeights, noPurchasePreference, capacity, offerSet, products,
            revenues):
    """
    Over one complete booking horizon with *numPeriod* periods and a total *capacity*, the selling history is recorded. A customer comes with *arrivalProbability* and has given *preferenceWeights* and *noPurchasePreferences*.
    TODO: calculate *offerSet* over time.
    RETURN: data frame with columns (time, capacity (at start), customer arrived, product sold, revenue)
    *customerArrived*: ID of 
    *randomNumber*: to specify which product will be sold
    *customerPreferences*: for each customer segment stores the preferences to determine which product will be bought
    
    Parameters
    ----------
    customerArrived :
        the ID of the customer segment that has arrived (used for customer preferences later on)
    """

    index = np.arange(numPeriods + 1)[::-1]  # first row dummy (for for loop)
    columns = ['capacityStart', 'customerArrived', 'productSold', 'revenue', 'capacityEnd', 'randomNumber']

    dfHistory = pd.DataFrame(index=index, columns=columns)
    dfHistory = dfHistory.fillna(0)

    dfHistory.loc[numPeriods, 'capacityStart'] = dfHistory.loc[numPeriods, 'capacityEnd'] = capacity
    dfHistory.loc[(numPeriods - 1):0, 'customerArrived'] = sample_path(numPeriods, arrivalProbability)

    dfHistory.loc[(numPeriods - 1):0, 'randomNumber'] = np.random.uniform(size=numPeriods)

    indexCustomer = np.array([1])  # num customer segments
    columnsCustomer = products

    dfCustomer = pd.DataFrame(index=indexCustomer, columns=columnsCustomer)
    dfCustomer = dfCustomer.fillna(0)

    dfCustomer.loc[1, :] = np.cumsum(customer_choice(preferenceWeights, noPurchasePreference, offerSet))

    for i in np.delete(index, 0):  # start in second row
        if dfHistory.loc[i, 'customerArrived'] == 1:
            if dfHistory.loc[i + 1, 'capacityEnd'] == 0:
                break

            dfHistory.loc[i, 'capacityStart'] = dfHistory.loc[i + 1, 'capacityEnd']

            prodPurchasable = (
                        dfCustomer.loc[dfHistory.loc[i, 'customerArrived'], :] < dfHistory.loc[i, 'randomNumber'])
            dfHistory.loc[i, 'productSold'] = max(prodPurchasable * products)

            dfHistory.loc[i, 'revenue'] = revenues[dfHistory.loc[i, 'productSold']]

            dfHistory.loc[i, 'capacityEnd'] = dfHistory.loc[i, 'capacityStart'] - 1
        else:
            # no customer arrived
            dfHistory.loc[i, 'capacityEnd'] = dfHistory.loc[i, 'capacityStart'] = dfHistory.loc[i + 1, 'capacityEnd']

    return dfHistory


dfResult = history(numPeriods, arrivalProbability, preferenceWeights, noPurchasePreference, capacity, offerSet,
                   products, revenues)

# %%

x = -dfResult.index
y = np.cumsum(dfResult['revenue'])
plt.plot(x, y)
plt.show()
