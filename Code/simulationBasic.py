# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:36:31 2019

@author: Stefan
"""
#%% PACKAGES
import numpy as np
import pandas as pd
from scipy.stats import bernoulli

#%% OVERALL PARAMETERS
numProducts = 4
products = np.arange(numProducts)
revenues = np.array([1000, 800, 600, 400])

numPeriods = 40

numCustomerSegments = 1
arrivalProbability = 0.5
preferenceWeights = np.array([0.4, 0.8, 1.2, 1.6])

varNoPurchasePreferences = np.array([1,2,3])
varCapacity = np.arange(40, 120, 20)

#%% LOCAL PARAMETERS

#%% JUST TEMPORARY
noPurchasePreference = varNoPurchasePreferences[1]
offerSet = np.array([1,0,1,1])

#%% ACTUAL CODE



# =============================================================================
# For one customer of one customer segment, determine its purchase probabilities given one offer set.
# Output: vector of purchase probabilities starting with no purchase
# =============================================================================
def customerChoice(preferenceWeights, noPurchasePreference, offerSet):
    ret = preferenceWeights*offerSet
    ret = np.array(ret/(noPurchasePreference + sum(ret)))
    ret = np.insert(ret, 0, 1-sum(ret))
    return ret

def samplePath(numPeriods, arrivalProbability):
    return bernoulli.rvs(size=numPeriods, p=arrivalProbability)

def history(numPeriods, arrivalProbability, preferenceWeights, noPurchasePreference, offerSet):
    