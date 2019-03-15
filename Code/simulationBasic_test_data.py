import numpy as np


# %% OVERALL PARAMETERS
numProducts = 4
products = np.arange(numProducts) + 1  # only real products (starting from 1)
revenues = np.array([1000, 800, 600, 400])  # only real products

T = 10

customer_segments_num = 1
arrival_probability = 0.8
preference_weights = np.array([0.4, 0.8, 1.2, 1.6])

varNoPurchasePreferences = np.array([1, 2, 3])
varCapacity = np.arange(40, 120, 20)

preference_no_purchase = 2
capacity = 6
offer_set = np.array([1, 0, 1, 1])

#%% for working
offer_tuple = (1, 0, 1, 1)
pi = np.zeros(shape=(T+1, capacity +1))