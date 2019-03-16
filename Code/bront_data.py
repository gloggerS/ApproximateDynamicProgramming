import numpy as np


# %% OLD

numProducts = n = 3
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

#%% example for 4.2.2 Greedy Heuristic
numProducts = n = 3
preference_weights = np.array([[1, 1, 1],
                              [0, 1, 0],
                              [0, 0, 1]])
preference_no_purchase = np.array([1, 1, 1])
revenues = np.array([100, 19, 19])
arrival_probability = np.array([0.2, 0.3, 0.5])
pi = np.zeros(shape=(T+1, capacity +1))
K = 20
I = 100