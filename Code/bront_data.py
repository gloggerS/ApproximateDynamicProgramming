import numpy as np

example = "example0"
# example = "example for Greedy Heuristic"
# example = "example parallel flights"

if example == "example0":
    # toy example for explaining stuff, check implementation of CDLP
    numProducts = n = 8
    revenues = np.array([1200, 800, 500, 500, 800, 500, 300, 300])
    capacities = np.array([10, 5, 5])

    # capacity demand matrix A (rows: resources, cols: products)
    # a_ij = 1 if resource i is used by product j
    A = np.array([[0, 1, 1, 0, 0, 1, 1, 0],
                  [1, 0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1]])

    T = 30
    lam = 1

    arrival_probability = np.array([0.15, 0.15, 0.2, 0.25, 0.25])
    preference_weights = np.array([[5, 0, 0, 0, 8, 0, 0, 0],
                                  [10, 6, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 8, 5, 0, 0],
                                  [0, 0, 4, 0, 0, 0, 8, 0],
                                  [0, 0, 0, 6, 0, 0, 0, 8]])
    preference_no_purchase = np.array([2, 5, 2, 2, 2])

elif example == "example for Greedy Heuristic":
    # example for 4.2.2 Greedy Heuristic
    numProducts = n = 3
    preference_weights = np.array([[1, 1, 1],
                                   [0, 1, 0],
                                   [0, 0, 1]])
    preference_no_purchase = np.array([1, 1, 1])
    revenues = np.array([100, 19, 19])
    arrival_probability = np.array([0.2, 0.3, 0.5])
    pi = np.zeros(shape=(T + 1, capacity + 1))
    K = 20
    I = 100
elif example == "example parallel flights":
    # example for parallel flights
    numProducts = n = 6
    revenues = np.array([400, 800, 500, 1000, 300, 600])
    preference_weights = np.array([[0, 5, 0, 10, 0, 1],
                                   [5, 0, 1, 0, 10, 0],
                                   [10, 8, 6, 4, 3, 1],
                                   [8, 10, 4, 6, 1, 3]])
    preference_no_purchase = np.array([1, 5, 5, 1])
    arrival_probability = np.array([0.1, 0.15, 0.2, 0.05])
    capacities = np.array([30, 50, 40])
    T = 10  # 300

    offer_set_tuple = (1, 1, 1, 1, 1, 1)


# # %% OLD
#
# numProducts = n = 3
# products = np.arange(numProducts) + 1  # only real products (starting from 1)
# revenues = np.array([1000, 800, 600, 400])  # only real products
#
# T = 10
#
# customer_segments_num = 1
# arrival_probability = 0.8
# preference_weights = np.array([0.4, 0.8, 1.2, 1.6])
#
# varNoPurchasePreferences = np.array([1, 2, 3])
# varCapacity = np.arange(40, 120, 20)
#
# preference_no_purchase = 2
# capacity = 6
# offer_set = np.array([1, 0, 1, 1])
#


