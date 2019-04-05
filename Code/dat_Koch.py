import numpy as np
import pandas as pd

# example = "singleLegFlight"
# example = "threeParallelFlights"
# example = "example0"
# example = "example for Greedy Heuristic"
# example = "example parallel flights"
example = "efficient sets"

if example == "singleLegFlight":
    n = 4
    products = np.arange(n)
    revenues = np.array([1000, 800, 600, 400])

    T = 400
    times = np.arange(T)

    L = 1
    customer_segments = np.arange(L)
    arrival_probabilities = np.array([0.5])
    preference_weights = np.array([[0.4, 0.8, 1.2, 1.6]])

    var_no_purchase_preferences = np.array([[1], [2], [3]])
    preference_no_purchase = np.array([var_no_purchase_preferences[0]])

    m = 1
    resources = np.arange(m)

    var_capacities = np.array([[40], [60], [80], [100], [120]])
    capacities = var_capacities[0]

    # capacity demand matrix A (rows: resources, cols: products)
    # a_ij = 1 if resource i is used by product j
    A = np.array([[1, 1, 1, 1]])

elif example == "threeParallelFlights":
    n = 6
    products = np.arange(n)
    revenues = np.array([400, 800, 500, 1000, 300, 600])

    T = 300
    times = np.arange(T)

    L = 4
    customer_segments = np.arange(L)
    arrival_probabilities = np.array([0.1, 0.15, 0.2, 0.05])
    preference_weights = np.array([[0, 5, 0, 10, 0, 1],
                                   [5, 0, 1, 0, 10, 0],
                                   [10, 8, 6, 4, 3, 1],
                                   [8, 10, 4, 6, 1, 3]])

    var_no_purchase_preferences = np.array([[1, 5, 5, 1],
                                            [1, 10, 5, 1],
                                            [5, 20, 10, 5]])
    preference_no_purchase = var_no_purchase_preferences[0]

    m = 3
    resources = np.arange(m)

    base_capacity = np.array([30, 50, 40])
    delta = np.arange(0.4, 1.21, 0.2)
    var_capacities = np.zeros((len(delta), len(base_capacity)))
    for i in np.arange(len(delta)):
        var_capacities[i] = delta[i]*base_capacity
    capacities = var_capacities[0]

    # capacity demand matrix A (rows: resources, cols: products)
    # a_ij = 1 if resource i is used by product j
    A = np.array([[1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 1, 1]])

elif example == "example0":
    # toy example for explaining stuff, check implementation of CDLP
    n = 8
    products = np.arange(n)
    revenues = np.array([1200, 800, 500, 500, 800, 500, 300, 300], dtype=np.float)

    m = 3
    resources = np.arange(m)
    capacities = np.array([10, 5, 5])

    # capacity demand matrix A (rows: resources, cols: products)
    # a_ij = 1 if resource i is used by product j
    A = np.array([[0, 1, 1, 0, 0, 1, 1, 0],
                  [1, 0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1]])

    T = 30
    times = np.arange(T)

    L = 5
    customer_segments = np.arange(L)
    arrival_probabilities = np.array([0.15, 0.15, 0.2, 0.25, 0.25])
    preference_weights = np.array([[5, 0, 0, 0, 8, 0, 0, 0],
                                  [10, 6, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 8, 5, 0, 0],
                                  [0, 0, 4, 0, 0, 0, 8, 0],
                                  [0, 0, 0, 6, 0, 0, 0, 8]])
    preference_no_purchase = np.array([2, 5, 2, 2, 2])

elif example == "example for Greedy Heuristic":
    # example for 4.2.2 Greedy Heuristic
    n = 3
    products = np.arange(n)

    A = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    L = 3
    customer_segments = np.arange(L)
    preference_weights = np.array([[1, 1, 1],
                                   [0, 1, 0],
                                   [0, 0, 1]])
    preference_no_purchase = np.array([1, 1, 1])
    w = revenues = np.array([100, 19, 19])
    arrival_probabilities = np.array([1, 1, 1])

    pi = 0

if example == "efficient sets":
    purchase_rate_vectors = np.array([[0, 0, 0, 1],
                                      [0.3, 0, 0, 0.7],
                                      [0, 0.4, 0, 0.6],
                                      [0, 0, 0.5, 0.5],
                                      [0.1, 0.6, 0, 0.3],
                                      [0.3, 0, 0.5, 0.2],
                                      [0, 0.4, 0.5, 0.1],
                                      [0.1, 0.4, 0.5, 0]])
    sets_quantities = np.array([0, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1])
    sets_revenues = np.array([0, 240, 200, 225, 380, 465, 425, 505])
    offer_sets = np.array(['0', 'Y', 'M', 'K', 'Y,M', 'Y,K', 'M,K', 'Y,M,K'])

    data_sets = pd.DataFrame(data=np.array([sets_quantities, sets_revenues]).T, columns=['q', 'r'], index=offer_sets)
# %% Check up
print("Check of dimensions: \n  ------------------------")
print("Ressourcen: \t", len(resources) == len(capacities) == A.shape[0])
print("Produkte: \t\t", len(products) == len(revenues) == preference_weights.shape[1] == A.shape[1])
print("Kundensgemente:\t", len(customer_segments) == len(arrival_probabilities) == preference_weights.shape[0] ==
      len(preference_no_purchase))


# %% Export functionality
def get_data():
    return resources, capacities, \
           products, revenues, A, \
           customer_segments, preference_weights, preference_no_purchase, arrival_probabilities, \
           times


def get_data_without_variations():
    return resources, \
           products, revenues, A, \
           customer_segments, preference_weights, arrival_probabilities, \
           times


def get_capacities_and_preferences_no_purchase():
    return capacities, preference_no_purchase


def get_variations():
    return var_capacities, var_no_purchase_preferences


def get_preference_no_purchase():
    return preference_no_purchase

