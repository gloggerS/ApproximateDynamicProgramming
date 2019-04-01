import numpy as np

example = "singleLegFlight"

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

    varNoPurchasePreferences = np.array([1, 2, 3])
    preference_no_purchase = np.array([varNoPurchasePreferences[0]])

    m = 1
    resources = np.arange(m)

    varCapacity = np.arange(40, 120, 20)
    capacities = np.array([varCapacity[0]])

    # capacity demand matrix A (rows: resources, cols: products)
    # a_ij = 1 if resource i is used by product j
    A = np.array([[1, 1, 1, 1]])

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

def get_data_for_table1():
    return resources, \
           products, revenues, A, \
           customer_segments, preference_weights, arrival_probabilities, \
           times
