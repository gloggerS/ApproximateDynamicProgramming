"""
This script is for entering data. It is checked and included in the "database".
"""

import numpy as np
import pandas as pd
import pickle

# # example = "singleLegFlight"
# example = "threeParallelFlights"
# # example = "example0"
# # example = "example for Greedy Heuristic"
# # example = "example parallel flights"
# # example = "efficient sets"

#%% quick on-/off
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

pi = np.array([0, 0, 0])

#%% old code
# if example == "efficient sets":
#     purchase_rate_vectors = np.array([[0, 0, 0, 1],
#                                       [0.3, 0, 0, 0.7],
#                                       [0, 0.4, 0, 0.6],
#                                       [0, 0, 0.5, 0.5],
#                                       [0.1, 0.6, 0, 0.3],
#                                       [0.3, 0, 0.5, 0.2],
#                                       [0, 0.4, 0.5, 0.1],
#                                       [0.1, 0.4, 0.5, 0]])
#     sets_quantities = np.array([0, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1])
#     sets_revenues = np.array([0, 240, 200, 225, 380, 465, 425, 505])
#     offer_sets = np.array(['0', 'Y', 'M', 'K', 'Y,M', 'Y,K', 'M,K', 'Y,M,K'])
#
#     data_sets = pd.DataFrame(data=np.array([sets_quantities, sets_revenues]).T, columns=['q', 'r'], index=offer_sets)

#%%

data_by_name = {}

#%% singleLegFlight
data_by_name["singleLegFlight"] = {}

data_by_name["singleLegFlight"]["products"] = np.arange(4)  # n
data_by_name["singleLegFlight"]["revenues"] = np.array([1000, 800, 600, 400])

data_by_name["singleLegFlight"]["times"] = np.arange(400)  # T

data_by_name["singleLegFlight"]["customer_segments"] = np.arange(1)  # L
data_by_name["singleLegFlight"]["arrival_probabilities"] = np.array([0.5])
data_by_name["singleLegFlight"]["preference_weights"] = np.array([[0.4, 0.8, 1.2, 1.6]])

data_by_name["singleLegFlight"]["var_no_purchase_preferences"] = np.array([[1], [2], [3]])
data_by_name["singleLegFlight"]["preference_no_purchase"] = \
    np.array(data_by_name["singleLegFlight"]["var_no_purchase_preferences"][0])

data_by_name["singleLegFlight"]["resources"] = np.arange(1)  # m

data_by_name["singleLegFlight"]["var_capacities"] = np.array([[40], [60], [80], [100], [120]])
data_by_name["singleLegFlight"]["capacities"] = data_by_name["singleLegFlight"]["var_capacities"][0]

# capacity demand matrix A (rows: resources, cols: products)
# a_ij = 1 if resource i is used by product j
data_by_name["singleLegFlight"]["A"] = np.array([[1, 1, 1, 1]])


#%% smallTest
data_by_name["smallTest"] = {}

data_by_name["smallTest"]["products"] = np.arange(4)  # n
data_by_name["smallTest"]["revenues"] = np.array([1000, 800, 600, 400])

data_by_name["smallTest"]["times"] = np.arange(10)  # T

data_by_name["smallTest"]["customer_segments"] = np.arange(1)  # L
data_by_name["smallTest"]["arrival_probabilities"] = np.array([0.5])
data_by_name["smallTest"]["preference_weights"] = np.array([[0.4, 0.8, 1.2, 1.6]])

data_by_name["smallTest"]["var_no_purchase_preferences"] = np.array([[1]])
data_by_name["smallTest"]["preference_no_purchase"] = \
    np.array(data_by_name["smallTest"]["var_no_purchase_preferences"][0])

data_by_name["smallTest"]["resources"] = np.arange(1)  # m

data_by_name["smallTest"]["var_capacities"] = np.array([[4]])
data_by_name["smallTest"]["capacities"] = data_by_name["smallTest"]["var_capacities"][0]

# capacity demand matrix A (rows: resources, cols: products)
# a_ij = 1 if resource i is used by product j
data_by_name["smallTest"]["A"] = np.array([[1, 1, 1, 1]])


#%% three parallel flights
data_by_name["threeParallelFlights"] = {}

data_by_name["threeParallelFlights"]["products"] = np.arange(6)  # n
data_by_name["threeParallelFlights"]["revenues"] = np.array([400, 800, 500, 1000, 300, 600])

data_by_name["threeParallelFlights"]["times"] = np.arange(3)  # T = 300

data_by_name["threeParallelFlights"]["customer_segments"] = np.arange(4)  # L = 4
data_by_name["threeParallelFlights"]["arrival_probabilities"] = np.array([0.1, 0.15, 0.2, 0.05])
data_by_name["threeParallelFlights"]["preference_weights"] = np.array([[0, 5, 0, 10, 0, 1],
                               [5, 0, 1, 0, 10, 0],
                               [10, 8, 6, 4, 3, 1],
                               [8, 10, 4, 6, 1, 3]])

data_by_name["threeParallelFlights"]["var_no_purchase_preferences"] = np.array([[1, 5, 5, 1],
                                        [1, 10, 5, 1],
                                        [5, 20, 10, 5]])
data_by_name["threeParallelFlights"]["preference_no_purchase"] = \
    data_by_name["threeParallelFlights"]["var_no_purchase_preferences"][0]

data_by_name["threeParallelFlights"]["resources"] = np.arange(3)  # m = 3

base_capacity = np.array([30, 50, 40])
delta = np.arange(0.4, 1.21, 0.2)
data_by_name["threeParallelFlights"]["var_capacities"] = np.zeros((len(delta), len(base_capacity)), dtype=int)
for i in np.arange(len(delta)):
    data_by_name["threeParallelFlights"]["var_capacities"][i] = delta[i] * base_capacity
data_by_name["threeParallelFlights"]["capacities"] = data_by_name["threeParallelFlights"]["var_capacities"][0]

# capacity demand matrix A (rows: resources, cols: products)
# a_ij = 1 if resource i is used by product j
data_by_name["threeParallelFlights"]["A"] = np.array([[1, 1, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1]])

# %%
# toy example for explaining stuff, check implementation of CDLP
data_by_name["example0"] = {}

data_by_name["example0"]["products"] = np.arange(8)  # n
data_by_name["example0"]["revenues"] = np.array([1200, 800, 500, 500, 800, 500, 300, 300], dtype=np.float)

data_by_name["example0"]["resources"] = np.arange(3)  # m
data_by_name["example0"]["capacities"] = np.array([10, 5, 5])

# capacity demand matrix A (rows: resources, cols: products)
# a_ij = 1 if resource i is used by product j
data_by_name["example0"]["A"] = np.array([[0, 1, 1, 0, 0, 1, 1, 0],
                  [1, 0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1]])

data_by_name["example0"]["times"] = np.arange(30)  # T

data_by_name["example0"]["customer_segments"] = np.arange(5)  # L
data_by_name["example0"]["arrival_probabilities"] = np.array([0.15, 0.15, 0.2, 0.25, 0.25])
data_by_name["example0"]["preference_weights"] = np.array([[5, 0, 0, 0, 8, 0, 0, 0],
                                  [10, 6, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 8, 5, 0, 0],
                                  [0, 0, 4, 0, 0, 0, 8, 0],
                                  [0, 0, 0, 6, 0, 0, 0, 8]])
data_by_name["example0"]["preference_no_purchase"] = np.array([2, 5, 2, 2, 2])


#%% check data
def check_example(name):
    print("Checking ", name)
    print("Check of dimensions: \n  ------------------------")
    print("Ressourcen: \t", len(data_by_name[name]["resources"]) ==
          len(data_by_name[name]["capacities"]) ==
          data_by_name[name]["A"].shape[0])
    print("Produkte: \t\t", len(data_by_name[name]["products"]) ==
          len(data_by_name[name]["revenues"]) ==
          data_by_name[name]["preference_weights"].shape[1] ==
          data_by_name[name]["A"].shape[1])
    print("Kundensgemente:\t", len(data_by_name[name]["customer_segments"]) ==
          len(data_by_name[name]["arrival_probabilities"]) ==
          data_by_name[name]["preference_weights"].shape[0] ==
          len(data_by_name[name]["preference_no_purchase"]))
    print("\n\n")


for i in data_by_name.keys():
    check_example(i)


#%%
f = open("0-data_by_name", "wb")
pickle.dump(data_by_name, f)
f.close()

