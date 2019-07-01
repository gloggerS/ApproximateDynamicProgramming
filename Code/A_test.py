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

#%%
test_by_name = []
test_by_name["single_leg_flight"] =
a = [np.random.random(400) for _ in range(5000)]

#%% check data
def check_example(name):
    print("########################")
    print("Checking ", name)
    print("------------------------ \n Check of dimensions: \n  ------------------------")
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
    try:
        print("Capacity thresholds: \t", len(data_by_name[name]["capacities"]) ==
              len(data_by_name[name]["capacities_thresholds"]))
    except:
        pass
    print("------------------------ \n Check of validity: \n ------------------------")
    print("sum of arrival probabilities <= 1: ", sum(data_by_name[name]["arrival_probabilities"]) <= 1)

    print("\n\n")


for i in data_by_name.keys():
    check_example(i)


#%%
f = open("0-test_by_name", "wb")
pickle.dump(data_by_name, f)
f.close()

