"""
This script will calculate the Approximate Policy Iteration with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
"""

import pandas as pd
import numpy as np

import datetime
import time

import os
from shutil import copyfile, move

import itertools

from A_data_read import *
from B_helper import *
from ast import literal_eval

import pickle
import copy

import random

#%%
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
    customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
    epsilon, exponential_smoothing,\
    K, online_K, I \
    = setup_testing("APILinearSingleLeg-Evaluation")
capacities = var_capacities[0]
preferences_no_purchase = var_no_purchase_preferences[0]

#%%
np.random.seed(23)
test_customer = [np.random.random(len(times)) for _ in range(online_K)]
test_sales = [np.random.random(len(times)) for _ in range(online_K)]


#%%
# todo get the folder in which the parameters (theta, pi) to use are stored; e.g. via sys.argv (this script is called after the calculation of those parameters)
for i in sys.argv:
    print(i)

result_folder = os.getcwd() + '\\Results\\singleLegFlight-True-APILinearSingleLeg-190719-1125'

# %%
# Actual Code

# theta and pi as calculated
with open(result_folder+"\\thetaToUse.data", "rb") as filehandle:
    thetas_to_use = pickle.load(filehandle)
with open(result_folder+"\\piToUse.data", "rb") as filehandle:
    pis_to_use = pickle.load(filehandle)

#%%
# online_K+1 policy iterations (starting with 0)
v_results = np.array([np.zeros(len(times))]*online_K)
c_results = np.array([np.zeros(shape=(len(times), len(capacities)))]*online_K)

# to use in single timestep (will be overwritten)
pi = np.zeros(len(resources))

# setup result storage empty
value_final = pd.DataFrame(v_results[:, 0])
#%%
for capacities in var_capacities:
    for preferences_no_purchase in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities.tolist()), " - and - ", preferences_no_purchase, "of", str(var_no_purchase_preferences.tolist()), "starting.")

        thetas = thetas_to_use[str(capacities)][str(preferences_no_purchase)]
        pis = pis_to_use[str(capacities)][str(preferences_no_purchase)]

        for k in np.arange(online_K)+1:
            print(k, "of", online_K, "starting.")
            customer_random_stream = test_customer[k-1]
            sales_random_stream = test_sales[k-1]
            
            # line 3
            r_result = np.zeros(len(times))  # will become v_result
            c_result = np.zeros(shape=(len(times), len(capacities)), dtype=int)

            # line 5
            c = copy.deepcopy(capacities)  # (starting capacity at time 0)

            for t in times:
                # line 7  (starting capacity at time t)
                c_result[t] = c

                # line 12  (epsilon greedy strategy)
                pi[c == 0] = np.inf
                pi[c > 0] = pis[t][c > 0]
                # eps set to 0
                offer_set = determine_offer_tuple(pi, 0,
                                                  revenues, A, arrival_probabilities,
                                                  preference_weights, preferences_no_purchase)

                # line 13  (simulate sales)
                sold = simulate_sales(offer_set, customer_random_stream[t], sales_random_stream[t],
                                      arrival_probabilities, preference_weights, preferences_no_purchase)

                # line 14
                try:
                    r_result[t] = revenues[sold]
                    c -= A[:, sold]
                except IndexError:
                    # no product was sold
                    pass

            # line 16-18
            v_results[k-1] = np.cumsum(r_result[::-1])[::-1]
            c_results[k-1] = c_result

        value_final['' + str(capacities[0]) + '-' + str(preferences_no_purchase[0])] = pd.DataFrame(v_results[:, 0])


# %%
# write result of calculations
with open(newpath+"\\vResultsTable1.data", "wb") as filehandle:
    pickle.dump(value_final, filehandle)


# %%
wrapup(logfile, time_start, newpath)
