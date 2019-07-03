"""
This script will calculate the Value Expected exactly and store them in a large dataframe
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

from joblib import Parallel, delayed, dump, load
import multiprocessing

import pickle
import random

#%%
# Get settings, prepare data, create storage for results
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K \
    = setup_testing("DPSingleLeg-Evaluation")
capacities = var_capacities[0]
preferences_no_purchase = var_no_purchase_preferences[0]

with open("0-test_customer.data", "rb") as filehandle:
    test_customer = pickle.load(filehandle)
with open("0-test_sales.data", "rb") as filehandle:
    test_sales = pickle.load(filehandle)

if online_K > len(test_sales) or online_K > len(test_customer):
    raise ValueError("online_K as specified in 0_settings.csv has to be smaller then the test data given in test_sales and test_customer")

#%%
# todo get the folder in which the parameters (theta, pi) to use are stored; e.g. via sys.argv (this script is called after the calculation of those parameters)
for i in sys.argv:
    print(i)

result_folder = os.getcwd() + '\\Results\\singleLegFlight-True-DPSingleLeg-190701-1801'


# %%
# Actual Code
def get_offer_set(c, t):
    # via index of optimal offer set
    return tuple(offer_sets.iloc[dat_lookup[t].iloc[c, 1]])


def simulate_sales(offer_set, random_customer, random_sales):
    customer = random_customer <= np.array([*np.cumsum(arrival_probabilities), 1.0])
    customer = min(np.array(range(0, len(customer)))[customer])

    if customer == len(arrival_probabilities):
        return len(products), customer  # no customer arrives => no product sold (product out of range)
    else:
        product = random_sales <= np.cumsum(customer_choice_individual(offer_set,
                                                             preference_weights[customer],
                                                             preferences_no_purchase[customer]))
        product = min(np.arange(len(products) + 1)[product])
        return product, customer


#%%
# online_K+1 policy iterations (starting with 0)
v_results = np.array([np.zeros(len(times))]*online_K)
c_results = np.array([np.zeros(shape=(len(times), len(capacities)))]*online_K)

with open(result_folder+"\\totalresults.data", "rb") as filehandle:
    dat_lookup = pickle.load(filehandle)

dat_lookup = dat_lookup[0]

offer_sets = pd.DataFrame(get_offer_sets_all(products))

value_final = pd.DataFrame(v_results[:,0])  # setup result storage empty
value_final['' + str(capacities[0]) + '-' + str(preferences_no_purchase[0])] = pd.DataFrame(v_results[:, 0])
#%%
for capacities in var_capacities:
    for preferences_no_purchase in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities), " - and - ", preferences_no_purchase, "of", str(var_no_purchase_preferences), "starting.")
        for k in np.arange(online_K)+1:
            print(k, "of", online_K, "starting.")
            customer_random_stream = test_customer[k-1]
            sales_random_stream = test_sales[k-1]

            # line 3
            r_result = np.zeros(len(times))  # will become v_result
            c_result = np.zeros(shape=(len(times), len(capacities)), dtype=int)

            # line 5
            c = copy.deepcopy(capacities)  # (starting capacity at time 0)
            c = c[0]

            for t in times:
                if c < 0:
                    raise ValueError
                # line 7  (starting capacity at time t)
                c_result[t] = c

                offer_set = get_offer_set(c, t)

                # line 13  (simulate sales)
                sold, customer = simulate_sales(offer_set, customer_random_stream[t], sales_random_stream[t])

                # line 14
                try:
                    r_result[t] = revenues[sold]
                    c -= A[0, sold]  # IMPORTANT: have integer values here, no arrays
                except IndexError:
                    # no product was sold
                    pass

                # line 16-18
            v_results[k - 1] = np.cumsum(r_result[::-1])[::-1]
            c_results[k - 1] = c_result

        value_final['' + str(capacities[0]) + '-' + str(preferences_no_purchase[0])] = pd.DataFrame(v_results[:, 0])

# %%
# write result of calculations
with open(newpath+"\\vResultsTable1.data", "wb") as filehandle:
    pickle.dump(value_final, filehandle)

# %%
wrapup(logfile, time_start, newpath)
