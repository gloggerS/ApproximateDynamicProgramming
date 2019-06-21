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
    customer_segments, preference_weights, arrival_probabilities, times, T, time_start, \
    K, online_K, epsilon, exponential_smoothing \
    = setup_testing("DPSingleLeg-Evaluation")
capacities = var_capacities[0]
preferences_no_purchase = var_no_purchase_preferences[0]

#%%
# todo get the folder in which the parameters (theta, pi) to use are stored; e.g. via sys.argv (this script is called after the calculation of those parameters)
for i in sys.argv:
    print(i)

result_folder = 'C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\smallTest3-False-DP-190621-0838'


# %%
# Actual Code
def get_offer_set(c, t):
    # via index of optimal offer set
    return tuple(offer_sets.iloc[dat_lookup[t].iloc[c, 1]])


# needed results: for each time and capacity: value of coming revenues, optimal set to offer, list of other optimal sets
# capacities max: run it just for one set of capacities, the rest follows)
def return_raw_data_per_time(capacity):
    rows = list(range(capacity + 1))
    row_names = [str(i) for i in rows]
    df = pd.DataFrame(index=row_names,
                      columns=['value', 'offer_set_optimal', 'num_offer_set_optimal'])
    df.value = pd.to_numeric(df.value)
    return df


def simulate_sales(offer_set):
    customer = int(np.random.choice(np.arange(len(arrival_probabilities)+1),
                                    size=1,
                                    p=np.array([*arrival_probabilities, 1-sum(arrival_probabilities)])))
    if customer == len(arrival_probabilities):
        return len(products), customer  # no customer arrives => no product sold
    else:
        return int(np.random.choice(np.arange(len(products) + 1),
                                    size=1,
                                    p=customer_choice_individual(offer_set,
                                                                 preference_weights[customer],
                                                                 preferences_no_purchase[customer]))), customer


#%%
# online_K+1 policy iterations (starting with 0)
v_results = np.array([np.zeros(len(times))]*online_K)
c_results = np.array([np.zeros(shape=(len(times), len(capacities)))]*online_K)
customers_visited = np.array([np.zeros(len(times))]*online_K)  # to test whether we have the same customers arriving

with open(result_folder+"\\totalresults.data", "rb") as filehandle:
    dat_lookup = pickle.load(filehandle)

dat_lookup = dat_lookup[0]

offer_sets = pd.DataFrame(get_offer_sets_all(products))

for k in np.arange(online_K)+1:
    print(k, "of", online_K, "starting.")
    np.random.seed(K+k)
    random.seed(K+k)

    # line 3
    r_result = np.zeros(len(times))  # will become v_result
    c_result = np.zeros(shape=(len(times), len(capacities)), dtype=int)

    # line 5
    c = copy.deepcopy(capacities)  # (starting capacity at time 0)

    for t in times:
        # line 7  (starting capacity at time t)
        c_result[t] = c

        offer_set = get_offer_set(c, t)

        # line 13  (simulate sales)
        sold, customer = simulate_sales(offer_set)
        customers_visited[k-1, t] = customer

        # line 14
        try:
            r_result[t] = revenues[sold]
            c -= A[:, sold]
        except IndexError:
            # no product was sold
            pass

        # line 16-18
    v_results[k - 1] = np.cumsum(r_result[::-1])[::-1]
    c_results[k - 1] = c_result

# %%
# write result of calculations
with open(newpath+"\\vAll.data", "wb") as filehandle:
    pickle.dump(v_results, filehandle)

with open(newpath+"\\cAll.data", "wb") as filehandle:
    pickle.dump(c_results, filehandle)

with open(newpath+"\\vResults.data", "wb") as filehandle:
    pickle.dump(v_results[:, 0], filehandle)

# %%
wrapup(logfile, time_start, newpath)
