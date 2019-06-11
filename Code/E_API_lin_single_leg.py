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

from joblib import Parallel, delayed, dump, load
import multiprocessing

import pickle
import copy

#%%
# Get settings, prepare data, create storage for results
print("API linear single leg starting.\n\n")

# settings
settings = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
example = settings.iloc[0, 1]
use_variations = (settings.iloc[1, 1] == "True") | (settings.iloc[1, 1] == "true")  # if var. capacities should be used
storage_folder = example + "-" + str(use_variations) + "-API-lin-" + time.strftime("%y%m%d-%H%M")

# data
dat = get_all(example)
print("\n Data used. \n")
for key in dat.keys():
    print(key, ":\n", dat[key])
print("\n\n")
del dat

# prepare storage location
newpath = get_storage_path(storage_folder)
os.makedirs(newpath)

# copy settings to storage location
copyfile("0_settings.csv", newpath+"\\0_settings.csv")
logfile = open(newpath+"\\0_logging.txt", "w+")  # write and create (if not there)

# time
print("Time:", datetime.datetime.now())
print("Time (starting):", datetime.datetime.now(), file=logfile)
time_start = time.time()

# settings
for row in settings:
    print(settings.loc[row, 0], ":\t", settings.loc[row, 1])
    print(settings.loc[row, 0], ":\t", settings.loc[row, 1], file=logfile)

# variations (capacity and no purchase preference)
if use_variations:
    var_capacities, var_no_purchase_preferences = get_variations(example)
else:
    capacities, no_purchase_preferences = get_capacities_and_preferences_no_purchase(example)
    var_capacities = np.array([capacities])
    var_no_purchase_preferences = np.array([no_purchase_preferences])

# no varying capacity here
capacities = var_capacities[0]

# other data
resources, \
    products, revenues, A, \
    customer_segments, preference_weights, arrival_probabilities, \
    times = get_data_without_variations(example)
T = len(times)

print("\nEverything set up.")


# %%
# Actual Code
K = 3  # 60
I = 10  # 800

# all parameters in one row of a dataframe
# K+1 policy iterations (starting with 0)
# T time steps
# theta and bid price for each resource
param = np.array([[np.zeros(1+len(resources))]*T]*(K+1))
# pi for each time step
pi_act = np.zeros(len(capacities))

# line 1
param[0] = 0

for k in np.arange(K)+1:
    # line 3
    r_sample = np.zeros(len(times))  # will become v_sample
    c_sample = np.zeros(shape=(len(times), len(capacities)), dtype=int)

    # line 5
    c = capacities  # (starting capacity at time 0)

    for t in times:
        # line 7  (starting capacity at time t)
        c_sample[t] = c

        # line 9  (adjust bid price)
        pi_act[c_sample[t] == 0] = np.inf
        pi_act[c_sample[t] > 0] = param[k-1][t][np.append(False, c_sample[t] > 0)]

        # line 12
        x = determine_offer_tuple(pi_act)




#%%
def determine_offer_tuple(pi):
    """
    OLD Implementation
    Determines the offerset given the bid prices for each resource.

    Implement the Greedy Heuristic from Bront et al: A Column Generation Algorithm ... 4.2.2

    :param pi:
    :return:
    """

    # setup
    offer_tuple = np.zeros_like(revenues)

    # line 1
    S_prime = revenues - pi > 0  # vector - scalar = vector - np.ones_like(vector)*scalar

    # line 2-3
    # offer_sets_to_test has in each row an offer set, we want to test
    offer_sets_to_test = np.zeros((sum(S_prime), len(revenues)))
    offer_sets_to_test[np.arange(sum(S_prime)), np.where(S_prime)] = 1
    offer_sets_to_test += offer_tuple
    offer_sets_to_test = (offer_sets_to_test > 0)

    value_marginal = np.apply_along_axis(calc_value_marginal, axis=1, arr=offer_sets_to_test)

    offer_tuple[np.argmax(value_marginal)] = 1
    S_prime = S_prime & offer_tuple == 0
    v_S = np.amax(value_marginal)

    # line 4
    while True:
        # 4a
        # offer_sets_to_test has in each row an offer set, we want to test
        offer_sets_to_test = np.zeros((sum(S_prime), len(revenues)))
        offer_sets_to_test[np.arange(sum(S_prime)), np.where(S_prime)] = 1
        offer_sets_to_test += offer_tuple
        offer_sets_to_test = (offer_sets_to_test > 0)

        # 4b
        value_marginal = np.apply_along_axis(calc_value_marginal, axis=1, arr=offer_sets_to_test)

        if np.amax(value_marginal) > v_S:
            v_S = np.amax(value_marginal)
            offer_tuple[np.argmax(value_marginal)] = 1
            S_prime = S_prime & offer_tuple == 0
            if all(offer_tuple == 1):
                break
        else:
            break
    return tuple(offer_tuple)


def calc_value_marginal(indices_inner_sum):
    v_temp = 0
    for l in np.arange(len(preference_weights)):
        v_temp += arrival_probabilities[l] * \
                  sum(indices_inner_sum * (revenues - np.apply_along_axis(sum, 1, A.T * pi)) * preference_weights[l, :]) / \
                  (sum(indices_inner_sum * preference_weights[l, :]) + var_no_purchase_preferences[l])
    return v_temp


#%%



















# needed results: for each time and capacity: value of coming revenues, optimal set to offer, list of other optimal sets
# capacities max: run it just for one set of capacities, the rest follows)
def return_raw_data_per_time(capacity):
    rows = list(range(capacity + 1))
    row_names = [str(i) for i in rows]
    df = pd.DataFrame(index=row_names,
                      columns=['value', 'offer_set_optimal', 'num_offer_set_optimal'])
    df.value = pd.to_numeric(df.value)
    return df


# %%%
final_results = list(return_raw_data_per_time(capacity_max) for t in np.arange(T+1))
final_results[T].value = 0.0
final_results[T].offer_set_optimal = 0
final_results[T].num_offer_set_optimal = 0

total_results = list(object for u in var_no_purchase_preferences)

index_total_results = 0
for no_purchase_preference in var_no_purchase_preferences:

    offer_sets = pd.DataFrame(get_offer_sets_all(products))
    prob = offer_sets.apply(tuple, axis=1)
    prob = prob.apply(customer_choice_vector, args=(preference_weights, no_purchase_preference, arrival_probabilities))
    prob = pd.DataFrame(prob.values.tolist(), columns=np.arange(len(products) + 1))  # probabilities for each product

    prob = np.array(prob)

    for t in times[::-1]:  # running through the other way round, starting from second last time
        print("Time point: ", t)

        # c = 0
        final_results[t].iloc[0, :] = (0.0, 0, 0)

        # c > 0
        # Delta-Lookup (same for all products, as each product costs 1 capacity)
        v_t_plus_1: np.array = np.array(final_results[t+1].value)
        indices_c_minus_A = np.repeat([[i] for i in np.arange(capacity_max)+1], repeats=len(products), axis=1) - \
                            np.repeat(A, repeats=capacity_max, axis=0)
        delta_value = np.repeat([[i] for i in v_t_plus_1[1:]], repeats=len(products), axis=1) - \
                      v_t_plus_1[indices_c_minus_A]

        # TODO parallize via three dimensional array (t, c, offer-sets)
        for c in np.arange(capacity_max)+1:
            np_max = prob[:, 0:-1]*(revenues-delta_value[c-1])
            np_max = np.sum(np_max, axis=1)

            # get maximum value, index of offer_set of maximum value, count of maximum values
            final_results[t].iloc[c, :] = (np.amax(np_max),
                                           np.argmax(np_max),
                                           np.unique(np_max, return_counts=True)[1][-1])

        final_results[t].value += final_results[t+1].value

    total_results[index_total_results] = final_results
    index_total_results += 1

# %%
# write result of calculations
with open(newpath+"\\totalresults.data", "wb") as filehandle:
    pickle.dump(total_results, filehandle)

# %%
# write summary latex
erg_paper = pd.DataFrame(index=range(len(var_no_purchase_preferences)*len(var_capacities)),
                         columns=["capacity", "no-purchase preference", "DP-value", "DP-optimal offer set"])
i = 0
for u in range(len(var_no_purchase_preferences)):
    for c in var_capacities:
        tmp = total_results[u][0].iloc[c[0], :]
        erg_paper.iloc[i, :] = (c[0], u+1,
                                round(float(tmp.value), 2),
                                str(tuple(offer_sets.iloc[int(tmp.offer_set_optimal), :])))
        i += 1
erg_latex = open(newpath+"\\erg_paper.txt", "w+")  # write and create (if not there)
print(erg_paper.to_latex(), file=erg_latex)
erg_latex.close()

p = final_results[200].plot.hist()


# %%
time_elapsed = time.time() - time_start
print("\n\nTotal time needed:\n", time_elapsed, "seconds = \n", time_elapsed/60, "minutes", file=logfile)
logfile.close()
print("\n\n\n\nDone. Time elapsed:", time.time() - time_start, "seconds.")
print("Results stored in: "+newpath)
