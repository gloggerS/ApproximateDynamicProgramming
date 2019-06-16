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

import random

#%%
# Get settings, prepare data, create storage for results
print("API linear single leg starting.\n\n")

# settings
settings = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
example = settings.iloc[0, 1]
use_variations = (settings.iloc[1, 1] == "True") | (settings.iloc[1, 1] == "true")  # if var. capacities should be used
storage_folder = example + "-" + str(use_variations) + "-API-lin-evaluation-" + time.strftime("%y%m%d-%H%M")
K = int(settings.loc[settings[0] == "K", 1])
I = int(settings.loc[settings[0] == "I", 1])
epsilon = eval(str(settings.loc[settings[0] == "epsilon", 1].item()))
exponential_smoothing = settings.loc[settings[0] == "exponential_smoothing", 1].item()
exponential_smoothing = (exponential_smoothing == "True") | (exponential_smoothing == "true")
online_K = int(settings.loc[settings[0] == "online_K", 1].item())

#%%
# todo get the folder in which the parameters (theta, pi) to use are stored; e.g. via sys.argv (this script is called after the calculation of those parameters)
for i in sys.argv:
    print(i)

result_folder = 'C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\smallTest2-False-API-lin-190616-1023'

#%%
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
print("Time (starting):", datetime.datetime.now(), "\n", file=logfile)
time_start = time.time()

# settings
for row in np.arange(len(settings)):
    print(settings.loc[row, 0], ":\t", settings.loc[row, 1])
    print(settings.loc[row, 0], ":\t", settings.loc[row, 1], file=logfile)
print("correct epsilon for epsilon-greedy strategy: ", len(epsilon) == K+1)
print("\ncorrect epsilon for epsilon-greedy strategy: ", len(epsilon) == K+1, file=logfile)

# variations (capacity and no purchase preference)
if use_variations:
    var_capacities, var_no_purchase_preferences = get_variations(example)
    preferences_no_purchase = var_no_purchase_preferences[0]
else:
    capacities, preferences_no_purchase = get_capacities_and_preferences_no_purchase(example)
    var_capacities = np.array([capacities])
    var_no_purchase_preferences = np.array([preferences_no_purchase])

# no varying capacity here
capacities = var_capacities[0]

# other data
resources, \
    products, revenues, A, \
    customer_segments, preference_weights, arrival_probabilities, \
    times = get_data_without_variations(example)
T = len(times)

print("\nEverything set up.")

#%%
def determine_offer_tuple(pi, eps=0):
    """
    OLD Implementation
    Determines the offerset given the bid prices for each resource.

    Implement the Greedy Heuristic from Bront et al: A Column Generation Algorithm ... 4.2.2
    and extend it for the epsilon greedy strategy

    :param pi:
    :param eps: epsilon greedy strategy (will be set to 0 to have no influence)
    :return:
    """

    # setup
    offer_tuple = np.zeros_like(revenues)

    # epsilon greedy strategy - offer no products
    eps_prob = random.random()
    if eps_prob < eps/2:
        return tuple(offer_tuple)

    # line 1
    s_prime = revenues - np.apply_along_axis(sum, 1, A.T * pi) > 0
    if all(np.invert(s_prime)):
        return tuple(offer_tuple)

    # epsilon greedy strategy - offer all products
    if eps_prob < eps:
        return tuple(offer_tuple)

    # epsilon greedy strategy - greedy heuristic

    # line 2-3
    # offer_sets_to_test has in each row an offer set, we want to test
    offer_sets_to_test = np.zeros((sum(s_prime), len(revenues)))
    offer_sets_to_test[np.arange(sum(s_prime)), np.where(s_prime)] = 1
    offer_sets_to_test += offer_tuple
    offer_sets_to_test = (offer_sets_to_test > 0)

    value_marginal = np.apply_along_axis(calc_value_marginal, 1, offer_sets_to_test, pi)

    offer_tuple[np.argmax(value_marginal)] = 1
    s_prime = s_prime & offer_tuple == 0
    v_s = np.amax(value_marginal)

    # line 4
    while True:
        # 4a
        # offer_sets_to_test has in each row an offer set, we want to test
        offer_sets_to_test = np.zeros((sum(s_prime), len(revenues)))
        offer_sets_to_test[np.arange(sum(s_prime)), np.where(s_prime)] = 1
        offer_sets_to_test += offer_tuple
        offer_sets_to_test = (offer_sets_to_test > 0)

        # 4b
        value_marginal = np.apply_along_axis(calc_value_marginal, 1, offer_sets_to_test, pi)

        if np.amax(value_marginal) > v_s:
            v_s = np.amax(value_marginal)
            offer_tuple[np.argmax(value_marginal)] = 1
            s_prime = s_prime & offer_tuple == 0
            if all(offer_tuple == 1):
                break
        else:
            break
    return tuple(offer_tuple)


def calc_value_marginal(indices_inner_sum, pi):
    v_temp = 0
    for l in np.arange(len(preference_weights)):
        v_temp += arrival_probabilities[l] * \
                  sum(indices_inner_sum * (revenues - np.apply_along_axis(sum, 1, A.T * pi)) *
                      preference_weights[l, :]) / \
                  (sum(indices_inner_sum * preference_weights[l, :]) + var_no_purchase_preferences[l])
    return v_temp


def simulate_sales(offer_set):
    customer = int(np.random.choice(np.arange(len(arrival_probabilities)+1),
                                    size=1,
                                    p=np.array([*arrival_probabilities, 1-sum(arrival_probabilities)])))
    if customer == len(arrival_probabilities):
        return len(products)  # no customer arrives => no product sold
    else:
        return int(np.random.choice(np.arange(len(products) + 1),
                                    size=1,
                                    p=customer_choice_individual(offer_set,
                                                                 preference_weights[customer],
                                                                 preferences_no_purchase[customer])))

# %%
# Actual Code

# theta and pi as calculated
with open(result_folder+"\\thetaResult.data", "rb") as filehandle:
    thetas = pickle.load(filehandle)
with open(result_folder+"\\piResult.data", "rb") as filehandle:
    pis = pickle.load(filehandle)

#%%
# online_K+1 policy iterations (starting with 0)
v_results = np.array([np.zeros(len(times))]*online_K)
c_results = np.array([np.zeros(shape=(len(times), len(capacities)))]*online_K)

# to use in single timestep (will be overwritten)
pi = np.zeros(len(resources))

for k in np.arange(online_K)+1:
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

        # line 12  (epsilon greedy strategy)
        pi[c == 0] = np.inf
        pi[c > 0] = pis[t][c > 0]
        offer_set = determine_offer_tuple(pi, eps=0)

        # line 13  (simulate sales)
        sold = simulate_sales(offer_set)

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


# %%
# write result of calculations
with open(newpath+"\\vAll.data", "wb") as filehandle:
    pickle.dump(v_results, filehandle)

with open(newpath+"\\cAll.data", "wb") as filehandle:
    pickle.dump(c_results, filehandle)

with open(newpath+"\\vResults.data", "wb") as filehandle:
    pickle.dump(v_results[:, 0], filehandle)


# %%
time_elapsed = time.time() - time_start
print("\n\nTotal time needed:\n", time_elapsed, "seconds = \n", time_elapsed/60, "minutes", file=logfile)
logfile.close()
print("\n\n\n\nDone. Time elapsed:", time.time() - time_start, "seconds.")
print("Results stored in: "+newpath)
