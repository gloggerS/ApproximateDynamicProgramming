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
storage_folder = example + "-" + str(use_variations) + "-API-lin-" + time.strftime("%y%m%d-%H%M")
K = int(settings.loc[settings[0] == "K", 1])
I = int(settings.loc[settings[0] == "I", 1])
epsilon = eval(str(settings.loc[settings[0] == "epsilon", 1].item()))
exponential_smoothing = settings.loc[settings[0] == "exponential_smoothing", 1].item()
exponential_smoothing = (exponential_smoothing == "True") | (exponential_smoothing == "true")

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


# %%
# Actual Code
# K+1 policy iterations (starting with 0)
# T time steps
theta_all = np.array([[np.zeros(1)]*T]*(K+1))
pi_all = np.array([[np.zeros(len(resources))]*T]*(K+1))

# theta and pi for each time step
# line 1
thetas = 0
pis = np.zeros(len(resources))

for k in np.arange(K)+1:
    np.random.seed(k)
    random.seed(k)

    v_samples = np.array([np.zeros(len(times))]*I)
    c_samples = np.array([np.zeros(shape=(len(times), len(capacities)))]*I)

    for i in np.arange(I):
        # line 3
        r_sample = np.zeros(len(times))  # will become v_sample
        c_sample = np.zeros(shape=(len(times), len(capacities)), dtype=int)

        # line 5
        c = copy.deepcopy(capacities)  # (starting capacity at time 0)

        for t in times:
            # line 7  (starting capacity at time t)
            c_sample[t] = c

            # line 8-11  (adjust bid price)
            pis[c_sample[t] == 0] = np.inf
            pis[c_sample[t] > 0] = pi_all[k - 1][t][c_sample[t] > 0]

            # line 12  (epsilon greedy strategy)
            offer_set = determine_offer_tuple(pis, epsilon[k])

            # line 13  (simulate sales)
            sold = simulate_sales(offer_set)

            # line 14
            try:
                r_sample[t] = revenues[sold]
                c -= A[:, sold]
            except IndexError:
                # no product was sold
                pass

        # line 16-18
        v_samples[i] = np.cumsum(r_sample[::-1])[::-1]
        c_samples[i] = c_sample

    # line 20
    theta_all[k], pi_all[k] = update_parameters(v_samples, c_samples, theta_all[k-1], pi_all[k-1], k)


# %%
# write result of calculations
with open(newpath+"\\thetaAll.data", "wb") as filehandle:
    pickle.dump(theta_all, filehandle)

with open(newpath+"\\piAll.data", "wb") as filehandle:
    pickle.dump(pi_all, filehandle)


# %%
time_elapsed = time.time() - time_start
print("\n\nTotal time needed:\n", time_elapsed, "seconds = \n", time_elapsed/60, "minutes", file=logfile)
logfile.close()
print("\n\n\n\nDone. Time elapsed:", time.time() - time_start, "seconds.")
print("Results stored in: "+newpath)
