"""
This script will calculate the Value Expected exactly and store them in a large dataframe
as specified in 0_settings.csv.
"""

import pandas as pd
import numpy as np

import datetime
import time

import os
from shutil import copyfile

from A_data_read import *
from B_helper import *
from ast import literal_eval

from joblib import Parallel, delayed, dump, load
import multiprocessing

#%%
# Get settings, prepare data, create storage for results
print("Value Expected starting.\n\n")

# settings
settings = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
example = settings.iloc[0, 1]
use_variations = (settings.iloc[1, 1] == "True") | (settings.iloc[1, 1] == "true")  # if var. capacities should be used
storage_folder = settings.iloc[2, 1] + "-" + example + "-" + str(use_variations)

# data
dat = get_all(example)
print("\n Data used. \n")
for key in dat.keys():
    print(key, ":\n", dat[key])
print("\n\n")
del dat

# prepare storage location
existing_folders = os.listdir(os.getcwd()+"\\Results\\")
if storage_folder in existing_folders:
    # TODO put this in do while
    answer = input("The calculation was already done. Do you really want to restart calculation? [y/n]")
    if answer == "n":
        exit()  # stops the script immediately
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

# capacities max: run it just for one set of capacities, the rest follows)
capacities_max = np.max(var_capacities, axis=0)

# other data
resources, \
    products, revenues, A, \
    customer_segments, preference_weights, arrival_probabilities, \
    times = get_data_without_variations(example)
T = len(times)

print("\nEverything set up.")

# %%
# Actual Code

# needed results: for each time and capacity: value of coming revenues, optimal set to offer, list of other optimal sets
# capacities max: run it just for one set of capacities, the rest follows)
def return_raw_data_per_time(capacities_max):
    rows = list(itertools.product(*[range(i+1) for i in capacities_max]))
    rows2 = [str(i) for i in rows]
    df = pd.DataFrame(index=rows2,
                      columns=['value', 'offer_set_optimal', 'num_offer_set_optimal'])
    return df

def optimal_value(t, capacities, no_purchase_preferences):
    # get to purchase probabilities for each product
    offer_sets = pd.DataFrame(get_offer_sets_all(products))

    # reduce offersets to possible offersets
    for c_index in range(capacities.__len__()):
        if capacities[c_index] == 0:
            offer_sets -= A[c_index, :]
    offer_sets[offer_sets<0] = 0

    prob = offer_sets.apply(tuple, axis=1)
    prob2 = prob.apply(purchase_rate_vector, args=(preference_weights, no_purchase_preferences, arrival_probabilities))
    prob3 = pd.DataFrame(prob2.values.tolist(), columns=np.arange(len(products)+1))  # probabilities for each product

    print(capacities)

    # get to expected value for each product purchase
    work = np.array([*1.0*revenues, 0])
    for j in offer_sets.columns[offer_sets.max() == 1]:  # only those products, that can be offered
        work[j] += get_optimal_value(t+1, tuple(capacities-A[:,j]))
    work[-1] = get_optimal_value(t+1, tuple(capacities))

    res = prob3*work
    res2 = res.apply(sum, axis=1)
    index_max = res2.idxmax()

    return res2[index_max], str(prob[index_max]), sum(res2 == res2[index_max])


def optimal_value_end(capacities, no_purchase_preferences):
    # get to purchase probabilities for each product
    offer_sets = pd.DataFrame(get_offer_sets_all(products))

    # reduce offersets to possible offersets
    for c_index in range(capacities.__len__()):
        if capacities[c_index] == 0:
            offer_sets -= A[c_index, :]
    offer_sets[offer_sets<0] = 0

    prob = offer_sets.apply(tuple, axis=1)
    prob2 = prob.apply(purchase_rate_vector, args=(preference_weights, no_purchase_preferences, arrival_probabilities))
    prob3 = pd.DataFrame(prob2.values.tolist(), columns=np.arange(len(products) + 1))  # probabilities for each product

    work = prob3*np.array([*revenues, 0])
    work2 = work.apply(sum, axis=1)
    index_max = work2.idxmax()

    return work2[index_max], str(prob[index_max]), sum(work2 == work2[index_max])

# parallelisation over no_purchase_preferences
no_purchase_preferences = var_no_purchase_preferences[0]

final_results = list(return_raw_data_per_time(capacities_max) for t in times)

def get_optimal_value(t, c):
    return final_results[t].loc[str(c)].value

# start from the end (last possibility to sell); mostly the same calculation (as just capacity > 0 matters)
t = times[-1]
c_short = list(itertools.product([1, 0], repeat = capacities_max.size))
rows = [literal_eval(i) for i in final_results[t].index]
rows = np.array(rows)
rows[rows>1] = 1
for c in c_short:
    rows_to_consider = np.apply_along_axis(all, 1, rows == c)
    print(sum(rows_to_consider))
    df = pd.DataFrame([optimal_value_end(c, no_purchase_preferences)], columns=final_results[t].columns)
    df2 = df.iloc[np.repeat(0, sum(rows_to_consider)), :]
    df2.index = final_results[t].loc[rows_to_consider].index
    final_results[t].loc[rows_to_consider] = df2

num_cores = 2  # multiprocessing.cpu_count()

for t in times[::-1][1:]:  # running through the other way round, starting from second last time
    rows = [literal_eval(i) for i in final_results[t].index]
    res = Parallel(n_jobs=num_cores)(delayed(optimal_value)(t, c, no_purchase_preferences) for c in rows)
    final_results[t] = res
