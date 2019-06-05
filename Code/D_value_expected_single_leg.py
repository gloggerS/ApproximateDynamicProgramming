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

#%%
# Get settings, prepare data, create storage for results
print("Value Expected single leg starting.\n\n")

# settings
settings = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
example = settings.iloc[0, 1]
use_variations = (settings.iloc[1, 1] == "True") | (settings.iloc[1, 1] == "true")  # if var. capacities should be used
storage_folder = example + "-" + str(use_variations) + "-DP-" + time.strftime("%y%m%d-%H%M")

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

# capacities max: run it just for one set of capacities, the rest follows)
capacity_max = int(np.max(var_capacities, axis=0))

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
def return_raw_data_per_time(capacity_max):
    rows = list(range(capacity_max+1))
    row_names = [str(i) for i in rows]
    df = pd.DataFrame(index=row_names,
                      columns=['value', 'offer_set_optimal', 'num_offer_set_optimal'])
    df.value = pd.to_numeric(df.value)
    return df


# %%%
no_purchase_preference = var_no_purchase_preferences[2]

final_results = list(return_raw_data_per_time(capacity_max) for t in np.arange(T+1))
final_results[T].value = 0.0
final_results[T].offer_set_optimal = 0
final_results[T].num_offer_set_optimal = 0

offer_sets = pd.DataFrame(get_offer_sets_all(products))
prob = offer_sets.apply(tuple, axis=1)
prob = prob.apply(purchase_rate_vector, args=(preference_weights, no_purchase_preference, arrival_probabilities))
prob = pd.DataFrame(prob.values.tolist(), columns=np.arange(len(products) + 1))  # probabilities for each product

prob = np.array(prob)

for t in times[::-1]:  # running through the other way round, starting from second last time
    print("Time point: ", t)

    # c = 0
    final_results[t].iloc[0, :] = (0.0, 0, 0)

    # c > 0
    # Delta-Lookup (same for all products, as each product costs 1 capacity)
    v_t_plus_1 = np.array(final_results[t+1].value)
    indices_c_minus_A = np.repeat([[i] for i in np.arange(capacity_max)+1], repeats=len(products), axis=1) - \
                        np.repeat(A, repeats=capacity_max, axis=0)
    delta_value = np.repeat([[i] for i in v_t_plus_1[1:]], repeats=len(products), axis=1) - v_t_plus_1[indices_c_minus_A]

    # TODO parallize via three dimensional array (t, c, offer-sets)
    for c in np.arange(capacity_max)+1:
        np_max = prob[:, 0:-1]*(revenues-delta_value[c-1])
        np_max = np.sum(np_max, axis=1)

        # get maximum value, index of offer_set of maximum value, count of maximum values
        final_results[t].iloc[c, :] = (max(np_max), np.argmax(np_max), np.unique(np_max, return_counts=True)[1][-1])

    final_results[t].value += final_results[t+1].value


p = final_results[200].plot.hist()


def optimal_value_end(capacities, no_purchase_preferences):
    # get to purchase probabilities for each product


    work = prob3*np.array([*revenues, 0])
    work2 = work.apply(sum, axis=1)
    index_max = work2.idxmax()

    return work2[index_max], str(prob[index_max]), sum(work2 == work2[index_max])










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

    # get to expected value for each product purchase
    work = np.array([*1.0*revenues, 0])
    for j in offer_sets.columns[offer_sets.max() == 1]:  # only those products, that can be offered
        work[j] += get_optimal_value(t+1, tuple(capacities-A[:,j]))
    work[-1] = get_optimal_value(t+1, tuple(capacities))

    res = prob3*work
    res2 = res.apply(sum, axis=1)
    index_max = res2.idxmax()

    return res2[index_max], str(prob[index_max]), sum(res2 == res2[index_max])




def get_optimal_value(t, c):
    return final_results[t].loc[str(c)].value

# %%
# TODO parallelisation over no_purchase_preferences
no_purchase_preferences = var_no_purchase_preferences[0]


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

num_cores = 8 # multiprocessing.cpu_count()

for t in times[::-1][1:]:  # running through the other way round, starting from second last time
    print("Time point: ", t)
    rows = [literal_eval(i) for i in final_results[t].index]

    # parallel way
    # res = Parallel(n_jobs=num_cores)(delayed(optimal_value)(t, c, no_purchase_preferences) for c in rows)
    # final_results[t] = res

    # iterative way
    for c in rows:
        final_results[t].loc[str(c)] = optimal_value(t, c, no_purchase_preferences)

# %%
# write
with open("finalresults.data", "wb") as filehandle:
    pickle.dump(final_results, filehandle)

path_storage = move("finalresults.data", newpath)

time_elapsed = time.time() - time_start
print("\n\nTotal time needed:\n", time_elapsed, "seconds = \n", time_elapsed/60, "minutes", file=logfile)
logfile.close()
print("\n\n\n\nDone. Time elapsed:", time.time() - time_start, "seconds.")
print("Results stored in: "+newpath)

# read
# with open(path_storage, "rb") as filehandle:
#     f = pickle.load(filehandle)
