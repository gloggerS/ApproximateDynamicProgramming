"""
This script will calculate the CDLP and store them in a large dataframe
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
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
    customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
    epsilon, exponential_smoothing \
    = setup("CDLPSingleLeg")

# capacities max: run it just for one set of capacities, the rest follows)
capacity_max = int(np.max(var_capacities, axis=0))


# %%
# Actual Code


# needed results: for each time and capacity: expected value of revenues to go, optimal set to offer,
# list of other optimal sets
# capacities max: run it just for one set of capacities, the rest follows)
def return_raw_data_per_time(capacity):
    rows = list(range(capacity + 1))
    row_names = [str(i) for i in rows]
    df = pd.DataFrame(index=row_names,
                      columns=['value', 'offer_set_optimal', 'num_offer_set_optimal'])
    df.value = pd.to_numeric(df.value)
    return df


def purchase_rate_vector(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities):
    """
    P_j(S) for all j, P_0(S) at the end

    :param offer_set_tuple: S
    :param preference_weights
    :param preference_no_purchase
    :param arrival_probabilities
    :return: P_j(S) for all j, P_0(S) at the end
    """
    probs = np.zeros(len(offer_set_tuple) + 1)
    p = arrival_probabilities/(sum(arrival_probabilities))
    for l in np.arange(len(preference_weights)):
        probs += p[l] * customer_choice_individual(offer_set_tuple, preference_weights[l, :],
                                                   preference_no_purchase[l])
    return probs


def revenue(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities):
    """
    R(S)

    :param offer_set_tuple: S
    :return: R(S)
    """
    return sum(revenues * purchase_rate_vector(offer_set_tuple, preference_weights, preference_no_purchase,
                                               arrival_probabilities)[:-1])


def quantity_i(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities, i):
    """
    Q_i(S)

    :param offer_set_tuple: S
    :param i: resource i
    :return: Q_i(S)
    """
    return sum(A[i, :] * purchase_rate_vector(offer_set_tuple, preference_weights, preference_no_purchase,
                                              arrival_probabilities)[:-1])

# %%
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
    prob[:, -1] = 1 - np.apply_along_axis(sum, 1, prob[:, :-1])

    for t in times[::-1]:  # running through the other way round, starting from second last time
        print("Time point: ", t)

        # c = 0
        final_results[t].iloc[0, :] = (0.0, 0, 0)

        # c > 0
        if True:
            # direct implementation
            for c in np.arange(capacity_max)+1:
                np_max = prob*np.append(np.array(revenues + final_results[t+1].iloc[c-A[0], 0]), final_results[t+1].iloc[c, 0])
                np_max = np.sum(np_max, axis=1)

                # get maximum value, index of offer_set of maximum value, count of maximum values
                final_results[t].iloc[c, :] = (np.amax(np_max),
                                               np.argmax(np_max),
                                               np.unique(np_max, return_counts=True)[1][-1])
        else:
            # complicated via delta function
            # Delta-Lookup (same for all products, as each product costs 1 capacity)
            # calculate all delta values (for all capacities) at once
            v_t_plus_1: np.array = np.array(final_results[t+1].value)
            indices_c_minus_A = np.repeat([[i] for i in np.arange(capacity_max)+1], repeats=len(products), axis=1) - \
                                np.repeat(A, repeats=capacity_max, axis=0)
            delta_value = np.repeat([[i] for i in v_t_plus_1[1:]], repeats=len(products), axis=1) - \
                          v_t_plus_1[indices_c_minus_A]

            # TODO parallize via three dimensional array (t, c, offer-sets)
            for c in np.arange(capacity_max)+1:
                np_max2 = prob[:, 0:-1]*(revenues-delta_value[c-1])
                np_max2 = np.sum(np_max2, axis=1)

                # get maximum value, index of offer_set of maximum value, count of maximum values
                final_results[t].iloc[c, :] = (np.amax(np_max2),
                                               np.argmax(np_max2),
                                               np.unique(np_max2, return_counts=True)[1][-1])

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
                         columns=["capacity", "no-purchase preference", "DP-value", "DP-optimal offer set at start"])
i = 0
for u in range(len(var_no_purchase_preferences)):
    for c in var_capacities:
        tmp = total_results[u][0].iloc[c[0], :]
        erg_paper.iloc[i, :] = (c[0], var_no_purchase_preferences[u][0],
                                round(float(tmp.value), 2),
                                str(tuple(offer_sets.iloc[int(tmp.offer_set_optimal), :])))
        i += 1
erg_latex = open(newpath+"\\erg_paper.txt", "w+")  # write and create (if not there)
print(erg_paper.to_latex(), file=erg_latex)
erg_latex.close()


# %%
wrapup(logfile, time_start, newpath)
