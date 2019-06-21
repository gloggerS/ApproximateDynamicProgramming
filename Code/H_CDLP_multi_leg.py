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
capacities = var_capacities[0]
no_purchase_preference = var_no_purchase_preferences[0]


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


def revenue(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities):
    """
    R(S)

    :param offer_set_tuple: S
    :return: R(S)
    """
    return sum(revenues * customer_choice_vector(offer_set_tuple, preference_weights, preference_no_purchase,
                                                 arrival_probabilities)[:-1])


def quantity_i(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities, i):
    """
    Q_i(S)

    :param offer_set_tuple: S
    :param i: resource i
    :return: Q_i(S)
    """
    return sum(A[i, :] * customer_choice_vector(offer_set_tuple, preference_weights, preference_no_purchase,
                                                arrival_probabilities)[:-1])


def CDLP(offer_sets):
    """
    Implements (4) of Bront et al. Needs the offer-sets to look at (N) as input.

    :param offer_sets: N
    :return: dictionary of (offer set, time offered),
    """

    S = {}
    R = {}
    Q = {}

    for index, offer_array in offer_sets.iterrows():
        if index != 0:
            S[index] = tuple(offer_array)
            R[index] = revenue(tuple(offer_array), preference_weights, no_purchase_preference, arrival_probabilities)
            temp = {}
            for i in resources:
                temp[i] = quantity_i(tuple(offer_array), preference_weights, no_purchase_preference,
                                     arrival_probabilities, i)
            Q[index] = temp

    try:
        m = Model()

        # Variables
        mt = m.addVars(offer_sets.index.values, name="t", lb=0.0)  # last constraint

        # Objective Function
        m.setObjective(quicksum(R[s] * mt[s] for s in offer_sets.index.values), GRB.MAXIMIZE)

        mc = {}
        # Constraints
        for i in resources:
            mc[i] = m.addConstr(quicksum(Q[s][i] * mt[s] for s in offer_sets.index.values), GRB.LESS_EQUAL,
                                capacities[i],
                                name="constraintOnResource")
        msigma = m.addConstr(quicksum(mt[s] for s in offer_sets.index.values), GRB.LESS_EQUAL, T)

        m.optimize()

        ret = {}
        pat = r".*?\[(.*)\].*"
        for v in m.getVars():
            if v.X > 0:
                match = re.search(pat, v.VarName)
                erg_index = match.group(1)
                ret[int(erg_index)] = (tuple(offer_sets.loc[int(erg_index)]), v.X)
                print(offer_sets.loc[int(erg_index)], ": ", v.X)

        dual_pi = np.zeros_like(resources)
        for i in resources:
            dual_pi[i] = mc[i].pi
        dual_sigma = msigma.pi

        val_opt = m.objVal

        return ret, val_opt, dual_pi, dual_sigma

    except GurobiError:
        print('Error reported')


# %%

offer_sets = pd.DataFrame(get_offer_sets_all(products))
offer_sets = offer_sets[1:]  # exclude empty offerset, otherwise result of Bront for example0 can't be reproduced

ret, val_opt, dual_pi, dual_sigma = CDLP(offer_sets)

for k in ret.keys():
    print("Decision variable: \t", ret[k])
print("Optimal value: \t", val_opt)
print("Dual pi's: \t", dual_pi)
print("Dual sigma: \t", dual_sigma)

for k in ret.keys():
    print("Decision variable: \t", ret[k], file=logfile)
print("Optimal value: \t", val_opt, file=logfile)
print("Dual pi's: \t", dual_pi, file=logfile)
print("Dual sigma: \t", dual_sigma, file=logfile)


# %%
wrapup(logfile, time_start, newpath)
