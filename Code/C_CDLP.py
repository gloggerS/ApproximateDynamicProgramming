"""
This script will calculate the CDLP (choice based deterministic linear programme).
"""

import pandas as pd
import numpy as np

import datetime
import time

import os
from shutil import copyfile

from A_data_read import *
from B_helper import *

#%%
# Get settings, prepare data, create storage for results
print("CDLP starting.\n\n")

# settings
settings = pd.read_csv("0_settings.csv", delimiter=";", header=None)
example = settings.iloc[0, 1]
use_variations = settings.iloc[1, 1]  # true if varying capacities should be used


# prepare storage location
newpath = os.getcwd()+"\\Results\\CDLP-"+example+"-"+str(datetime.datetime.now()).replace(":", "-").replace(".", "-")
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
    capacities, no_purchase_preference = get_capacities_and_preferences_no_purchase(example)
    var_capacities = np.array([capacities])
    var_no_purchase_preference = np.array([no_purchase_preference])

# other data
resources, \
    products, revenues, A, \
    customer_segments, preference_weights, arrival_probabilities, \
    times = get_data_without_variations(example)
T = len(times)

print("\nEverything set up.")

#%%
# Actual Code
def CDLP(capacities, preference_no_purchase, offer_sets: np.ndarray):
    """
    Implements (4) of Bront et al. Needs the offer-sets to look at (N) as input.

    :param offer_sets: N
    :return: dictionary of (offer set, time offered),
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()

    offer_sets = pd.DataFrame(offer_sets)
    lam = sum(arrival_probabilities)
    T = len(times)

    S = {}
    R = {}
    Q = {}
    for index, offer_array in offer_sets.iterrows():
        S[index] = tuple(offer_array)
        R[index] = revenue(tuple(offer_array), preference_weights, preference_no_purchase, arrival_probabilities,
                           revenues)
        temp = {}
        for i in resources:
            temp[i] = quantity_i(tuple(offer_array), preference_weights, preference_no_purchase,
                                 arrival_probabilities, i, A)
        Q[index] = temp

    try:
        m = Model()

        # Variables
        mt = m.addVars(offer_sets.index.values, name="t", lb=0.0)  # last constraint

        # Objective Function
        m.setObjective(lam * quicksum(R[s] * mt[s] for s in offer_sets.index.values), GRB.MAXIMIZE)

        mc = {}
        # Constraints
        for i in resources:
            mc[i] = m.addConstr(lam * quicksum(Q[s][i] * mt[s] for s in offer_sets.index.values), GRB.LESS_EQUAL,
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

        dualPi = np.zeros_like(resources, dtype=float)
        for i in resources:
            dualPi[i] = mc[i].pi
        dualSigma = msigma.pi

        valOpt = m.objVal

        return ret, valOpt, dualPi, dualSigma

    except GurobiError:
        print('Error reported')


# %%
# Run CDLP
num_rows = len(var_capacities)*len(var_no_purchase_preferences)
df = pd.DataFrame(index=np.arange(num_rows), columns=['c', 'u', 'DP', 'CDLP'])
indexi = 0
for capacity in var_capacities:
    for preference_no_purchase in var_no_purchase_preferences:
        print(capacity, "-", preference_no_purchase)
        print(str(datetime.datetime.now()), ":", capacity, "-", preference_no_purchase, file=logfile)

        df.loc[indexi] = [capacity, preference_no_purchase, value_expected(capacities=capacity, t=0,
                                                                           preference_no_purchase=preference_no_purchase,
                                                                           example=example),
                          CDLP_by_column_generation(capacities=capacity, preference_no_purchase=preference_no_purchase)]
        indexi += 1

df.to_pickle("CDLP-"+example+"-"+use_variations+".pkl", newpath)

time_elapsed = time.time() - time_start
print("\n\nTotal time needed:\n", time_elapsed, "seconds = \n", time_elapsed/60, "minutes", file=logfile)
logfile.close()
print("Done. Time elapsed:", time.time() - time_start)