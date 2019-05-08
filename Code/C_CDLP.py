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


def column_MIP(preference_no_purchase, pi, w=0):  # pass w to test example for greedy heuristic
    """
    Implements MIP formulation on p. 775 lhs

    :param pi:
    :param w:
    :return: optimal tuple of products to offer
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()

    K = 1/min(preference_no_purchase.min(), np.min(preference_weights[np.nonzero(preference_weights)]))+1

    if isinstance(w, int) and w == 0:  # 'and' is lazy version of &
        w = np.zeros_like(revenues, dtype=float)
        for j in products:
            w[j] = revenues[j] - sum(A[:, j]*pi)

    try:
        m = Model()

        mx = {}
        my = {}
        mz = {}

        # Variables
        for j in products:
            my[j] = m.addVar(0, 1, vtype=GRB.BINARY, name="y["+str(j)+"]")
        for l in customer_segments:
            mx[l] = m.addVar(0.0, name="x["+str(l)+"]")
            temp = {}
            for j in products:
                temp[j] = m.addVar(0.0, name="z["+str(l)+","+str(j)+"]")
            mz[l] = temp

        # Objective
        m.setObjective(quicksum(arrival_probabilities[l] * w[j] * preference_weights[l, j] * mz[l][j]
                                for l in customer_segments for j in products), GRB.MAXIMIZE)

        # Constraints
        mc1 = m.addConstrs((mx[l]*preference_no_purchase[l] +
                            quicksum(preference_weights[l, j]*mz[l][j] for j in products) == 1
                            for l in customer_segments), name="mc1")
        mc2 = m.addConstrs((mx[l] - mz[l][j] <= K - K*my[j] for l in customer_segments for j in products),
                           name="mc2")
        mc3 = m.addConstrs((mz[l][j] <= mx[l] for l in customer_segments for j in products), name="mc3")
        mc4 = m.addConstrs((mz[l][j] <= K*my[j] for l in customer_segments for j in products), name="mc4")

        m.optimize()

        y = np.zeros_like(revenues)
        for j in products:
            y[j] = my[j].x

        return tuple(y), m.objVal

    except GurobiError:
        print('Error reported')


def column_greedy(preference_no_purchase, pi, w=0, dataName=""):  # pass w to test example for greedy heuristic
    """
    Implements Greedy Heuristic on p. 775 rhs

    :param pi:
    :param w:
    :return: heuristically optimal tuple of products to offer
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations(dataName)

    # Step 1
    y = np.zeros_like(revenues)

    if isinstance(w, int) and w == 0:  # and is lazy version of &
        w = np.zeros_like(revenues, dtype=float)  # calculate the opportunity costs
        for j in products:
            w[j] = revenues[j] - sum(A[:, j]*pi)

    # Step 2
    Sprime = set(np.where(w > 0)[0])

    # Step 3
    value_marginal = np.zeros_like(w, dtype=float)
    for j in Sprime:
        for l in customer_segments:
            value_marginal[j] += preference_weights[l, j]/(preference_weights[l, j] + preference_no_purchase[l])
        value_marginal[j] *= w[j]

    jstar = np.argmax(value_marginal)
    v_new = value_marginal[jstar]

    S = {jstar}
    Sprime = Sprime-S

    # Step 4
    while True:
        v_akt = copy.deepcopy(v_new)  # deepcopy to be on the safe side
        v_temp = np.zeros_like(revenues, dtype=float)  # uses more space then necessary, but simplifies indices below
        for j in Sprime:
            for l in customer_segments:
                z = 0
                n = 0
                for p in S.union({j}):
                    z += w[p]*preference_weights[l, p]
                    n += preference_weights[l, p]
                n += preference_no_purchase[l]
                v_temp[j] += arrival_probabilities[l]*z/n
        jstar = np.argmax(value_marginal)  # argmax always returns index of first maxima (if there is > 1)
        v_new = value_marginal[jstar]
        if v_new > v_akt:
            S = S.union({jstar})
            Sprime = Sprime - {jstar}
        else:
            break

    # Step 5
    y[list(S)] = 1
    return tuple(y), v_new


# CDLP by column generation
def CDLP_by_column_generation(capacities, preference_no_purchase):
    """
    Implements Bront et als approach for CDLP by column generation as pointed out on p. 775 just above "5. Decomp..."

    :return:
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()

    dual_pi = np.zeros(len(A))

    col_offerset, col_val = column_greedy(preference_no_purchase, dual_pi)
    if all(col_offerset == np.zeros_like(col_offerset)):
        print("MIP solution used to solve CDLP by column generation")
        col_offerset, col_val = column_MIP(preference_no_purchase, dual_pi)

    offer_sets = pd.DataFrame([np.array(col_offerset)])

    val_akt_CDLP = 0
    ret, val_new_CDLP, dual_pi, dual_sigma = CDLP(capacities, preference_no_purchase, offer_sets)

    while val_new_CDLP > val_akt_CDLP:
        val_akt_CDLP = copy.deepcopy(val_new_CDLP)  # deepcopy and new name to be on the safe side

        col_offerset, col_val = column_greedy(preference_no_purchase, dual_pi)
        if not offer_sets[(offer_sets == np.array(col_offerset)).all(axis=1)].index.empty:
            col_offerset, col_val = column_MIP(preference_no_purchase, dual_pi)
            if not offer_sets[(offer_sets == np.array(col_offerset)).all(axis=1)].index.empty:
                break  # nothing changed

        offer_sets = offer_sets.append([np.array(col_offerset)], ignore_index=True)
        ret, val_new_CDLP, dual_pi, dual_sigma = CDLP(capacities, preference_no_purchase, offer_sets)

    return ret, val_new_CDLP, dual_pi, dual_sigma


# %%
# Run CDLP as in Bront et al (CDLP by column generation, first greedy heuristig to identify entering column to the base,
# no entering column found => exact MIP procedure
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