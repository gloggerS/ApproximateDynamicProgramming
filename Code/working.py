from simulationBasic import *

import numpy as np
from copy import deepcopy
import pandas as pd

from gurobipy import *
import re

# %% working on Bront CDLP -
# Ziel: Revenue results for parallel flights example reproduzieren (Table A.1, Spalte CDLP)


def revenue_all(offer_set_tuple):
    return sum(revenues * customer_choice_all(offer_set_tuple)[1:])

def quantity_all(offer_set_tuple, i):
    return sum(A[i-1, :] * customer_choice_all(offer_set_tuple)[1:])

def CDLP():
    offer_sets_all = pd.DataFrame(list(map(list, itertools.product([0, 1], repeat=numProducts))))
    offer_sets_all = offer_sets_all[1:]  # always at least one product to offer

    S = {}
    R = {}
    Q = {}
    for index, offer_array in offer_sets_all.iterrows():
        S[index] = tuple(offer_array)
        R[index] = revenue_all(tuple(offer_array))
        temp = {}
        for i in np.arange(len(A)):
            temp[i] = quantity_all(tuple(offer_array), i)
        Q[index] = temp


    try:
        m = Model()

        # Variables
        mt = m.addVars(offer_sets_all.index.values, name="t", lb=0.0)  # last constraint

        # Objective Function
        m.setObjective(lam * quicksum(R[s]*mt[s] for s in offer_sets_all.index.values), GRB.MAXIMIZE)

        mc = {}
        # Constraints
        for i in np.arange(len(A)):
            mc[i] = m.addConstr(lam * quicksum(Q[s][i]*mt[s] for s in offer_sets_all.index.values), GRB.LESS_EQUAL, capacities[i],
                        name="constraintOnResource")
        msigma = m.addConstr(quicksum(mt[s] for s in offer_sets_all.index.values), GRB.LESS_EQUAL, T)

        m.optimize()

        ret = {}
        pat = r".*?\[(.*)\].*"
        for v in m.getVars():
            if v.X > 0:
                match = re.search(pat, v.VarName)
                erg_index = match.group(1)
                ret[int(erg_index)] = (tuple(offer_sets_all.loc[int(erg_index)]), v.X)
                print(offer_sets_all.loc[int(erg_index)], ": ", v.X)
        return ret

    except GurobiError:
        print('Error reported')

# %%

def CDLP_reduced(offer_sets_all):
    S = {}
    R = {}
    Q = {}
    for index, offer_array in offer_sets_all.iterrows():
        S[index] = tuple(offer_array)
        R[index] = revenue_all(tuple(offer_array))
        temp = {}
        for i in np.arange(len(A)):
            temp[i] = quantity_all(tuple(offer_array), i)
        Q[index] = temp

    try:
        m = Model()

        # Variables
        mt = m.addVars(offer_sets_all.index.values, name="t", lb=0.0)  # last constraint

        # Objective Function
        m.setObjective(lam * quicksum(R[s] * mt[s] for s in offer_sets_all.index.values), GRB.MAXIMIZE)

        mc = {}
        # Constraints
        for i in np.arange(len(A)):
            mc[i] = m.addConstr(lam * quicksum(Q[s][i] * mt[s] for s in offer_sets_all.index.values), GRB.LESS_EQUAL,
                                capacities[i],
                                name="constraintOnResource")
        msigma = m.addConstr(quicksum(mt[s] for s in offer_sets_all.index.values), GRB.LESS_EQUAL, T)

        m.optimize()

        ret = {}
        pat = r".*?\[(.*)\].*"
        for v in m.getVars():
            if v.X > 0:
                match = re.search(pat, v.VarName)
                erg_index = match.group(1)
                ret[int(erg_index)] = (tuple(offer_sets_all.loc[int(erg_index)]), v.X)
                print(offer_sets_all.loc[int(erg_index)], ": ", v.X)

        dualPi = np.arange(len(A))
        for i in np.arange(len(A)):
            dualPi[i] = mc[i].pi
        dualSigma = msigma.pi

        return ret, dualPi, dualSigma

    except GurobiError:
        print('Error reported')


#%%
# Column Generation

def column_MIP(pi, w = 0):  # pass w to test example for greedy heuristic
    L = len(preference_no_purchase)
    K = 1/min(preference_no_purchase.min(), np.min(preference_weights[np.nonzero(preference_weights)]))+1

    if isinstance(w, int) and w == 0:  # and is lazy version of &
        w = np.zeros_like(revenues, dtype=float)
        for j in np.arange(len(revenues)):
            w[j] = revenues[j] - sum(A[:, j]*pi)

    try:
        m = Model()

        mx = {}
        my = {}
        mz = {}

        # Variables
        for j in np.arange(numProducts):
            my[j] = m.addVar(0, 1, vtype=GRB.BINARY, name="y["+str(j)+"]")
        for l in np.arange(L):
            mx[l] = m.addVar(0.0, name="x["+str(l)+"]")
            temp = {}
            for j in np.arange(numProducts):
                temp[j] = m.addVar(0.0, name="z["+str(l)+","+str(j)+"]")
            mz[l] = temp

        # Objective TODO einfügen: arrival_probability[l]*
        m.setObjective(quicksum(w[j]*preference_weights[l, j]*mz[l][j]
                                for l in np.arange(L) for j in np.arange(numProducts)), GRB.MAXIMIZE)

        # Constraints
        mc1 = m.addConstrs((mx[l]*preference_no_purchase[l] + quicksum(preference_weights[l, i]*mz[l][i] for i in np.arange(numProducts)) == 1 for l in range(L)), name="mc1")
        mc2 = m.addConstrs((mx[l] - mz[l][i] <= K - K*my[i] for l in range(L) for i in range(numProducts)), name="mc2")
        mc3 = m.addConstrs((mz[l][i] <= mx[l] for l in range(L) for i in range(numProducts)), name="mc3")
        mc4 = m.addConstrs((mz[l][i] <= K*my[i] for l in range(L) for i in range(numProducts)), name="mc4")

        m.optimize()

        y = np.zeros_like(revenues)
        for j in np.arange(numProducts):
            y[j] = my[j].x

        return tuple(y)

    except GurobiError:
        print('Error reported')


def column_greedy(pi, w = 0):  # pass w to test example for greedy heuristic
    L = len(preference_weights)

    # Step 1
    y = np.zeros_like(revenues)

    if isinstance(w, int) and w == 0:  # and is lazy version of &
        w = np.zeros_like(revenues, dtype=float)  # calculate the opportunity costs
        for j in np.arange(len(revenues)):
            w[j] = revenues[j] - sum(A[:, j]*pi)

    # Step 2
    Sprime = set(np.where(w > 0)[0])

    # Step 3
    value_marginal = np.zeros_like(w, dtype=float)
    for j in Sprime:
        for l in np.arange(L):
            value_marginal[j] += preference_weights[l, j]/(preference_weights[l, j] + preference_no_purchase[l])
        value_marginal[j] *= w[j]
    jstar = np.argmax(value_marginal)
    v_new = value_marginal[jstar]

    S = {jstar}
    Sprime = Sprime-S

    # Step 4
    while True:
        v_akt = v_new
        v_temp = np.zeros_like(revenues, dtype=float)  # uses more space then necessary, but simplifies indices below
        for j in Sprime:
            for l in np.arange(L):
                z = 0
                n = 0
                for i in S.union({j}):
                    z += w[i]*preference_weights[l, i]
                    n += preference_weights[l, i] + preference_no_purchase[l]
                v_temp[j] += arrival_probability[l]*z/n
        jstar = np.argmax(value_marginal)
        v_new = value_marginal[jstar]
        if v_new > v_akt:
            S = S.union({jstar})
            Sprime = Sprime - {jstar}
        else:
            break

    # Step 5
    y[list(S)] = 1
    return tuple(y)


#%%
offer_sets_all = pd.DataFrame([[1, 1, 1, 1, 1, 1, 1, 1]])

w, e, r = CDLP_reduced(offer_sets_all)
