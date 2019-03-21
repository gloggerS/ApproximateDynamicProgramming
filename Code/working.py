from simulationBasic import *

import numpy as np
from copy import deepcopy
import pandas as pd

from gurobipy import *
import re

# %% working on Bront CDLP -
# Ziel: Revenue results for parallel flights example reproduzieren (Table A.1, Spalte CDLP)

# j für Produkte
# i für Ressourcen


def products():
    """
    Products are indexed from 1 to numProducts

    :return:
    """
    return np.arange(numProducts)+1


def resources():
    """
    Resources are indexed from 0 to len(A)-1

    :return:
    """
    return np.arange(len(A))


def offer_sets():
    """
    All possible offer sets

    :return:
    """
    offer_sets_all = pd.DataFrame(list(map(list, itertools.product([0, 1], repeat=numProducts))))
    offer_sets_all = offer_sets_all[1:]  # always at least one product to offer
    return offer_sets_all


def customer_segments():
    """
    Customer segments are indexed from 0 to L-1

    :return:
    """
    return np.arange(len(preference_no_purchase))

@memoize
def purchase_rate(offer_set_tuple, j):
    """
    P_j(S)

    :param offer_set_tuple: S
    :param j: product j
    :return: P_j(S)
    """
    return customer_choice_vector(offer_set_tuple)[j]


@memoize
def revenue(offer_set_tuple):
    """
    R(S)

    :param offer_set_tuple: S
    :return: R(S)
    """
    return sum(revenues * customer_choice_vector(offer_set_tuple)[1:])


@memoize
def quantity_i(offer_set_tuple, i):
    """
    Q_i(S)

    :param offer_set_tuple: S
    :param i: resource i
    :return: Q_i(S)
    """
    return sum(A[i, :] * customer_choice_vector(offer_set_tuple)[1:])


@memoize
def quantity_vector(offer_set_tuple):
    """
    Q(S)

    :param offer_set_tuple: S
    :return: Q(S)
    """
    ret = np.zeros(len(A))
    for i in resources():
        ret[i] = quantity_i(offer_set_tuple, i)
    return ret


@memoize
def revenue_i(offer_set_tuple, pi, i):
    """
    Revenue rate when offering set S (compare "Solution to the One-Dimensional DP Approximation")
    R^i(S)

    :param offer_set_tuple:
    :param pi:
    :param i: product i
    :return: R^i(S)
    """
    ret = 0
    for j in products():
        ret += purchase_rate(offer_set_tuple, j) * (revenues[j-1] + pi[i] * A[i, j-1] -
                                                    sum(pi[A[:, j-1] == 1 & (np.arange(len(pi)) != i)]))
    return ret


def CDLP():
    """
    Implements (3) of Bront et al

    :return: dictionary of (offer set S, time offered)
    """
    offer_sets_all = offer_sets()

    S = {}
    R = {}
    Q = {}
    for index, offer_array in offer_sets_all.iterrows():
        S[index] = tuple(offer_array)
        R[index] = revenue(tuple(offer_array))
        temp = {}
        for i in resources():
            temp[i] = quantity_i(tuple(offer_array), i)
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
            mc[i] = m.addConstr(lam * quicksum(Q[s][i]*mt[s] for s in offer_sets_all.index.values), GRB.LESS_EQUAL,
                                capacities[i], name="constraintOnResource")
        msigma = m.addConstr(quicksum(mt[s] for s in offer_sets_all.index.values), GRB.LESS_EQUAL, T)

        m.optimize()

        ret = {}
        pat = r".*?\[(.*)\].*"
        for v in m.getVars():
            if v.X > 0:
                match = re.search(pat, v.VarName)
                erg_index = match.group(1)
                ret[int(erg_index)] = (tuple(offer_sets_all.loc[int(erg_index)]), v.X)
                print(tuple(offer_sets_all.loc[int(erg_index)]), ": ", v.X)

        dualPi = np.zeros_like(resources())
        for i in resources():
            dualPi[i] = mc[i].pi
        dualSigma = msigma.pi

        valOpt = m.objVal

        return ret, valOpt, dualPi, dualSigma

    except GurobiError:
        print('Error reported')


# %%
def CDLP_reduced(offer_sets):
    """
    Implements (4) of Bront et al. Needs the offer-sets to look at (N) as input.

    :param offer_sets: N
    :return: dictionary of (offer set, time offered),
    """
    S = {}
    R = {}
    Q = {}
    for index, offer_array in offer_sets.iterrows():
        S[index] = tuple(offer_array)
        R[index] = revenue(tuple(offer_array))
        temp = {}
        for i in np.arange(len(A)):
            temp[i] = quantity_i(tuple(offer_array), i)
        Q[index] = temp

    try:
        m = Model()

        # Variables
        mt = m.addVars(offer_sets.index.values, name="t", lb=0.0)  # last constraint

        # Objective Function
        m.setObjective(lam * quicksum(R[s] * mt[s] for s in offer_sets.index.values), GRB.MAXIMIZE)

        mc = {}
        # Constraints
        for i in np.arange(len(A)):
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

        dualPi = np.zeros_like(resources())
        for i in resources():
            dualPi[i] = mc[i].pi
        dualSigma = msigma.pi

        valOpt = m.objVal

        return ret, valOpt, dualPi, dualSigma

    except GurobiError:
        print('Error reported')


#%%
# CDLP by column generation
def CDLP_by_column_generation():
    """
    Implements Bront et als approach for CDLP by column generation as pointed out on p. 775 just above "5. Decomp..."

    :return:
    """
    pi = np.zeros(len(A))

    offer_sets = column_greedy(pi)
    if all(offer_sets == np.zeros_like(offer_sets)):
        print("MIP solution used to solve CDLP by column generation")
        offer_sets = column_MIP(pi)

    offer_sets = pd.DataFrame([np.array(offer_sets)])

    val_akt = 0
    ret, val_new, dualPi, dualSigma = CDLP_reduced(offer_sets)

    while val_new > val_akt:
        val_akt = val_new

        offer_set_new = column_greedy(dualPi)
        if not offer_sets[(offer_sets == np.array(offer_set_new)).all(axis=1)].index.empty:
            offer_set_new = column_MIP(dualPi)
            if not offer_sets[(offer_sets == np.array(offer_set_new)).all(axis=1)].index.empty:
                break  # nothing changed

        offer_sets = offer_sets.append([np.array(offer_set_new)], ignore_index=True)
        ret, val_new, dualPi, dualSigma = CDLP_reduced(offer_sets)

    return ret, val_new, dualPi, dualSigma


def column_MIP(pi, w=0):  # pass w to test example for greedy heuristic
    """
    Implements MIP formulation on p. 775 lhs

    :param pi:
    :param w:
    :return: optimal tuple of products to offer
    """
    K = 1/min(preference_no_purchase.min(), np.min(preference_weights[np.nonzero(preference_weights)]))+1

    if isinstance(w, int) and w == 0:  # and is lazy version of &
        w = np.zeros_like(revenues, dtype=float)
        for j in products():
            w[j-1] = revenues[j-1] - sum(A[:, j-1]*pi)

    try:
        m = Model()

        mx = {}
        my = {}
        mz = {}

        # Variables
        for j in products():
            my[j] = m.addVar(0, 1, vtype=GRB.BINARY, name="y["+str(j)+"]")
        for l in customer_segments():
            mx[l] = m.addVar(0.0, name="x["+str(l)+"]")
            temp = {}
            for j in products():
                temp[j] = m.addVar(0.0, name="z["+str(l)+","+str(j)+"]")
            mz[l] = temp

        # Objective TODO einfügen: arrival_probability[l]*
        m.setObjective(quicksum(w[j-1]*preference_weights[l, j-1]*mz[l][j]
                                for l in customer_segments() for j in products()), GRB.MAXIMIZE)

        # Constraints
        mc1 = m.addConstrs((mx[l]*preference_no_purchase[l] +
                            quicksum(preference_weights[l, j-1]*mz[l][j] for j in products()) == 1
                            for l in customer_segments()), name="mc1")
        mc2 = m.addConstrs((mx[l] - mz[l][j] <= K - K*my[j] for l in customer_segments() for j in products()),
                           name="mc2")
        mc3 = m.addConstrs((mz[l][j] <= mx[l] for l in customer_segments() for j in products()), name="mc3")
        mc4 = m.addConstrs((mz[l][j] <= K*my[j] for l in customer_segments() for j in products()), name="mc4")

        m.optimize()

        y = np.zeros_like(revenues)
        for j in products():
            y[j-1] = my[j].x

        return tuple(y)

    except GurobiError:
        print('Error reported')


def column_greedy(pi, w=0):  # pass w to test example for greedy heuristic
    """
    Implements Greedy Heuristic on p. 775 rhs

    :param pi:
    :param w:
    :return: heuristically optimal tuple of products to offer
    """
    # Step 1
    y = np.zeros_like(revenues)

    if isinstance(w, int) and w == 0:  # and is lazy version of &
        w = np.zeros_like(revenues, dtype=float)  # calculate the opportunity costs
        for j in products():
            w[j-1] = revenues[j-1] - sum(A[:, j-1]*pi)

    # Step 2
    Sprime = set(np.where(w > 0)[0])

    # Step 3
    value_marginal = np.zeros_like(w, dtype=float)
    for j in Sprime:
        for l in customer_segments():
            value_marginal[j-1] += preference_weights[l, j-1]/(preference_weights[l, j-1] + preference_no_purchase[l])
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
            for l in customer_segments():
                z = 0
                n = 0
                for i in S.union({j}):
                    z += w[i]*preference_weights[l, i]
                    n += preference_weights[l, i] + preference_no_purchase[l]
                v_temp[j] += arrival_probability[l]*z/n
        jstar = np.argmax(value_marginal)  # argmax always returns index of first maxima (if there is > 1)
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
# leg level decomposition
@memoize
def value_leg(i, x_i, t, pi):
    """
    Implements the table of value leg decomposition on p. 776

    :param i:
    :param x_i:
    :param t:
    :param pi:
    :return:
    """
    if t == T+1:
        return 0
    elif x_i == 0:
        return 0

    offer_sets_all = offer_sets()

    val_akt = 0
    index_max = 0

    for index, offer_array in offer_sets_all.iterrows():
        val_new = (revenue_i(tuple(offer_array), pi, i) -
                   quantity_i(tuple(offer_array), i) * (value_leg(i, x_i, t+1, pi) - value_leg(i, x_i-1, t+1, pi)))
        val_new = val_new
        if val_new > val_akt:
            index_max = index
            val_akt = val_new

    return lam*val_akt + value_leg(i, x_i, t+1, pi)


def displacement_costs_vector(capacities_remaining, t, pi, beta=1):
    """
    Implements the displacement vector on p. 777

    :param capacities_remaining:
    :param t:
    :param pi:
    :param beta:
    :return:
    """
    delta_v = np.zeros_like(resources())
    for i in resources():
        delta_v[i] = beta*(value_leg(i, capacities_remaining[i], t+1, pi) -
                           value_leg(i, capacities_remaining[i]-1, t+1, pi)) + \
                     (1-beta)*pi[i]
    return delta_v


def calculate_offer_set(capacities_remaining, t, pi, beta=1):
    """
    Implements (14) on p. 777

    :param capacities_remaining:
    :param t:
    :param pi:
    :param beta:
    :return:
    """
    val_akt = 0
    index_max = 0

    offer_sets_all = offer_sets()

    for index, offer_array in offer_sets_all.iterrows():
        val_new = 0
        for j in products():
            if offer_array[j-1] > 0 and all(capacities_remaining - A[:, j-1] >= 0):
                print("yea")
                print(purchase_rate(tuple(offer_array), j) )
                print(revenues[j-1] - sum(displacement_costs_vector(capacities_remaining, t, pi, beta=1)*A[:, j-1]))
                val_new += purchase_rate(tuple(offer_array), j) * \
                           (revenues[j-1] - sum(displacement_costs_vector(capacities_remaining, t, pi, beta=1)*A[:, j-1]))
        val_new = lam*val_new
        # print(index, val_new)

        if val_new > val_akt:
            index_max = index
            val_akt = val_new

    return tuple(offer_sets_all[index_max:])




#%%
# CDLP testen mit example 0 (compare results with values on page 774)
ret, val, dualPi, dualSigma = CDLP()

#%%
# todo
ret, val, dualPi, dualSigma = CDLP_by_column_generation()

#%%
# todo
dualPi = np.array([0, 1, 134.55])
capacities_remaining = np.array([0, 0, 1])
t = calculate_offer_set(capacities_remaining, 27, dualPi, beta=1)

