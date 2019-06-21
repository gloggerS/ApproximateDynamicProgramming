# -*- coding: utf-8 -*-
"""
Created on Wed Mar  27 10:00:31 2019

@author: Stefan
"""

#  PACKAGES
# Data
import numpy as np
import pandas as pd

# Calculation and Counting
import itertools
import math
import copy

# Memoization
import functools

# Gurobi
from gurobipy import *
import re

# Plot
import matplotlib.pyplot as plt

# Some hacks
import sys
from contextlib import redirect_stdout

from dat_Koch import get_data
from dat_Koch import get_data_without_variations
from dat_Koch import get_variations
from dat_Koch import get_capacities_and_preferences_no_purchase
from dat_Koch import get_preference_no_purchase


# %% HELPER-FUNCTIONS
def memoize(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoizer


def get_offer_sets_all(products):
    """
    Generates all possible offer sets, starting with offering nothing.

    :param products:
    :return:
    """
    n = len(products)
    offer_sets_all = np.array(list(map(list, itertools.product([0, 1], repeat=n))))
    return offer_sets_all


# %% FUNCTIONS
@memoize
def customer_choice_individual(offer_set_tuple, preference_weights, preference_no_purchase):
    """
    For one customer of one customer segment, determine its purchase probabilities given one offer set and given
    that one customer of this segment arrived.

    Tuple needed for memoization.

    :param offer_set_tuple: tuple with offered products indicated by 1=product offered, 0=product not offered
    :param preference_weights: preference weights of one customer
    :param preference_no_purchase: no purchase preference of one customer
    :return: array of purchase probabilities ending with no purchase
    """

    if offer_set_tuple is None:
        ret = np.zeros_like(preference_weights)
        return np.append(ret, 1 - np.sum(ret))

    offer_set = np.asarray(offer_set_tuple)
    ret = preference_weights * offer_set
    ret = np.array(ret / (np.sum(ret) + preference_no_purchase))
    ret = np.append(ret, 1 - np.sum(ret))
    return ret


@memoize
def customer_choice_vector(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities):
    """
    From perspective of retailer: With which probability can he expect to sell each product (respectively non-purchase)

    :param offer_set_tuple: tuple with offered products indicated by 1=product offered
    :param preference_weights: preference weights of all customers
    :param preference_no_purchase: preference for no purchase for all customers
    :param arrival_probabilities: vector with arrival probabilities of all customer segments
    :return: array with probabilities of purchase ending with no purchase

    NOTE: probabilities don't have to sum up to one? BEACHTE: Unterschied zu (1) in Bront et all
    """
    probs = np.zeros(len(offer_set_tuple) + 1)
    for l in np.arange(len(preference_weights)):
        probs += arrival_probabilities[l] * customer_choice_individual(offer_set_tuple, preference_weights[l, :],
                                                                       preference_no_purchase[l])
    return probs


@memoize
def value_expected(capacities, t, preference_no_purchase):
    """
    Recursive implementation of the value function, i.e. dynamic program (DP) as described on p. 241.

    :param capacities:
    :param t: time to go (last chance for revenue is t=0)
    :param preference_no_purchase:
    :return: value to be expected and optimal policy
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()
    T = len(times)

    offer_sets_to_test = get_offer_sets_all(products)

    max_index = 0
    max_val = 0

    if all(capacities == 0):
        return 0, None
    if any(capacities < 0):
        return -math.inf, None
    if t == T:
        return 0, None

    for offer_set_index in range(len(offer_sets_to_test)):
        offer_set = offer_sets_to_test[offer_set_index, :]
        probs = customer_choice_vector(tuple(offer_set), preference_weights, preference_no_purchase,
                                       arrival_probabilities)

        val = float(value_expected(capacities, t + 1, preference_no_purchase)[0])  # ohne "float" würde ein numpy array
        #  zurückgegeben, und das später (max_val = val) direkt auf val verknüpft (call by reference)
        for j in products:  # nett, da Nichtkaufalternative danach (products+1) kommt und so also nicht betrachtet wird
            p = float(probs[j])
            if p > 0.0:
                value_delta_j = delta_value_j(j, capacities, t + 1, A, preference_no_purchase)
                val += p * (revenues[j] - value_delta_j)

        if val > max_val:
            max_index = offer_set_index
            max_val = val
    return max_val, tuple(offer_sets_to_test[max_index, :])


def delta_value_j(j, capacities, t, A, preference_no_purchase):
    """
    For one product j, what is the difference in the value function if we sell one product.
    TODO: stört mich etwas, Inidices von t, eng verbandelt mit value_expected()

    :param j:
    :param capacities:
    :param t:
    :param preference_no_purchase:
    :return:
    """
    return value_expected(capacities, t, preference_no_purchase)[0] - \
        value_expected(capacities - A[:, j], t, preference_no_purchase)[0]


# %%
# FUNCTIONS for Bront et al
# helpers
def purchase_rate_vector(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities):
    """
    P_j(S) for all j, P_0(S) at the end

    :param offer_set_tuple: S
    :param preference_weights
    :param preference_no_purchase
    :param arrival_probabilities
    :return: P_j(S) for all j, P_0(S) at the end

    TODO: p wird hier normiert, also gegeben ein Customer kommt, zu welchem Segment gehört
    s. p. 772 in Bront et al.
    vgl. customer_choice_vector()
    Lsg. wird hier mit \lambda lam wieder ausgebügelt in CDLP()
    """
    probs = np.zeros(len(offer_set_tuple) + 1)
    p = arrival_probabilities/(sum(arrival_probabilities))
    for l in np.arange(len(preference_weights)):
        probs += p[l] * customer_choice_individual(offer_set_tuple, preference_weights[l, :],
                                                   preference_no_purchase[l])
    return probs


def revenue(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities, revenues):
    """
    R(S)

    :param offer_set_tuple: S
    :return: R(S)
    """
    return sum(revenues * purchase_rate_vector(offer_set_tuple, preference_weights, preference_no_purchase,
                                               arrival_probabilities)[:-1])


def quantity_i(offer_set_tuple, preference_weights, preference_no_purchase, arrival_probabilities, i, A):
    """
    Q_i(S)

    :param offer_set_tuple: S
    :param i: resource i
    :return: Q_i(S)
    """
    return sum(A[i, :] * purchase_rate_vector(offer_set_tuple, preference_weights, preference_no_purchase,
                                              arrival_probabilities)[:-1])


# %%
# FUNCTIONS for Bront et al
# functions
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
# DECOMPOSITION APPROXIMATION ALGORITHM

# leg level decomposition directly via (11)
@memoize
def value_leg_i_11(i, x_i, t, pi, preference_no_purchase):
    """
    Implements the table of value leg decomposition on p. 776

    :param i:
    :param x_i:
    :param t:
    :param pi:
    :return: optimal value, index of optimal offer set, tuple optimal offer set
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()
    T = len(times)
    lam = sum(arrival_probabilities)

    if t == T+1:
        return 0, None, None, None
    elif x_i <= 0:
        return 0, None, None, None

    offer_sets_all = get_offer_sets_all(products)
    offer_sets_all = pd.DataFrame(offer_sets_all)

    val_akt = 0.0
    index_max = 0

    for index, offer_array in offer_sets_all.iterrows():
        temp = np.zeros_like(products, dtype=float)
        for j in products:
            if offer_array[j] > 0:
                temp[j] = (revenues[j] -
                           (value_leg_i_11(i, x_i, t+1, pi, preference_no_purchase)[0] -
                            value_leg_i_11(i, x_i-1, t+1, pi, preference_no_purchase)[0] - pi[i]) * A[i, j] -
                           sum(pi[A[:, j] == 1]))
        val_new = sum(purchase_rate_vector(tuple(offer_array), preference_weights,
                                           preference_no_purchase, arrival_probabilities)[:-1] * temp)
        if val_new > val_akt:
            index_max = copy.copy(index)
            val_akt = copy.deepcopy(val_new)

    return lam * val_akt + value_leg_i_11(i, x_i, t+1, pi, preference_no_purchase)[0], \
        index_max, tuple(offer_sets_all.iloc[index_max])

def displacement_costs_vector(capacities_remaining, preference_no_purchase, t, pi, beta=1):
    """
    Implements the displacement vector on p. 777

    :param capacities_remaining:
    :param t:
    :param pi:
    :param beta:
    :return:
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()

    delta_v = 1.0*np.zeros_like(resources)
    for i in resources:
        delta_v[i] = beta * (value_leg_i_11(i, capacities_remaining[i], t, pi, preference_no_purchase)[0] -
                             value_leg_i_11(i, capacities_remaining[i] - 1, t, pi, preference_no_purchase)[0]) + \
                     (1-beta) * pi[i]
    return delta_v


def calculate_offer_set(capacities_remaining, preference_no_purchase, t, pi, beta=1, dataName=""):
    """
    Implements (14) on p. 777

    :param capacities_remaining:
    :param t:
    :param pi:
    :param beta:
    :return: index of optimal offer set, optimal offer set (the products)
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations(dataName)
    lam = sum(arrival_probabilities)

    val_akt = 0
    index_max = 0

    offer_sets_all = get_offer_sets_all(products)
    offer_sets_all = pd.DataFrame(offer_sets_all)

    displacement_costs = displacement_costs_vector(capacities_remaining, preference_no_purchase, t + 2, pi, beta)

    for index, offer_array in offer_sets_all.iterrows():
        val_new = 0
        purchase_rate = purchase_rate_vector(tuple(offer_array), preference_weights,
                                             preference_no_purchase, arrival_probabilities)
        for j in products:
            if offer_array[j] > 0 and all(capacities_remaining - A[:, j] >= 0):
                val_new += purchase_rate[j] * \
                           (revenues[j] - sum(displacement_costs*A[:, j]))
        val_new = lam*val_new

        if val_new > val_akt:
            index_max = copy.copy(index)
            val_akt = copy.deepcopy(val_new)

    return index_max, products[np.array(offer_sets_all.iloc[[index_max]] == 1)[0]] + 1, offer_sets_all

#%%
# Talluri and van Ryzin
def efficient_sets(data_sets):
    """
    Calculates the efficient sets by marginal revenue ratio as discussed in Talluri and van Ryzin p. 21 upper left

    :param data_sets: A dataset with the quantities and revenues of all sets. Index has the sets.
    :return:
    """
    ES = ['0']
    data_sets = data_sets.sort_values(['q'])
    sets_quantities = data_sets.loc[:, 'q']
    sets_revenues = data_sets.loc[:, 'r']

    while True:
        print(ES)
        q_max = max(sets_quantities[ES])
        r_max = max(sets_revenues[ES])
        tocheck = set(data_sets.index[(sets_quantities >= q_max) & (sets_revenues >= r_max)])
        tocheck = tocheck - set(ES)
        if len(tocheck) == 0:
            return ES
        marg_revenues = pd.DataFrame(data=np.zeros(len(tocheck)), index=tocheck)
        for i in tocheck:
            marg_revenues.loc[i] = (sets_revenues[i] - r_max) / (sets_quantities[i] - q_max)
        ES.append(max(marg_revenues.idxmax()))


#%%
def exclude_index(narray: np.array , index_to_exclude):
    return narray[np.arange(len(narray)) != index_to_exclude]

# DPD
def DPD(capacities, preference_no_purchase, dualPrice, t=0):
    """
    Implements Bront et als approach for DPD (12) on p. 776

    :return:
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()

    val = 0.0
    for i in resources:
        val += value_leg_i_11(i, capacities[i], t, dualPrice, preference_no_purchase)[0]
        temp = dualPrice*capacities
        val += sum(exclude_index(temp, i))

    val = val/len(resources)

    return val


# %% Approximate Policy Iteration


# %%
# Koch - Approximate Policy Iteration
#
# var_capacities, var_no_purchase_preferences = get_variations()
#
# num_rows = len(var_capacities)*len(var_no_purchase_preferences)
# df = pd.DataFrame(index=np.arange(num_rows), columns=['c', 'u', 'DP', 'CDLP'])
# indexi = 0
# for capacity in var_capacities:
#     for preference_no_purchase in var_no_purchase_preferences:
#         print(capacity)
#         print(preference_no_purchase)
#
#         df.loc[indexi] = [capacity, preference_no_purchase, value_expected(capacities=capacity, t=0,
#                                                                            preference_no_purchase=preference_no_purchase),
#                           CDLP_by_column_generation(capacities=capacity, preference_no_purchase=preference_no_purchase)]
#         indexi += 1
#
# df.to_pickle("table1_DP.pkl")

#%%
# # df2 = pd.read_pickle("table1_DP.pkl")
# #
# # # %%
# # customer_choice_individual(offer_set_tuple=tuple(np.array([0, 0, 0, 1])),
# #                            preference_weights=np.array([0.4, 0.8, 1.2, 1.6]),
# #                            preference_no_purchase=np.array([1]))
# #
# # # %%
# # # CHECKS for correctness
# # # example0
# # # CDLP with all possible offerset
# #
# # capacities, preference_no_purchase = get_capacities_and_preferences_no_purchase()
# #
# # with open('res_CDLP.txt', 'w') as f:
# #     with redirect_stdout(f):
# #         CDLP(capacities, preference_no_purchase, get_offer_sets_all(products))
# #
# # #%%
# # # "example for Greedy Heuristic"
# # # column_MIP
# #
# # capacities, preference_no_purchase = get_capacities_and_preferences_no_purchase()
# #
# # with open('res_MIP.txt', 'w') as f:
# #     with redirect_stdout(f):
# #         column_MIP(preference_no_purchase, pi, w)
# #
# # # column_greedy
# # with open('res_GreedyHeuristic.txt', 'w') as f:
# #     with redirect_stdout(f):
# #         print(column_greedy(preference_no_purchase, pi, w))
# #
# #
# #
#%%
example = "threeParallelFlights"
# CDLP by column generation and DPD
var_capacities, var_no_purchase_preferences = get_variations()

num_rows = len(var_capacities)*len(var_no_purchase_preferences)

df = pd.DataFrame(index=np.arange(num_rows), columns=['c', 'u', 'CDLP', 'DPD'])
indexi = 0
for capacities in var_capacities:
    for preference_no_purchase in var_no_purchase_preferences:
        print(capacities)
        print(preference_no_purchase)

        temp = CDLP_by_column_generation(capacities=np.round(capacities), preference_no_purchase=preference_no_purchase)
        dualPrice = temp[2]
        df.loc[indexi] = [np.round(capacities), preference_no_purchase, temp[1], DPD(capacities, preference_no_purchase, dualPrice)]
        indexi += 1

# df.to_pickle("table3_CDLP_DPD.pkl")


# %%
# # # example 0
# # # towards Table 3 in Miranda and Bront
# # capacities, preference_no_purchase = get_capacities_and_preferences_no_purchase()
# # ret, val_new_CDLP, dual_pi, dual_sigma = CDLP_by_column_generation(capacities, preference_no_purchase)
# # print(dual_pi)
#
# # %%
# remaining_capacity = np.array([[2,2,1],
#                                [2,1,2],
#                                [1,2,2],
#                                [2,1,1],
#                                [1,2,1],
#                                [1,1,2],
#                                [2,1,0],
#                                [2,0,1],
#                                [1,2,0],
#                                [0,2,1],
#                                [1,0,2],
#                                [0,1,2],
#                                [1,1,1],
#                                [1,1,0],
#                                [1,0,1],
#                                [0,1,1],
#                                [1,0,0],
#                                [0,1,0],
#                                [0,0,1]])
# preference_no_purchase = get_preference_no_purchase()
# df = pd.DataFrame(index=np.arange(len(remaining_capacity)), columns=['rem cap', 'offer set'])
# for indexi in np.arange(len(df)):
#     df.loc[indexi] = [remaining_capacity[indexi], calculate_offer_set(remaining_capacity[indexi],
#                                                                       preference_no_purchase, 27,
#                                                                       np.array([0, 1134.55, 500]))[1]]
# print(df)
# df.to_pickle("table3_DPD-offerset_bront-1.pkl")
#
# remaining_capacity2 = np.array([[3, 2, 2],
#                                [2, 3, 2],
#                                [2, 2, 3],
#                                [3, 2, 1],
#                                [3, 1, 2],
#                                [2, 3, 1],
#                                [1, 3, 2],
#                                [2, 1, 3],
#                                [1, 2, 3],
#                                [3, 1, 1],
#                                [1, 3, 1],
#                                [1, 1, 3],
#                                [3, 1, 0],
#                                [3, 0, 1],
#                                [1, 3, 0],
#                                [0, 3, 1],
#                                [1, 0, 3],
#                                [0, 1, 3],
#                                [2, 2, 2]])
# preference_no_purchase = get_preference_no_purchase()
# df2 = pd.DataFrame(index=np.arange(len(remaining_capacity2)), columns=['rem cap', 'offer set'])
# for indexi in np.arange(len(df2)):
#     df2.loc[indexi] = [remaining_capacity2[indexi], calculate_offer_set(remaining_capacity2[indexi],
#                                                                       preference_no_purchase, 27,
#                                                                       np.array([0, 1134.55, 500]))[1]]
# print(df2)
# df2.to_pickle("table3_DPD-offerset_bront-2.pkl")

#%%
#
#
#
# #%%
# preference_no_purchase = get_preference_no_purchase()
# i, df = calculate_offer_set(np.array([1,0,1]), preference_no_purchase, 27, np.array([0, 1134.55, 500]))
# max_val = max(df.iloc[:, -1])
# indices = df.iloc[:, -1] > 0.9*max_val
# sum(indices)
# # df[indices]
# # df[(df.iloc[:, :-1] == np.array([0, 1, 1, 1, 0, 0, 0, 0])).apply(all, axis=1)]
# # df[(df.iloc[:, :-1] == np.array([0, 1, 1, 1, 0, 1, 0, 0])).apply(all, axis=1)]
#
# #%%
# # preference_no_purchase = get_preference_no_purchase()
# # t = 30
# # print(value_leg_i_11(0, 1, t, np.zeros(3), preference_no_purchase))
# # print(value_leg_i_11(1, 1, t, np.zeros(3), preference_no_purchase))
# # print(value_leg_i_11(2, 1, t, np.zeros(3), preference_no_purchase))
# #
# # of = tuple([0, 1, 1, 0, 0, 1, 0, 0])
# # purchase_rate_vector(of, preference_weights, preference_no_purchase, arrival_probabilities)
# # of = tuple([0, 1, 1, 0, 0, 0, 0, 0])
# # purchase_rate_vector(of, preference_weights, preference_no_purchase, arrival_probabilities)
#
#
# # #%%
# # i, df = calculate_offer_set(capacities, preference_no_purchase, 0, np.zeros_like(resources))
# # (value_leg_i_11(0, capacities[0], 0, np.zeros_like(resources))[0] + \
# # value_leg_i_11(1, capacities[1], 0, np.zeros_like(resources))[0] + \
# # value_leg_i_11(2, capacities[2], 0, np.zeros_like(resources))[0])/3
# #
# # # value_expected(capacities, 0, preference_no_purchase)
