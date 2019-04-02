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

# Memoization
import functools

# Gurobi
from gurobipy import *
import re

from dat_Koch import get_data
from dat_Koch import get_data_for_table1
from dat_Koch import get_variations

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
    Generates all possible offer sets.

    :param products:
    :return:
    """
    n = len(products)
    offer_sets_all = np.array(list(map(list, itertools.product([0, 1], repeat=n))))
    offer_sets_all = offer_sets_all[1:]  # always at least one product to offer
    return offer_sets_all


# %% FUNCTIONS
@memoize
def customer_choice_individual(offer_set_tuple, preference_weights, preference_no_purchase):
    """
    For one customer of one customer segment, determine its purchase probabilities given one offer set.

    Tuple needed for memoization.

    :param offer_set_tuple: tuple with offered products indicated by 1=product offered
    :param preference_weights: preference weights of one customer
    :param preference_no_purchase: no purchase preference of one customer
    :return: array of purchase probabilities ending with no purchase
    """

    if offer_set_tuple is None:
        ret = np.zeros_like(preference_weights)
        return np.append(ret, 1 - np.sum(ret))

    offer_set = np.asarray(offer_set_tuple)
    ret = preference_weights * offer_set
    ret = np.array(ret / (preference_no_purchase + np.sum(ret)))
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
    TODO probabilities don't have to sum up to one? BEACHTE: Unterschied zu (1) in Bront et all
    """
    probs = np.zeros(len(offer_set_tuple) + 1)
    for l in np.arange(len(preference_weights)):
        probs += arrival_probabilities[l] * customer_choice_individual(offer_set_tuple, preference_weights[l, :],
                                                                       preference_no_purchase[l])
    return probs


def delta_value_j(j, capacities, t, A, preference_no_purchase):
    """
    For one product j, what is the difference in the value function if we sell one product.

    :param j:
    :param capacities:
    :param t:
    :param preference_no_purchase:
    :return:
    """
    return value_expected(capacities, t, preference_no_purchase)[0] - \
        value_expected(capacities - A[:, j], t, preference_no_purchase)[0]


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
        times = get_data_for_table1()
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
        for j in products:  # nett, da Nichtkaufalternative danach kommt und nicht betrachtet wird
            p = float(probs[j])
            if p > 0.0:
                value_delta_j = delta_value_j(j, capacities, t + 1, A, preference_no_purchase)
                val += p * (revenues[j] - value_delta_j)

        if val > max_val:
            max_index = offer_set_index
            max_val = val
    return max_val, tuple(offer_sets_to_test[max_index, :])

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
# FUNCTIONS for Bront et al
# functions
def CDLP(offer_sets: np.ndarray):
    """
    Implements (4) of Bront et al. Needs the offer-sets to look at (N) as input.

    :param offer_sets: N
    :return: dictionary of (offer set, time offered),
    """
    resources, capacities, \
        products, revenues, A, \
        customer_segments, preference_weights, preference_no_purchase, arrival_probabilities, \
        times = get_data()

    offer_sets = pd.DataFrame(offer_sets)

    S = {}
    R = {}
    Q = {}
    for index, offer_array in offer_sets.iterrows():
        S[index] = tuple(offer_array)
        R[index] = revenue(tuple(offer_array), preference_weights, preference_no_purchase, arrival_probabilities)
        temp = {}
        for i in resources:
            temp[i] = quantity_i(tuple(offer_array), preference_weights, preference_no_purchase,
                                 arrival_probabilities, i)
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

        dualPi = np.zeros_like(resources)
        for i in resources:
            dualPi[i] = mc[i].pi
        dualSigma = msigma.pi

        valOpt = m.objVal

        return ret, valOpt, dualPi, dualSigma

    except GurobiError:
        print('Error reported')

# %%
var_capacities, var_no_purchase_preferences = get_variations()

num_rows = len(var_capacities)*len(var_no_purchase_preferences)
df = pd.DataFrame(index=np.arange(num_rows), columns=['c', 'u', 'DP'])
indexi = 0
for capacity in var_capacities:
    for preference_no_purchase in var_no_purchase_preferences:
        print(capacity)
        print(preference_no_purchase)

        df.loc[indexi] = [capacity, preference_no_purchase, value_expected(capacities=capacity, t=0,
                                                                           preference_no_purchase=preference_no_purchase)]
        indexi += 1

df.to_pickle("table1_DP.pkl")

#%%
df2 = pd.read_pickle("table1_DP.pkl")

# %%
customer_choice_individual(offer_set_tuple=tuple(np.array([0, 0, 0, 1])),
                           preference_weights=np.array([0.4, 0.8, 1.2, 1.6]),
                           preference_no_purchase=np.array([1]))

# %%
# CHECKS for correctness
# CDLP with example 0
t = get_offer_sets_all(products)
CDLP(get_offer_sets_all(products))
