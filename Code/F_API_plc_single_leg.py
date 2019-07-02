"""
This script will calculate the Approximate Policy Iteration with plc value function and store it in a large dataframe
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
import copy

import random

#%%

logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K \
    = setup_testing("APIPLCSingleLeg")
capacities = var_capacities[1]
preferences_no_purchase = var_no_purchase_preferences[0]
I = 100

chunk_size = 4
capacities_thresholds = np.array([[i*chunk_size for i in range((capacities[0] + chunk_size) // chunk_size)]])

#%%
def determine_offer_tuple(pi, eps):
    """
    OLD Implementation
    Determines the offerset given the bid prices for each resource.

    Implement the Greedy Heuristic from Bront et al: A Column Generation Algorithm ... 4.2.2
    and extend it for the epsilon greedy strategy

    :param pi:
    :return:
    """

    # setup
    offer_tuple = np.zeros_like(revenues)

    # epsilon greedy strategy - offer no products
    eps_prob = random.random()
    if eps_prob < eps/2:
        return tuple(offer_tuple)

    # line 1
    s_prime = revenues - np.apply_along_axis(sum, 1, A.T * pi) > 0
    if all(np.invert(s_prime)):
        return tuple(offer_tuple)

    # epsilon greedy strategy - offer all products
    if eps_prob < eps:
        return tuple(offer_tuple)

    # epsilon greedy strategy - greedy heuristic

    # line 2-3
    # offer_sets_to_test has in each row an offer set, we want to test
    offer_sets_to_test = np.zeros((sum(s_prime), len(revenues)))
    offer_sets_to_test[np.arange(sum(s_prime)), np.where(s_prime)] = 1
    offer_sets_to_test += offer_tuple
    offer_sets_to_test = (offer_sets_to_test > 0)

    value_marginal = np.apply_along_axis(calc_value_marginal, axis=1, arr=offer_sets_to_test)

    offer_tuple[np.argmax(value_marginal)] = 1
    s_prime = s_prime & offer_tuple == 0
    v_s = np.amax(value_marginal)

    # line 4
    while True:
        # 4a
        # offer_sets_to_test has in each row an offer set, we want to test
        offer_sets_to_test = np.zeros((sum(s_prime), len(revenues)))
        offer_sets_to_test[np.arange(sum(s_prime)), np.where(s_prime)] = 1
        offer_sets_to_test += offer_tuple
        offer_sets_to_test = (offer_sets_to_test > 0)

        # 4b
        value_marginal = np.apply_along_axis(calc_value_marginal, axis=1, arr=offer_sets_to_test)

        if np.amax(value_marginal) > v_s:
            v_s = np.amax(value_marginal)
            offer_tuple[np.argmax(value_marginal)] = 1
            s_prime = s_prime & offer_tuple == 0
            if all(offer_tuple == 1):
                break
        else:
            break
    return tuple(offer_tuple)


def calc_value_marginal(indices_inner_sum):
    v_temp = 0
    for l in np.arange(len(preference_weights)):
        v_temp += arrival_probabilities[l] * \
                  sum(indices_inner_sum * (revenues - np.apply_along_axis(sum, 1, A.T * pis)) *
                      preference_weights[l, :]) / \
                  (sum(indices_inner_sum * preference_weights[l, :]) + var_no_purchase_preferences[l])
    return v_temp


def update_parameters(v_samples, c_samples, thetas, pis, k):
    """
    Updates the parameter theta, pi given the sample path using least squares linear regression with constraints.
    Using gurobipy

    :param v_samples:
    :param c_samples:
    :return:
    """
    set_i = np.arange(len(v_samples))
    set_t = np.arange(len(thetas))
    set_h = np.arange(len(pis[0]))

    theta_multidict = {}
    for t in set_t:
        theta_multidict[t] = thetas[t]
    theta_indices, theta_tuples = multidict(theta_multidict)

    pi_multidict = {}
    for t in set_t:
        for h in set_h:
            pi_multidict[t, h] = pis[t, h]
    pi_indices, pi_tuples = multidict(pi_multidict)

    try:
        m = Model()

        # Variables
        m_theta = m.addVars(theta_indices, name="theta", lb=0.0)  # Constraint 10
        m_pi = m.addVars(pi_indices, name="pi", ub=max(revenues))  # Constraint 11

        for t in set_t:
            m_theta[t].start = theta_tuples[t]
            for h in set_h:
                m_pi[t, h].start = pi_tuples[t, h]

        # Goal Function (14)
        lse = quicksum((v_samples[i][t] - m_theta[t] - quicksum(m_pi[t, h] * c_samples[i][t][h] for h in set_h)) *
                       (v_samples[i][t] - m_theta[t] - quicksum(m_pi[t, h] * c_samples[i][t][h] for h in set_h))
                       for t in set_t for i in set_i)
        m.setObjective(lse, GRB.MINIMIZE)

        # Constraints
        # C12 (not needed yet)
        for t in set_t[:-1]:
            m.addConstr(m_theta[t], GRB.GREATER_EQUAL, m_theta[t+1], name="C15")  # Constraint 15
            for h in set_h:
                m.addConstr(m_pi[t, h], GRB.GREATER_EQUAL, m_pi[t+1, h], name="C16")  # Constraint 16

        m.optimize()

        theta_new = copy.deepcopy(thetas)
        pi_new = copy.deepcopy(pis)

        for t in set_t:
            theta_new[t] = m.getVarByName("theta[" + str(t) + "]").X
            for h in set_h:
                pi_new[t, h] = m.getVarByName("pi[" + str(t) + "," + str(h) + "]").X

        # without exponential smoothing
        if exponential_smoothing:
            return (1 - 1 / k) * thetas + 1 / k * theta_new, (1 - 1 / k) * pis + 1 / k * pi_new
        else:
            return theta_new, pi_new
    except GurobiError:
        print('Error reported')

        return 0, 0


def update_parameters_plc(v_samples, c_samples, thetas, pis, k):
    """
    Updates the parameter theta, pi given the sample path using least squares linear regression with constraints.
    Using gurobipy

    :param v_samples:
    :param c_samples:
    :return:
    """
    set_i = np.arange(len(v_samples))
    set_t = np.arange(len(thetas))
    set_h = np.arange(len(pis[0]))
    set_s = np.arange(intervals_capacities_num(capacities_thresholds)-1)

    # i, t, h, s
    f = np.array([[[np.zeros(len(set_s))]*len(set_h)]*len(set_t)]*len(set_i))
    for i in set_i:
        for t in set_t:
            for h in set_h:
                # c_h <= b_h^{s-1}
                index_c_smaller_lower = c_samples[i][t][h] <= capacities_thresholds[h][:-1]
                f[i][t][h][index_c_smaller_lower] = 0
                # b_h^{s-1} < c_h <= b_h^s
                index_c_smaller = c_samples[i][t][h] <= capacities_thresholds[h][1:]
                index_c_bigger = c_samples[i][t][h] > capacities_thresholds[h][:-1]
                f[i][t][h][index_c_smaller & index_c_bigger] = c_samples[i][t][h] - capacities_thresholds[h][[*(index_c_smaller & index_c_bigger), False]]
                # b_h^s > c_h
                index_c_bigger_upper = c_samples[i][t][h] > capacities_thresholds[h][1:]
                f[i][t][h][index_c_bigger_upper] = capacities_thresholds[h][[False, *(index_c_bigger_upper)]] - capacities_thresholds[h][[*(index_c_bigger_upper), False]]

    theta_multidict = {}
    for t in set_t:
        theta_multidict[t] = thetas[t]
    theta_indices, theta_tuples = multidict(theta_multidict)

    pi_multidict = {}
    for t in set_t:
        for h in set_h:
            for s in set_s:
                pi_multidict[t, h, s] = pis[t, h, s]
    pi_indices, pi_tuples = multidict(pi_multidict)

    try:
        m = Model()

        # Variables
        m_theta = m.addVars(theta_indices, name="theta", lb=0.0)  # Constraint 10
        m_pi = m.addVars(pi_indices, name="pi", ub=max(revenues))  # Constraint 11

        for t in set_t:
            m_theta[t].start = theta_tuples[t]
            for h in set_h:
                for s in set_s:
                    m_pi[t, h, s].start = pi_tuples[t, h, s]

        # Goal Function (14)
        lse = quicksum((v_samples[i][t] - m_theta[t] - quicksum(m_pi[t, h, s] * f[i][t][h][s] for s in set_s for h in set_h)) *
                       (v_samples[i][t] - m_theta[t] - quicksum(m_pi[t, h, s] * f[i][t][h][s] for s in set_s for h in set_h))
                       for t in set_t for i in set_i)
        m.setObjective(lse, GRB.MINIMIZE)

        # Constraints
        for t in set_t[:-1]:
            m.addConstr(m_theta[t], GRB.GREATER_EQUAL, m_theta[t+1], name="C15")  # Constraint 15
            for h in set_h:
                for s in set_s[:-1]:
                    m.addConstr(m_pi[t, h, s], GRB.GREATER_EQUAL, m_pi[t, h, s+1], name="C12")  # Constraint 12
                    m.addConstr(m_pi[t, h, s], GRB.GREATER_EQUAL, m_pi[t + 1, h, s], name="C16")  # Constraint 16

        m.optimize()

        theta_new = copy.deepcopy(thetas)
        pi_new = copy.deepcopy(pis)

        for t in set_t:
            theta_new[t] = m.getVarByName("theta[" + str(t) + "]").X
            for h in set_h:
                for s in set_s:
                    pi_new[t, h, s] = m.getVarByName("pi[" + str(t) + "," + str(h) + "," + str(s) + "]").X

        # without exponential smoothing
        if exponential_smoothing:
            return (1 - 1 / k) * thetas + 1 / k * theta_new, (1 - 1 / k) * pis + 1 / k * pi_new
        else:
            return theta_new, pi_new
    except GurobiError:
        print('Error reported')

        return 0, 0


def simulate_sales(offer_set):
    customer = int(np.random.choice(np.arange(len(arrival_probabilities)+1),
                                    size=1,
                                    p=np.array([*arrival_probabilities, 1-sum(arrival_probabilities)])))
    if customer == len(arrival_probabilities):
        return len(products)  # no customer arrives => no product sold
    else:
        return int(np.random.choice(np.arange(len(products) + 1),
                                    size=1,
                                    p=customer_choice_individual(offer_set,
                                                                 preference_weights[customer],
                                                                 preferences_no_purchase[customer])))


def intervals_capacities_num(capacities_thresholds):
    return max(np.apply_along_axis(len, 1, capacities_thresholds))

# %%
# number of linear iterations
K_lin = 40
if K_lin > K:
    raise ValueError("iteration before applying plc must be smaller than total number of iterations")

# Actual Code
# K+1 policy iterations (starting with 0)
# T time steps
# --> theta
# m resources
# max(S_h) intervals
# --> pi
theta_all = np.array([[np.zeros(1)]*T]*(K+1))
pi_all = np.array([[np.zeros(len(resources))]*T] * (K_lin + 1))
pi_all_plc = np.array([[[np.zeros(intervals_capacities_num(capacities_thresholds)-1)]*len(resources)]*T] *
                      (K - K_lin + 1))

# theta and pi for each time step
# line 1
thetas = 0
pis = np.zeros(len(resources))

for k in np.arange(K)+1:
    print(k, "of", K, "starting.")
    np.random.seed(k)
    random.seed(k)

    v_samples = np.array([np.zeros(len(times))]*I)
    c_samples = np.array([np.zeros(shape=(len(times), len(capacities)))]*I)

    # initialize pi plc after the number of linear iterations went through
    if k == K_lin+1:
        tT = len(pi_all_plc[k-K_lin-1])
        hH = len(pi_all_plc[k-K_lin-1][0])
        for t in np.arange(tT):
            for h in np.arange(hH):
                pi_all_plc[k-K_lin-1][t][h] = pi_all[k-1][t][h]

    for i in np.arange(I):
        # line 3
        r_sample = np.zeros(len(times))  # will become v_sample
        c_sample = np.zeros(shape=(len(times), len(capacities)), dtype=int)

        # line 5
        c = copy.deepcopy(capacities)  # (starting capacity at time 0)

        for t in times:
            # line 7  (starting capacity at time t)
            c_sample[t] = c

            # line 8-11  (adjust bid price)
            pis[c_sample[t] == 0] = np.inf
            if k <= K_lin:
                # linear case
                pis[c_sample[t] > 0] = pi_all[k - 1][t][c_sample[t] > 0]
            else:
                # plc case (compare equation 13)
                for h in [i for i, x in enumerate(c_sample[t] > 0) if x]:
                    # find the relevant interval
                    # for which index the capacity is smaller or equal (starting with upper bound of first interval)
                    # for which index the capacity is greater (ending with lower bound of last interval)
                    index_c_smaller = c_sample[t][h] <= capacities_thresholds[h][1:]
                    index_c_bigger = c_sample[t][h] > capacities_thresholds[h][:-1]
                    pis[h] = pi_all_plc[k-K_lin][t][h][index_c_smaller & index_c_bigger]

            # line 12  (epsilon greedy strategy)
            offer_set = determine_offer_tuple(pis, epsilon[k])

            # line 13  (simulate sales)
            sold = simulate_sales(offer_set)

            # line 14
            try:
                r_sample[t] = revenues[sold]
                c -= A[:, sold]
            except IndexError:
                # no product was sold
                pass

        # line 16-18
        v_samples[i] = np.cumsum(r_sample[::-1])[::-1]
        c_samples[i] = c_sample

    # line 20
    if k <= K_lin:
        # linear case
        theta_all[k], pi_all[k] = update_parameters(v_samples, c_samples, theta_all[k-1], pi_all[k-1], k)
    else:
        # plc case
        theta_all[k], pi_all_plc[k-K_lin] = update_parameters_plc(v_samples, c_samples, theta_all[k - 1],
                                                                  pi_all_plc[k - K_lin - 1], k)

# %%
# write result of calculations
with open(newpath+"\\thetaAll.data", "wb") as filehandle:
    pickle.dump(theta_all, filehandle)

with open(newpath+"\\piAll.data", "wb") as filehandle:
    pickle.dump(pi_all, filehandle)

with open(newpath+"\\thetaResult.data", "wb") as filehandle:
    pickle.dump(theta_all[-1], filehandle)

with open(newpath+"\\piResult.data", "wb") as filehandle:
    pickle.dump(pi_all_plc[-1], filehandle)

# %%
time_elapsed = time.time() - time_start
print("\n\nTotal time needed:\n", time_elapsed, "seconds = \n", time_elapsed/60, "minutes", file=logfile)
logfile.close()
print("\n\n\n\nDone. Time elapsed:", time.time() - time_start, "seconds.")
print("Results stored in: "+newpath)
