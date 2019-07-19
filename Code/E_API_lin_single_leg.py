"""
This script will calculate the Approximate Policy Iteration with linear value function and store it in a large dataframe
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
# Setup of parameters
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("APILinearSingleLeg")
capacities = var_capacities[1]
preferences_no_purchase = var_no_purchase_preferences[0]


#%%
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
        m_pi = m.addVars(pi_indices, name="pi", ub=max(revenues), lb=0.0)  # Constraint 11

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

        # check exponential smoothing
        if exponential_smoothing:
            return (1 - 1 / k) * thetas + 1 / k * theta_new, (1 - 1 / k) * pis + 1 / k * pi_new
        else:
            return theta_new, pi_new
    except GurobiError:
        print('Error reported')

        return 0, 0


# %%
# Actual Code

pi_result = {}
theta_result = {}
value_result = {}
capacities_result = {}

# generate the random sample paths
np.random.seed(12)
random.seed(12)
customer_stream = [np.random.random(T) for _ in range(I)]
sales_stream = [np.random.random(T) for _ in range(I)]

for capacities in var_capacities:
    value_result[str(capacities)] = {}
    capacities_result[str(capacities)] = {}
    theta_result[str(capacities)] = {}
    pi_result[str(capacities)] = {}

    for preferences_no_purchase in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities.tolist()), " - and - ", preferences_no_purchase, "of", str(var_no_purchase_preferences.tolist()), "starting.")

        value_result[str(capacities)][str(preferences_no_purchase)] = {}
        capacities_result[str(capacities)][str(preferences_no_purchase)] = {}

        # store parameters over all policy iterations
        # K+1 policy iterations (starting with 0)
        # T time steps
        theta_all = np.array([[np.zeros(1)] * T] * (K + 1))
        pi_all = np.array([[np.zeros(len(resources))] * T] * (K + 1))

        # theta and pi for each time step
        # line 1
        thetas = 0
        pis = np.zeros(len(resources))

        for k in np.arange(K)+1:
            print(k, "of", K, "starting.")
            random.seed(13)  # to have the epsilon's (exploration vs exploitation) also the same for each policy iteration k

            v_samples = np.array([np.zeros(len(times))]*I)
            c_samples = np.array([np.zeros(shape=(len(times), len(capacities)))]*I)

            for i in np.arange(I):
                # line 3
                r_sample_i = np.zeros(len(times))  # will become v_sample
                c_sample_i = np.zeros(shape=(len(times), len(capacities)), dtype=int)

                # line 5
                c = copy.deepcopy(capacities)  # (starting capacity at time 0)

                for t in times:
                    # line 7  (starting capacity at time t)
                    c_sample_i[t] = c

                    # line 8-11  (adjust bid price)
                    pis[c_sample_i[t] == 0] = np.inf
                    pis[c_sample_i[t] > 0] = pi_all[k - 1][t][c_sample_i[t] > 0]

                    # line 12  (epsilon greedy strategy)
                    offer_set = determine_offer_tuple(pis, epsilon[k], revenues, A,
                                                      arrival_probabilities, preference_weights, preferences_no_purchase)

                    # line 13  (simulate sales)
                    sold = simulate_sales(offer_set, customer_stream[i][t], sales_stream[i][t],
                                          arrival_probabilities, preference_weights, preferences_no_purchase)

                    # line 14
                    try:
                        r_sample_i[t] = revenues[sold]
                        c -= A[:, sold]
                    except IndexError:
                        # no product was sold
                        pass

                # line 16-18
                v_samples[i] = np.cumsum(r_sample_i[::-1])[::-1]
                c_samples[i] = c_sample_i

            # line 20
            theta_all[k], pi_all[k] = update_parameters(v_samples, c_samples, theta_all[k-1], pi_all[k-1], k)
            
            value_result[str(capacities)][str(preferences_no_purchase)][k] = v_samples
            capacities_result[str(capacities)][str(preferences_no_purchase)][k] = c_samples

        theta_result[str(capacities)][str(preferences_no_purchase)] = theta_all
        pi_result[str(capacities)][str(preferences_no_purchase)] = pi_all


# %%
# write result of calculations
with open(newpath+"\\thetaAll.data", "wb") as filehandle:
    pickle.dump(theta_result, filehandle)

with open(newpath+"\\piAll.data", "wb") as filehandle:
    pickle.dump(pi_result, filehandle)

with open(newpath+"\\valueAll.data", "wb") as filehandle:
    pickle.dump(value_result, filehandle)

with open(newpath+"\\capacitiesAll.data", "wb") as filehandle:
    pickle.dump(capacities_result, filehandle)

with open(newpath+"\\thetaToUse.data", "wb") as filehandle:
    tmp = {}
    for capacities in var_capacities:
        tmp[str(capacities)] = {}
        for preferences_no_purchase in var_no_purchase_preferences:
            tmp[str(preferences_no_purchase)] = theta_result[str(capacities)][str(preferences_no_purchase)][K]
    pickle.dump(tmp, filehandle)

with open(newpath+"\\piResult.data", "wb") as filehandle:
    tmp = {}
    for capacities in var_capacities:
        tmp[str(capacities)] = {}
        for preferences_no_purchase in var_no_purchase_preferences:
            tmp[str(preferences_no_purchase)] = pi_result[str(capacities)][str(preferences_no_purchase)][K]
    pickle.dump(tmp, filehandle)


# %%
wrapup(logfile, time_start, newpath)
