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

import pickle
import copy

import random

#%%
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
    customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
    epsilon, exponential_smoothing,\
    K, online_K, I \
    = setup_testing("APILinearSingleLeg-Evaluation")
capacities = var_capacities[0]
preferences_no_purchase = var_no_purchase_preferences[0]

with open("0-test_customer.data", "rb") as filehandle:
    test_customer = pickle.load(filehandle)
with open("0-test_sales.data", "rb") as filehandle:
    test_sales = pickle.load(filehandle)

if online_K > len(test_sales) or online_K > len(test_customer):
    raise ValueError("online_K as specified in 0_settings.csv has to be smaller then the test data given in test_sales and test_customer")

#%%
# todo get the folder in which the parameters (theta, pi) to use are stored; e.g. via sys.argv (this script is called after the calculation of those parameters)
for i in sys.argv:
    print(i)

result_folder = os.getcwd() + '\\Results\\singleLegFlight-True-APILinearSingleLeg-190702-2200'


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

    # epsilon greedy strategy - offer all products
    if eps_prob < eps:
        return tuple(offer_tuple)

    # line 1
    s_prime = revenues - np.apply_along_axis(sum, 1, A.T * pi) > 0
    if all(np.invert(s_prime)):
        return tuple(offer_tuple)

    # line 2-3
    # offer_sets_to_test has in each row an offer set, we want to test
    offer_sets_to_test = np.zeros((sum(s_prime), len(revenues)))
    offer_sets_to_test[np.arange(sum(s_prime)), np.where(s_prime)] = 1
    offer_sets_to_test += offer_tuple
    offer_sets_to_test = (offer_sets_to_test > 0)

    value_marginal = np.apply_along_axis(calc_value_marginal, 1, offer_sets_to_test, pi)

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
        value_marginal = np.apply_along_axis(calc_value_marginal, 1, offer_sets_to_test, pi)

        if np.amax(value_marginal) > v_s:
            v_s = np.amax(value_marginal)
            offer_tuple[np.argmax(value_marginal)] = 1
            s_prime = s_prime & offer_tuple == 0
            if all(offer_tuple == 1):
                break
        else:
            break
    return tuple(offer_tuple)


def calc_value_marginal(indices_inner_sum, pi):
    v_temp = 0
    for l in np.arange(len(preference_weights)):  # sum over all customer segments
        v_temp += arrival_probabilities[l] * \
                  sum(indices_inner_sum * (revenues - np.apply_along_axis(sum, 1, A.T * pi)) *
                      preference_weights[l, :]) / \
                  (sum(indices_inner_sum * preference_weights[l, :]) + var_no_purchase_preferences[l][0])
    return v_temp


def simulate_sales(offer_set, random_customer, random_sales):
    customer = random_customer <= np.array([*np.cumsum(arrival_probabilities), 1.0])
    customer = min(np.array(range(0, len(customer)))[customer])

    if customer == len(arrival_probabilities):
        return len(products)  # no customer arrives => no product sold (product out of range)
    else:
        product = random_sales <= np.cumsum(customer_choice_individual(offer_set,
                                                             preference_weights[customer],
                                                             preferences_no_purchase[customer]))
        product = min(np.arange(len(products) + 1)[product])
        return product


# %%
# Actual Code

# theta and pi as calculated
with open(result_folder+"\\thetaResult.data", "rb") as filehandle:
    thetas = pickle.load(filehandle)
with open(result_folder+"\\piResult.data", "rb") as filehandle:
    pis = pickle.load(filehandle)

#%%
# online_K+1 policy iterations (starting with 0)
v_results = np.array([np.zeros(len(times))]*online_K)
c_results = np.array([np.zeros(shape=(len(times), len(capacities)))]*online_K)

# to use in single timestep (will be overwritten)
pi = np.zeros(len(resources))

value_final = pd.DataFrame(v_results[:,0])  # setup result storage empty
value_final['' + str(capacities[0]) + '-' + str(preferences_no_purchase[0])] = pd.DataFrame(v_results[:, 0])
#%%
for capacities in var_capacities:
    for preferences_no_purchase in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities.tolist()), " - and - ", preferences_no_purchase, "of", str(var_no_purchase_preferences.tolist()), "starting.")
        for k in np.arange(online_K)+1:
            print(k, "of", online_K, "starting.")
            customer_random_stream = test_customer[k-1]
            sales_random_stream = test_sales[k-1]
            
            # line 3
            r_result = np.zeros(len(times))  # will become v_result
            c_result = np.zeros(shape=(len(times), len(capacities)), dtype=int)

            # line 5
            c = copy.deepcopy(capacities)  # (starting capacity at time 0)

            for t in times:
                # line 7  (starting capacity at time t)
                c_result[t] = c

                # line 12  (epsilon greedy strategy)
                pi[c == 0] = np.inf
                pi[c > 0] = pis[t][c > 0]
                offer_set = determine_offer_tuple(pi, eps=0)

                # line 13  (simulate sales)
                sold = simulate_sales(offer_set, customer_random_stream[t], sales_random_stream[t])

                # line 14
                try:
                    r_result[t] = revenues[sold]
                    c -= A[:, sold]
                except IndexError:
                    # no product was sold
                    pass

            # line 16-18
            v_results[k-1] = np.cumsum(r_result[::-1])[::-1]
            c_results[k-1] = c_result

        value_final['' + str(capacities[0]) + '-' + str(preferences_no_purchase[0])] = pd.DataFrame(v_results[:, 0])


# %%
# write result of calculations
with open(newpath+"\\vResultsTable1.data", "wb") as filehandle:
    pickle.dump(value_final, filehandle)


# %%
wrapup(logfile, time_start, newpath)
