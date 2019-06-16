"""
This script will calculate the MLP regressor with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
https://www.machinelearningtutorial.net/2017/01/28/python-scikit-simple-function-approximation/

IDEE: Alles analog zu API single leg, aber jetzt theta und pi mit MLP Regressor lernen
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
from sklearn.neural_network import MLPRegressor

#%%
# Get settings, prepare data, create storage for results
print("MLP single leg starting.\n\n")

# settings
settings = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
example = settings.iloc[0, 1]
use_variations = (settings.iloc[1, 1] == "True") | (settings.iloc[1, 1] == "true")  # if var. capacities should be used
storage_folder = example + "-" + str(use_variations) + "-MLP-" + time.strftime("%y%m%d-%H%M")
K = int(settings.loc[settings[0] == "K", 1])
I = int(settings.loc[settings[0] == "I", 1])
epsilon = eval(str(settings.loc[settings[0] == "epsilon", 1].item()))
exponential_smoothing = settings.loc[settings[0] == "exponential_smoothing", 1].item()
exponential_smoothing = (exponential_smoothing == "True") | (exponential_smoothing == "true")

# data
dat = get_all(example)
print("\n Data used. \n")
for key in dat.keys():
    print(key, ":\n", dat[key])
print("\n\n")
del dat

# prepare storage location
newpath = get_storage_path(storage_folder)
os.makedirs(newpath)

# copy settings to storage location
copyfile("0_settings.csv", newpath+"\\0_settings.csv")
logfile = open(newpath+"\\0_logging.txt", "w+")  # write and create (if not there)

# time
print("Time:", datetime.datetime.now())
print("Time (starting):", datetime.datetime.now(), "\n", file=logfile)
time_start = time.time()

# settings
for row in np.arange(len(settings)):
    print(settings.loc[row, 0], ":\t", settings.loc[row, 1])
    print(settings.loc[row, 0], ":\t", settings.loc[row, 1], file=logfile)
print("correct epsilon for epsilon-greedy strategy: ", len(epsilon) == K+1)
print("\ncorrect epsilon for epsilon-greedy strategy: ", len(epsilon) == K+1, file=logfile)

# variations (capacity and no purchase preference)
if use_variations:
    var_capacities, var_no_purchase_preferences = get_variations(example)
    preferences_no_purchase = var_no_purchase_preferences[0]
else:
    capacities, preferences_no_purchase = get_capacities_and_preferences_no_purchase(example)
    var_capacities = np.array([capacities])
    var_no_purchase_preferences = np.array([preferences_no_purchase])

# no varying capacity here
capacities = var_capacities[0]

# other data
resources, \
    products, revenues, A, \
    customer_segments, preference_weights, arrival_probabilities, \
    times = get_data_without_variations(example)
T = len(times)

capacities_thresholds = get_capacities_thresholds(example)

print("\nEverything set up.")

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
    X_all = pd.DataFrame()
    y_all = pd.DataFrame()

    for t in times:
        # 1 für theta und capacity für pi
        eins = np.ones_like(c_samples[:, t])
        cap = c_samples[:, t]
        X = np.array([(*eins[i], *cap[i]) for i in np.arange(len(c_samples))])

        y = v_samples[:, t]

        X_all = X_all.append(pd.DataFrame(X))
        y_all = y_all.append(pd.DataFrame(y))

    clf = MLPRegressor(alpha=0.01, hidden_layer_sizes=(10,), max_iter=50000,
                       activation="logistic", verbose="True", learning_rate="adaptive")
    a = clf.fit(X_all, y_all)

    # Zeit muss auch mit ins Modell

    for t in times:
        clf.predict(np.array([[1, 12]])) - clf.predict(np.array([[1, 11]]))
    clf.predict(np.array([[1, 11]])) - clf.predict(np.array([[1, 10]]))
    clf.predict(np.array([[1, 1]])) - clf.predict(np.array([[1, 0]]))


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

# %%
# Actual Code
# K+1 policy iterations (starting with 0)
# T time steps
theta_all = np.array([[np.zeros(1)]*T]*(K+1))
pi_all = np.array([[np.zeros(len(resources))]*T]*(K+1))

# theta and pi for each time step
# line 1
thetas = 0
pis = np.zeros(len(resources))

for k in np.arange(K)+1:
    np.random.seed(k)
    random.seed(k)

    v_samples = np.array([np.zeros(len(times))]*I)
    c_samples = np.array([np.zeros(shape=(len(times), len(capacities)))]*I)

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
            pis[c_sample[t] > 0] = pi_all[k - 1][t][c_sample[t] > 0]

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
    theta_all[k], pi_all[k] = update_parameters(v_samples, c_samples, theta_all[k-1], pi_all[k-1], k)


# %%
# write result of calculations
with open(newpath+"\\thetaAll.data", "wb") as filehandle:
    pickle.dump(theta_all, filehandle)

with open(newpath+"\\piAll.data", "wb") as filehandle:
    pickle.dump(pi_all, filehandle)

with open(newpath+"\\thetaResult.data", "wb") as filehandle:
    pickle.dump(theta_all[K], filehandle)

with open(newpath+"\\piResult.data", "wb") as filehandle:
    pickle.dump(pi_all[K], filehandle)


# %%
time_elapsed = time.time() - time_start
print("\n\nTotal time needed:\n", time_elapsed, "seconds = \n", time_elapsed/60, "minutes", file=logfile)
logfile.close()
print("\n\n\n\nDone. Time elapsed:", time.time() - time_start, "seconds.")
print("Results stored in: "+newpath)
