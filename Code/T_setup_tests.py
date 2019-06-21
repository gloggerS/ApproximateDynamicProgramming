"""
Script started, but not so nice to incorporate probabilities in such a way
"""

import pandas as pd
import numpy as np

import datetime
import time

import os
from shutil import copyfile, move

from A_data_read import *
from B_helper import *
from ast import literal_eval

import pickle
import random


#%%
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
    customer_segments, preference_weights, arrival_probabilities, times, T, time_start, K, online_K \
    = setup_testing("TestingAll")

#%%

prob_customers = np.array([np.zeros(len(times))]*online_K)
prob_product_sold = np.array([np.zeros(len(times))]*online_K)

for k in np.arange(online_K):
    np.random.seed(K+k+1)  # +1 to avoid overlap with random number generation before (in API calculation)
    prob_customers[k] = np.random.choice(np.arange(len(arrival_probabilities)+1),
                                         size=len(times),
                                         p=np.array([*arrival_probabilities, 1-sum(arrival_probabilities)]))
