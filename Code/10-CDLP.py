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

#%%
# Get settings, prepare data, create storage for results
print("CDLP starting.\n\n")

# time
print("Time:", datetime.datetime.now())
time_start = time.time()

# settings
settings = pd.read_csv("0_settings.csv", delimiter=";", header=None)
for row in settings:
    print(settings.loc[row, 0], ":\t", settings.loc[row, 1])

example = settings.iloc[0, 1]
use_variations = settings.iloc[1, 1]  # true if varying capacities should be used

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

# prepare storage location
newpath = os.getcwd()+"\\Results\\"+str(datetime.datetime.now()).replace(":", "-").replace(".", "-")
os.makedirs(newpath)

# copy settings to storage location
copyfile("0_settings.csv", newpath+"\\0_settings.csv")

print("\nEverything set up.")

#%%
# Actual Code


# %%
# Run CDLP

var_capacities, var_no_purchase_preferences = get_variations()

num_rows = len(var_capacities)*len(var_no_purchase_preferences)
df = pd.DataFrame(index=np.arange(num_rows), columns=['c', 'u', 'DP', 'CDLP'])
indexi = 0
for capacity in var_capacities:
    for preference_no_purchase in var_no_purchase_preferences:
        print(capacity)
        print(preference_no_purchase)

        df.loc[indexi] = [capacity, preference_no_purchase, value_expected(capacities=capacity, t=0,
                                                                           preference_no_purchase=preference_no_purchase),
                          CDLP_by_column_generation(capacities=capacity, preference_no_purchase=preference_no_purchase)]
        indexi += 1

df.to_pickle("CDLP.pkl", newpath)
