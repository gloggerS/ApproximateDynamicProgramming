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

from scipy import stats

#%%
# Get settings, prepare data, create storage for results
print("Comparison of strategies starting.\n\n")

# settings
settings = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
example = settings.iloc[0, 1]
use_variations = (settings.iloc[1, 1] == "True") | (settings.iloc[1, 1] == "true")  # if var. capacities should be used
storage_folder = example + "-" + str(use_variations) + "-comparison-evaluation-" + time.strftime("%y%m%d-%H%M")
online_K = int(settings.loc[settings[0] == "online_K", 1].item())

#%%
def append_row(name, folder):
    return results.append({"name": name, "folder": folder},
                   ignore_index=True)

folder_results = 'C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\'
results = pd.DataFrame(columns=["name", "folder"])
# results = append_row("DP", folder_results + 'smallTest2-False-DP-evaluation-190616-1435')
# results = append_row("API_lin", folder_results + 'smallTest2-False-API-lin-evaluation-190616-1211')
# results = append_row("API_plc", folder_results + 'smallTest2-False-API-plc-evaluation-190616-1245')

results = append_row("DP", folder_results + 'smallTest3-False-DPSingleLeg-Evaluation-190624-1007')
results = append_row("API_lin", folder_results + 'smallTest3-False-APILinearSingleLeg-Evaluation-190624-1029')


#%%
for i in results.iterrows():
    print(i[1].iloc[1])

#%%
dat = pd.DataFrame()
for i in results.iterrows():
    with open(i[1].iloc[1] + "\\vResults.data", "rb") as filehandle:
        values = pickle.load(filehandle)
    dat[i[1].iloc[0]] = values

#%%
# https://pythonfordatascience.org/paired-samples-t-test-python/
def analysis(name):
    differences = dat[name] - dat["DP"]
    print("Mean of ", name, "-DP: \t", np.mean(differences))
    print("Standard t-test: \t")

# describe data
dat[["DP", "API_lin"]].describe()

# calculate differences
dat["diff_lin"] = dat["API_lin"] - dat["DP"]

# check for outliers
dat[["DP", "API_lin"]].plot(kind="box", title="Box Plots for strategy DP and for strategy API_lin")
# plt.savefig("outliers.png")
plt.show()

# check normal distribution
dat["diff_lin"].plot(kind="hist", title="Histogram of differences between final values API_lin - DP")
plt.show()

stats.probplot(dat["diff_lin"], plot=plt)
plt.title("Q-Q Plot of differences between final values API_lin - DP")
plt.show()

stats.shapiro(dat["diff_lin"])

# difference between paired means (matched-pairs t-test)
# https://www.stattrek.com/hypothesis-test/paired-means.aspx
# null hypothesis: difference smaller equal 0 (erwarte, dass API - DP < 0)
se = stats.sem(dat["diff_lin"], ddof=1)
t = (np.mean(dat["diff_lin"])-0)/se
n = len(dat["diff_lin"])
p = 1 - stats.t.cdf(t, df=n)
print("p-value =", p)
if p < 0.05:
    print("p-value is smaller then 0.05, therefore reject null hypothesis API < DP")
