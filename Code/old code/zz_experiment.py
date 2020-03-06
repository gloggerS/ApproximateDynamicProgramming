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
result_folder = 'C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\singleLegFlight-True-APILinearSingleLeg-190702-2200'
with open(result_folder+"\\piAll.data", "rb") as filehandle:
    pi_all = pickle.load(filehandle)

#%%
pi_all
