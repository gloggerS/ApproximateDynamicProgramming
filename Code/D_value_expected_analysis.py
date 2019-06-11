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

# %%
total_results = pickle.load(open("C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\singleLegFlight-True-DP-190611-0917\\totalresults.data", "rb"))

total_results.__len__()

type(total_results[0][0])
total_results[0][0].shape
total_results[0][0].iloc[40,:]
