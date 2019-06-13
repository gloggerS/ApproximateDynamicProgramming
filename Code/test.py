import pickle
import numpy as np
import pandas as pd
import copy

with open(r"C:\Users\Stefan\LRZ Sync+Share\Masterarbeit-Klein\Code\Results\1-no-exponential-smoothing--smallTest-False-API-lin-190612-1114\thetaAll.data", "rb") as f:
    thetas_no_es = pickle.load(f)

thetas_calculated = copy.deepcopy(thetas_no_es)
for k in np.arange(len(thetas_no_es)-1)+1:
    thetas_calculated[k] = np.average(thetas_no_es[1:(k+1)], axis=0)