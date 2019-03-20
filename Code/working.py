from simulationBasic import *

import numpy as np
from copy import deepcopy
import pandas as pd

from gurobipy import *
import re

# %% working on Bront CDLP -
# Ziel: Revenue results for parallel flights example reproduzieren (Table A.1, Spalte CDLP)


def revenue_all(offer_set_tuple):
    return sum(revenues * customer_choice_all(offer_set_tuple)[1:])

def quantity_all(offer_set_tuple, i):
    return sum(A[i-1, :] * customer_choice_all(offer_set_tuple)[1:])

def CDLP():
    offer_sets_all = pd.DataFrame(list(map(list, itertools.product([0, 1], repeat=numProducts))))
    offer_sets_all = offer_sets_all[1:]  # always at least one product to offer

    S = {}
    R = {}
    Q = {}
    for index, offer_array in offer_sets_all.iterrows():
        S[index] = tuple(offer_array)
        R[index] = revenue_all(tuple(offer_array))
        temp = {}
        for i in np.arange(len(A)):
            temp[i] = quantity_all(tuple(offer_array), i)
        Q[index] = temp


    try:
        m = Model()

        # Variables
        mt = m.addVars(offer_sets_all.index.values, name="t", lb=0.0)  # last constraint

        # Objective Function
        m.setObjective(lam * quicksum(R[s]*mt[s] for s in offer_sets_all.index.values), GRB.MAXIMIZE)

        # Constraints
        for i in np.arange(len(A)):
            m.addConstr(lam * quicksum(Q[s][i]*mt[s] for s in offer_sets_all.index.values), GRB.LESS_EQUAL, capacities[i],
                        name="constraintOnResource")
        m.addConstr(quicksum(mt[s] for s in offer_sets_all.index.values), GRB.LESS_EQUAL, T)

        m.optimize()

        pat = r".*?\[(.*)\].*"
        for v in m.getVars():
            if v.X > 0:
                match = re.search(pat, v.VarName)
                erg_index = match.group(1)
                print(offer_sets_all.loc[int(erg_index)], ": ", v.X)

    except GurobiError:
        print('Error reported')
