from simulationBasic import *

import numpy as np
from copy import deepcopy

from gurobipy import *

#%%
def policy_iteration(K, I):
    theta = np.zeros(T+1)
    pi = np.zeros(shape=(T+1, capacity + 1))

    v_samples = list()
    c_samples = list()

    for k in np.arange(K):
        # line 4
        for i in np.arange(I):
            v, c = policy_evaluation(pi)
            v_samples.append(v)
            c_samples.append(c)

        # line 20
        theta, pi = update_parameters(v_samples, c_samples, theta, pi, I)

    return theta, pi


def policy_evaluation(pi):
    """
    Evaluates one step of the policy iteration. Compare Sebastian Fig. 1
    :param i:
    :return: sample data v_sample (value function) and c_sample (capacity)
    """
    # line 3
    v_sample = np.zeros(T)
    c_sample = np.zeros(T)
    r_sample = np.zeros(T)

    # line 5
    c = capacity

    # line 6
    for t in np.arange(T):
        # line 7
        c_sample[t] = c

        # line 9
        if c == 0:
            pi[t+1, c] = np.inf
        else:
            # line 10
            pi[t+1, c] = pi[t+1, c]  # piecewise linear => compute_pi(c)

        # line 12
        x = determine_offer_tuple(pi[t + 1, c])

        # line 13
        j = simulate_sales(x)

        # line 14
        if j > 0:
            r_sample[t] = revenues[j-1]
            c = c-1

    # line 16-18
    v_sample = np.cumsum(r_sample[::-1])[::-1]

    return v_sample, c_sample


def determine_offer_tuple(pi):
    """
    Determines the offerset given the bid prices for each resource.

    Implement the Greedy Heuristic from Bront et al: A Column Generation Algorithm ... 4.2.2

    :param pi:
    :return:
    """

    # setup
    offer_tuple_new = np.zeros_like(revenues)

    # line 1
    S = revenues - np.ones_like(revenues)*pi > 0

    # line 2-3
    value_marginal = np.zeros_like(revenues)
    for i in np.arange(value_marginal.size):
        for l in np.arange(len(preference_weights)):
            value_marginal[i] += (revenues[i] - pi)*preference_weights[l, i]/\
                              (preference_weights[l, i] + preference_no_purchase[l])
    value_marginal = S * value_marginal
    offer_tuple_new[np.argmax(value_marginal)] = 1
    v_new = max(value_marginal)

    # line 4
    while True:
        # 4a
        offer_tuple = deepcopy(offer_tuple_new)
        v_akt = v_new
        # j_mat has in rows all the offer sets, we want to test now
        j_mat = np.zeros((sum(S), len(preference_weights)))
        j_mat[np.arange(sum(S)), np.where(S)] = 1
        j_mat += offer_tuple
        j_mat = (j_mat > 0)

        def calc_value_marginal(x):
            v_temp = 0
            for l in np.arange(len(preference_weights)):
                v_temp += sum(x*(revenues - np.ones_like(revenues)*pi) * preference_weights[l, :]) / \
                                     sum(x*(preference_weights[l, :] + preference_no_purchase[l]))
            return v_temp

        # 4b
        value_marginal = np.apply_along_axis(calc_value_marginal, axis=1, arr=j_mat)
        v_new = max(value_marginal)
        if v_new > v_akt:
            offer_tuple_new[np.argmax(value_marginal)] = 1
            if all(offer_tuple == offer_tuple_new):
                break
        else:
            break

    return tuple(offer_tuple)


def simulate_sales(offer_tuple):
    products_with_no_purchase = np.arange(revenues.size+1)
    customer_probabilities = customer_choice_all(offer_tuple)
    return int(np.random.choice(products_with_no_purchase, size=1, p=customer_probabilities))


def update_parameters(v_samples, c_samples, theta, pi, I):
    """
    Updates the parameter theta, pi given the sample path using least squares linear regression with constraints.
    Using gurobipy

    :param v_samples:
    :param c_samples:
    :param theta:
    :param pi:
    :return:
    """
    set_i = np.arange(I)
    set_t = np.arange(len(theta)-1)
    set_c = np.arange(len(pi[0]))

    theta_multidict = {}
    for t in set_t:
        theta_multidict[t] = theta[t]
    theta_indices, theta_tuples = multidict(theta_multidict)

    pi_multidict = {}
    for t in set_t:
        for c in set_c:
            pi_multidict[t, c] = pi[t, c]
    pi_indices, pi_tuples = multidict(pi_multidict)

    try:
        m = Model()

        # Variables
        mTheta = m.addVars(theta_indices, name="theta", lb=0.0)  # Constraint 10
        mPi = m.addVars(pi_indices, name="pi", ub=max(revenues))  # Constraint 11

        for t in set_t:
            mTheta[t].start = theta_tuples[t]
            for c in set_c:
                mPi[t, c].start = pi_tuples[t, c]

        # Goal Function
        lse = quicksum((v_samples[i][t] - mTheta[t]-quicksum(mPi[t, c]*c_samples[t][c] for c in set_c)) *
                       (v_samples[i][t] - mTheta[t]-quicksum(mPi[t, c]*c_samples[t][c] for c in set_c))
                       for i in set_i for t in set_t)
        m.setObjective(lse, GRB.MINIMIZE)

        # Constraints
        # C12 (not needed yet)
        for t in set_t[:-1]:
            m.addConstr(mTheta[t], GRB.GREATER_EQUAL, mTheta[t+1], name="C15")  # Constraint 15
            for c in set_c:
                m.addConstr(mPi[t, c], GRB.GREATER_EQUAL, mPi[t+1, c], name="C16")  # Constraint 16

        m.optimize()

        theta_new = deepcopy(theta)
        pi_new = deepcopy(pi)

        for t in set_t:
            theta_new[t] = m.getVarByName("theta[" + str(t) + "]").X
            for c in set_c:
                pi_new[t, c] = m.getVarByName("pi[" + str(t) + "," + str(c) + "]").X

        return theta_new, pi_new
    except GurobiError:
        print('Error reported')

        return 0, 0


#%% testing

determine_offer_tuple(0)