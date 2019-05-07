from dat_Koch import *

#%%
def policy_iteration(K, I, capacities, thresholds, T):
    """

    :param K: number of policy iterations
    :param I: number of samples in one policy iteration
    :param capacities: vector of initial capacity for each resource (i \in [m])
    :param thresholds: (i, S_i) data object of thresholds S_i for each resource i
    :return:
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()

    theta = np.zeros(len(times))
    length_thresholds = np.arange(len(thresholds))
    for i in length_thresholds:
        length_thresholds[i] = len(thresholds[i])

    pi = np.zeros(shape=(len(times)+1, len(capacities), max(length_thresholds)))

    customers_with_none = np.arange(len(customer_segments) + 1)
    customer_probabilities = np.append(arrival_probabilities, 1-sum(arrival_probabilities))

    customer_stream = np.zeros(shape=(I, len(times)), dtype=int)
    for i in np.arange(I):
        customer_stream[i] = np.random.choice(customers_with_none, size=len(times), p=customer_probabilities)

    v_samples = np.zeros(len(times))
    c_samples = list()

    for k in np.arange(K):
        # line 4
        for i in np.arange(I):
            v, c = policy_evaluation(pi, customer_stream[i])
            v_samples[i] = v
            c_samples.append(c)

        # line 20
        theta, pi = update_parameters(v_samples, c_samples, theta, pi, I)

    return theta, pi


def policy_evaluation(pi, customer_stream_vector):
    """
    Evaluates one step of the policy iteration. Compare Sebastian Fig. 1
    :param pi:
    :param customer_stream_vector:
    :return: sample data v_sample (value function) and c_sample (capacity)
    """
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()

    # line 3
    c_sample = np.zeros(shape=(len(times), len(capacities)), dtype=int)
    r_sample = np.zeros(len(times))

    pi_act = np.zeros(shape=(len(times)+1, len(capacities)))

    # line 5
    c = capacities  # (starting capacity at time 0)

    # line 6
    for t in times:
        # line 7  (starting capacity at time t)
        c_sample[t] = c

        # line 9  (adjust bid price)
        for i in resources:
            if c_sample[t, i] == 0:
                pi_act[t+1, i] = np.inf
            else:
                print(i)
                pi_act[t+1, i] = get_pi(pi, t, i, threshold_vector=thresholds[i], c_i=c_sample[t, i])

        # line 12
        x = calculate_offer_set(c_sample[t, :], preference_no_purchase, t, pi_act[t+1, :], beta=1, dataName="")
        offer_set = np.array(x[2].iloc[x[0], :])

        # line 13
        j = simulate_sale(offer_set=offer_set, customer_number=customer_stream_vector[t])

        # line 14
        if j < len(products):
            # product sold
            r_sample[t] = revenues[j]
            c = c_sample[t]-A[:, j]

    # line 16-18
    v_sample = np.cumsum(r_sample[::-1])[::-1]

    return v_sample, c_sample


def get_pi(pi, t, i, threshold_vector, c_i):
    cond1 = np.append([True], threshold_vector < c_i)[:-1]
    cond2 = (c_i <= threshold_vector)
    temp = np.where(cond1 & cond2)
    if len(temp) > 1:
        raise ValueError("Multiple Pi-values could be possible. For no thresholds, put threshold to upper bound.")
    return float(pi[t + 1, i, temp[0]])


def determine_offer_tuple(pi):
    """
    OLD Implementation
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


def simulate_sale(offer_set, customer_number):
    resources, \
        products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, \
        times = get_data_without_variations()

    if customer_number == len(customer_segments):
        return len(revenues)+1
    else:
        customer_prob = customer_choice_individual(offer_set,
                                                            preference_weights[customer_number],
                                                            preference_no_purchase[customer_number])
        return int(np.random.choice(np.arange(len(products)+1), size=1, p=customer_prob))

def simulate_sales(offer_tuple):
    products_with_no_purchase = np.arange(revenues.size+1)
    customer_probabilities = customer_choice_vector(offer_tuple)
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

# determine_offer_tuple(0)

capacities = var_capacities[1]
thresholds = [[]]*3
step_size = 4
thresholds[0] = np.arange(0, capacities[0]+step_size, step=step_size, dtype=int)  # will include 1 value >= max capacity
thresholds[1] = np.arange(0, capacities[1]+step_size, step=step_size, dtype=int)
thresholds[2] = np.arange(0, capacities[2]+step_size, step=step_size, dtype=int)


