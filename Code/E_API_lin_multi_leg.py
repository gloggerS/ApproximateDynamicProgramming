"""
This script will calculate the Approximate Policy Iteration with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
"""

from B_helper import *

#%%
# Setup of parameters
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("APILinearMultiLeg")


capacities = var_capacities[0]
preferences_no_purchase = var_no_purchase_preferences[0]

# %% to store workspace
import shelve
filename = newpath+"\\shelve.out"

#%%
def update_parameters(v_samples, c_samples, thetas, pis, k, exponential_smoothing):
    """
    Updates the parameter theta, pi given the sample path using least squares linear regression with constraints.
    Using gurobipy

    :param v_samples:
    :param c_samples:
    :return:
    """
    set_i = np.arange(len(v_samples))
    set_t = np.arange(len(thetas))
    set_h = np.arange(len(pis[0]))

    theta_multidict = {}
    for t in set_t:
        theta_multidict[t] = thetas[t]
    theta_indices, theta_tuples = multidict(theta_multidict)

    pi_multidict = {}
    for t in set_t:
        for h in set_h:
            pi_multidict[t, h] = pis[t, h]
    pi_indices, pi_tuples = multidict(pi_multidict)

    try:
        m = Model()

        # Variables
        m_theta = m.addVars(theta_indices, name="theta", lb=0.0)  # Constraint 10
        m_pi = m.addVars(pi_indices, name="pi", ub=max(revenues), lb=0.0)  # Constraint 11

        for t in set_t:
            m_theta[t].start = theta_tuples[t]
            for h in set_h:
                m_pi[t, h].start = pi_tuples[t, h]

        # Goal Function (14)
        lse = quicksum((v_samples[i][t] - m_theta[t] - quicksum(m_pi[t, h] * c_samples[i][t][h] for h in set_h)) *
                       (v_samples[i][t] - m_theta[t] - quicksum(m_pi[t, h] * c_samples[i][t][h] for h in set_h))
                       for t in set_t for i in set_i)
        m.setObjective(lse, GRB.MINIMIZE)

        # Constraints
        # C12 (not needed yet)
        for t in set_t[:-1]:
            m.addConstr(m_theta[t], GRB.GREATER_EQUAL, m_theta[t+1], name="C15")  # Constraint 15
            for h in set_h:
                m.addConstr(m_pi[t, h], GRB.GREATER_EQUAL, m_pi[t+1, h], name="C16")  # Constraint 16

        m.optimize()

        theta_new = deepcopy(thetas)
        pi_new = deepcopy(pis)

        for t in set_t:
            theta_new[t] = m.getVarByName("theta[" + str(t) + "]").X
            for h in set_h:
                pi_new[t, h] = m.getVarByName("pi[" + str(t) + "," + str(h) + "]").X

        # check exponential smoothing
        if exponential_smoothing:
            return (1 - 1 / k) * thetas + 1 / k * theta_new, (1 - 1 / k) * pis + 1 / k * pi_new, m.ObjVal
        else:
            return theta_new, pi_new, m.ObjVal
    except GurobiError:
        print('Error reported')

        return 0, 0, 0


# %%
# Actual Code

pi_result = {}
theta_result = {}
value_result = {}
capacities_result = {}

# generate the random sample paths (T+1 => have real life indexing starting at t=1)
np.random.seed(12)
seed(12)
customer_stream = np.random.random((I, T+1))
eps_random = np.random.random((I, T+1))
sales_stream = np.random.random((I, T+1))

# summary statistics
offer_sets_all = {str(tuple(obj)): count for count, obj in enumerate(get_offer_sets_all(products))}
offersets_offered_dic = {}
products_sold_dic = {}
sold_out_dic = {}
obj_vals_dic = {}

for capacities in var_capacities:
    value_result[str(capacities)] = {}
    capacities_result[str(capacities)] = {}
    theta_result[str(capacities)] = {}
    pi_result[str(capacities)] = {}

    offersets_offered_dic[str(capacities)] = {}
    products_sold_dic[str(capacities)] = {}
    sold_out_dic[str(capacities)] = {}
    obj_vals_dic[str(capacities)] = {}

    for preferences_no_purchase in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities.tolist()), " - and - ",
              preferences_no_purchase, "of", str(var_no_purchase_preferences.tolist()), "starting.")

        # summary statistics
        offersets_offered = np.zeros((K + 1, len(offer_sets_all)))
        products_sold = np.zeros((K + 1, len(products) + 1))
        sold_out = np.zeros((K + 1, I))
        obj_vals = np.zeros(K + 1)

        value_result[str(capacities)][str(preferences_no_purchase)] = np.zeros((K+1, I, T+1))
        capacities_result[str(capacities)][str(preferences_no_purchase)] = np.zeros((K+1, I, T+1, len(resources)))

        # store parameters over all policy iterations
        # K+1 policy iterations (starting with 0)
        # T time steps
        theta_all = np.zeros((K+1, T+1, 1))
        pi_all = np.zeros((K+1, T+1, len(resources)))

        # theta and pi for each time step
        # line 1
        thetas = 0
        pis = np.zeros(len(resources))

        for k in np.arange(K)+1:
            print(k, "of", K, "starting.")

            v_samples = np.zeros((I, T+1))
            c_samples = np.zeros((I, T+1, len(capacities)))

            for i in np.arange(I):
                # line 3
                r_sample_i = np.zeros(T+1)  # will become v_sample
                c_sample_i = np.zeros(shape=(T+1, len(capacities)), dtype=int)

                # line 5
                c = deepcopy(capacities)  # (starting capacity at time 0)

                for t in times+1:
                    # line 7  (starting capacity at time t)
                    c_sample_i[t-1] = c

                    # line 8-11  (adjust bid price)
                    pis[c == 0] = np.inf
                    pis[c > 0] = pi_all[k - 1, t, c > 0]

                    # line 12  (epsilon greedy strategy)
                    offer_set = determine_offer_tuple(pis, epsilon[k], eps_random[i, t], revenues, A,
                                                      arrival_probabilities, preference_weights, preferences_no_purchase)
                    offersets_offered[k, offer_sets_all[str(offer_set)]] += 1

                    # line 13  (simulate sales)
                    sold = simulate_sales(offer_set, customer_stream[i, t], sales_stream[i, t],
                                          arrival_probabilities, preference_weights, preferences_no_purchase)
                    products_sold[k, sold] += 1

                    # line 14
                    try:
                        r_sample_i[t] = revenues[sold]
                        c -= A[:, sold]
                    except IndexError:
                        # no product was sold
                        pass
                    
                    if all(c==0):
                        sold_out[k, i] = t

                # also adjust last sell
                c_sample_i[t] = c

                # line 16-18
                v_samples[i] = np.cumsum(r_sample_i[::-1])[::-1]
                c_samples[i] = c_sample_i

            # line 20
            theta_all[k], pi_all[k], obj_vals[k] = update_parameters(v_samples, c_samples, theta_all[k-1], pi_all[k-1], k, exponential_smoothing)
            
            value_result[str(capacities)][str(preferences_no_purchase)][k, :, :] = v_samples
            capacities_result[str(capacities)][str(preferences_no_purchase)][k, :, :] = c_samples

        theta_result[str(capacities)][str(preferences_no_purchase)] = theta_all
        pi_result[str(capacities)][str(preferences_no_purchase)] = pi_all

        offersets_offered_dic[str(capacities)][str(preferences_no_purchase)] = offersets_offered
        products_sold_dic[str(capacities)][str(preferences_no_purchase)] = products_sold[1:, :]
        sold_out_dic[str(capacities)][str(preferences_no_purchase)] = sold_out
        obj_vals_dic[str(capacities)][str(preferences_no_purchase)] = obj_vals_dic


# %%
# write result of calculations
with open(newpath+"\\thetaAll.data", "wb") as filehandle:
    pickle.dump(theta_result, filehandle)

with open(newpath+"\\piAll.data", "wb") as filehandle:
    pickle.dump(pi_result, filehandle)

with open(newpath+"\\valueAll.data", "wb") as filehandle:
    pickle.dump(value_result, filehandle)

with open(newpath+"\\capacitiesAll.data", "wb") as filehandle:
    pickle.dump(capacities_result, filehandle)

with open(newpath+"\\thetaToUse.data", "wb") as filehandle:
    tmp = {}
    for capacities in var_capacities:
        tmp[str(capacities)] = {}
        for preferences_no_purchase in var_no_purchase_preferences:
            tmp[str(capacities)][str(preferences_no_purchase)] = theta_result[str(capacities)][str(preferences_no_purchase)][K]
    pickle.dump(tmp, filehandle)

with open(newpath+"\\piToUse.data", "wb") as filehandle:
    tmp = {}
    for capacities in var_capacities:
        tmp[str(capacities)] = {}
        for preferences_no_purchase in var_no_purchase_preferences:
            tmp[str(capacities)][str(preferences_no_purchase)] = pi_result[str(capacities)][str(preferences_no_purchase)][K]
    pickle.dump(tmp, filehandle)

with open(newpath+"\\offersetsOffered.data", "wb") as filehandle:
    pickle.dump(offersets_offered_dic, filehandle)
    
with open(newpath+"\\soldOut.data", "wb") as filehandle:
    pickle.dump(sold_out_dic, filehandle)

# products sold
p = pd.DataFrame(products_sold_dic, columns=range(len(products)+1))
with open(newpath+"\\plotProducts.data", "wb") as filehandle:
    pickle.dump(p, filehandle)


# %%
wrapup(logfile, time_start, newpath)


# %%
my_shelf = shelve.open(filename, "n")  # "n" for new
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
    except Exception as e:
        print("Error {0}".format(e))
my_shelf.close()

# %% restore
# my_shelf = shelve.open(filename)
# for key in my_shelf:
#     globals()[key] = my_shelf[key]
# my_shelf.close()
