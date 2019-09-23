"""
This script will calculate the Approximate Policy Iteration with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
"""

from B_helper import *


#%%
result_folder = "C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\exampleStefan-True-APILinearMultiLeg-190918-1053"

#%%
a = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
b = pd.read_csv(result_folder+"\\0_settings.csv", delimiter="\t", header=None)
if not(a.equals(b)):
    raise ValueError("SettingFiles don't coincide.")

logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("APILinearMultiLeg-2-evaluation")
which_policy_iteration_to_use = -2

capacities = var_capacities[0]
no_purchase_preference = var_no_purchase_preferences[0]


#%% open result of calculations
with open(result_folder+"\\thetaAll.data", "rb") as filehandle:
    theta_result = pickle.load(filehandle)

with open(result_folder+"\\piAll.data", "rb") as filehandle:
    pi_result = pickle.load(filehandle)

with open(result_folder+"\\thetaToUse.data", "rb") as filehandle:
    theta_to_use = pickle.load(filehandle)

with open(result_folder+"\\piToUse.data", "rb") as filehandle:
    pi_to_use = pickle.load(filehandle)

#%%
p = pi_to_use
p = p[str(capacities)][str(no_purchase_preference)]

p2 = pi_result
p2 = p2[str(capacities)][str(no_purchase_preference)]

if np.sum(p != p2[-which_policy_iteration_to_use, :, :]) > 0:
    raise Warning("You currently don't use the most recent policy iteration. Variable to adjust: which_policy_iteration_to_use")

#%%
# generate the random sample paths (T+1 => have real life indexing starting at t=1)
# online_K = 100
np.random.seed(123)
seed(123)
customer_stream = np.random.random((online_K, T+1))
sales_stream = np.random.random((online_K, T+1))

def save_files(storagepath, *args):
    # makedirs(storagepath)

    for o in [*args]:
        o_name = re.sub("[\[\]']", "", str(varnameis(o)))
        # print(o_name)
        with open(storagepath + "\\" + o_name + ".data", "wb") as filehandle:
            pickle.dump(o, filehandle)

def varnameis(v): d = globals(); return [k for k in d if d[k] is v]

#%%
os_dict = {tuple(v):k for k,v in enumerate(get_offer_sets_all(products))}

#%%
# online_K+1 policy iterations (starting with 0)
# values, products sold, offersets offered, capacities remaining
v_results = np.array([np.zeros(len(times))]*online_K)
p_results = np.array([np.zeros(len(times))]*online_K)
o_results = np.zeros((online_K, len(times)))  # same as above, higher skill level
c_results = np.array([np.zeros(shape=(len(times), len(capacities)))]*online_K)

value_final = pd.DataFrame(v_results[:, 0])  # setup result storage empty
capacities_final = {}
products_all = {}
offersets_all = {}

#%%
for capacities in var_capacities:
    for no_purchase_preference in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities), " - and - ", no_purchase_preference, "of", str(var_no_purchase_preferences), "starting.")

        pis_for_setting = pi_result[str(capacities)][str(no_purchase_preference)][which_policy_iteration_to_use, :, :]

        # reset the data
        v_results = np.zeros_like(v_results)
        p_results = np.zeros_like(p_results)
        c_results = np.zeros_like(c_results)
        o_results = np.zeros_like(o_results)


        for k in np.arange(online_K)+1:
            print("k: ", k)
            customer_random_stream = customer_stream[k-1]
            sales_random_stream = sales_stream[k-1]

            # line 3
            r_result = np.zeros(len(times))  # will become v_result
            c_result = np.zeros(shape=(len(times), len(capacities)), dtype=int)
            p_result = np.zeros(len(times))
            o_result = np.zeros(len(times))

            # line 5
            c = deepcopy(capacities)  # (starting capacity at time 0)

            # theta and pi for each time step
            # line 1
            thetas = 0
            pis = np.zeros(len(resources))

            for t in times:
                if any(c < 0):
                    raise ValueError

                c_result[t] = c

                # line 8-11  (adjust bid price)
                pis[c == 0] = np.inf
                pis[c > 0] = pis_for_setting[t, c > 0]

                # line 12  (epsilon greedy strategy)
                offer_set = determine_offer_tuple(pis, 0, 1, revenues, A,
                                                  arrival_probabilities, preference_weights,
                                                  no_purchase_preference)
                o_result[t] = os_dict[offer_set]

                # line 13  (simulate sales)
                # sold, customer = simulate_sales(offer_set, customer_random_stream[t], sales_random_stream[t],
                #                                 arrival_probabilities, preference_weights, no_purchase_preference)
                sold = simulate_sales(offer_set, customer_random_stream[t], sales_random_stream[t],
                                      arrival_probabilities, preference_weights, no_purchase_preference)
                p_result[t] = sold

                # line 14
                try:
                    r_result[t] = revenues[sold]
                    c -= A[:, sold]
                except IndexError:
                    # no product was sold
                    pass

                # line 16-18
            v_results[k - 1] = np.cumsum(r_result[::-1])[::-1]
            c_results[k - 1] = c_result
            p_results[k - 1] = p_result
            o_results[k - 1] = o_result

        value_final['' + str(capacities) + '-' + str(no_purchase_preference)] = pd.DataFrame(v_results[:, 0])
        capacities_final['' + str(capacities) + '-' + str(no_purchase_preference)] = pd.DataFrame(c_results[:, -1, :])
        products_all['' + str(capacities) + '-' + str(no_purchase_preference)] = p_results
        offersets_all['' + str(capacities) + '-' + str(no_purchase_preference)] = o_results



# %%
# write result of calculations
save_files(newpath, value_final, capacities_final, products_all, offersets_all)


# %%
wrapup(logfile, time_start, newpath)

