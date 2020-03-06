"""
This script will calculate the MLP regressor with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
https://www.machinelearningtutorial.net/2017/01/28/python-scikit-simple-function-approximation/

IDEE: Alles analog zu API single leg, aber jetzt theta und pi mit MLP Regressor lernen
"""


from A_data_read import *
from B_helper import *
from ast import literal_eval

from joblib import Parallel, delayed, dump, load
import multiprocessing

from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm

from sklearn.metrics import r2_score

#%%
# Setup of parameters
logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("linRegSingleLeg")
capacities = var_capacities[1]
preferences_no_purchase = var_no_purchase_preferences[0]

#%%
times = np.arange(100)
T = len(times)
K = 20
I = 50

#%%
def update_parameters(v_samples, c_samples, thetas, pis, k):
    """
    Updates the parameter theta, pi given the sample path using least squares linear regression with constraints.
    Using statsmodels

    :param v_samples:
    :param c_samples:
    :return:
    """

    # Prepare the data into statistical setting of
    # X: training data (capacity, time) in row
    # y: true outcomes (value) in row

    v = v_samples
    c = c_samples

    v_df = pd.DataFrame(v)
    v_df.columns = ["v"+str(i) for i in v_df.columns]
    v_df = v_df.stack()

    t_df = pd.DataFrame([times] * len(v))
    t_df.columns = ["t"+str(i) for i in t_df.columns]
    t_df = tidy_up(t_df)

    c_df = pd.DataFrame(c[0]).T
    for i in np.arange(len(c) - 1) + 1:
        c_df = c_df.append(pd.DataFrame(c[i]).T)
    c_df.columns = ["c"+str(i) for i in c_df.columns]
    c_df = tidy_up(c_df)
    c_df.index = t_df.index

    X = pd.concat([t_df, c_df], axis=1)
    y = v_df
    y.index = X.index


    # Note the difference in argument order
    neural_net = MLPRegressor(alpha=0.001, hidden_layer_sizes = (10,), max_iter = 50000,
                 activation = 'logistic', verbose = 'True', learning_rate = 'adaptive')
    m = neural_net.fit(X, y)

    y_true = y
    y_pred = m.predict(X)
    r2_score(y_true, y_pred)

    # Still the OLS for comparison
    model = sm.OLS(y, X).fit()
    model.params


    y_pred_ols = model.predict(X)  # make the predictions by the model
    r2_score(y_true, y_pred_ols)

    # Print out the statistics
    model.summary()

    theta_new = np.array([[i] for i in model.params[0:len(times)]])
    pi_new = np.array([[i] for i in model.params[len(times):]])

    # check exponential smoothing
    if exponential_smoothing:
        return (1 - 1 / k) * thetas + 1 / k * theta_new, (1 - 1 / k) * pis + 1 / k * pi_new
    else:
        return theta_new, pi_new


def tidy_up(t_df):
    col = t_df.columns
    ind = t_df.index
    n_obs = t_df.shape[0]
    n_time = t_df.shape[1]
    t_numpy = np.zeros((n_obs * n_time, n_time))
    for i in np.arange(n_time):
        t_numpy[i * n_obs:(i + 1) * n_obs, i] = t_df[col[i]]
    t_df = pd.DataFrame(t_numpy)
    t_df.columns = col
    t_df.index = np.tile(ind, n_time)
    return t_df


# %%
# Actual Code

pi_result = {}
theta_result = {}
value_result = {}
capacities_result = {}

# generate the random sample paths
np.random.seed(12)
random.seed(12)
customer_stream = [np.random.random(T) for _ in range(I)]
sales_stream = [np.random.random(T) for _ in range(I)]

# for capacities in var_capacities:
    value_result[str(capacities)] = {}
    capacities_result[str(capacities)] = {}
    theta_result[str(capacities)] = {}
    pi_result[str(capacities)] = {}

    # for preferences_no_purchase in var_no_purchase_preferences:
        print(capacities, "of", str(var_capacities.tolist()), " - and - ", preferences_no_purchase, "of",
              str(var_no_purchase_preferences.tolist()), "starting.")

        value_result[str(capacities)][str(preferences_no_purchase)] = {}
        capacities_result[str(capacities)][str(preferences_no_purchase)] = {}

        # store parameters over all policy iterations
        # K+1 policy iterations (starting with 0)
        # T time steps
        theta_all = np.array([[np.zeros(1)] * T] * (K + 1))
        pi_all = np.array([[np.zeros(len(resources))] * T] * (K + 1))

        # theta and pi for each time step
        # line 1
        thetas = 0
        pis = np.zeros(len(resources))

        for k in np.arange(K) + 1:
            print(k, "of", K, "starting.")
            random.seed(13)  # to have the epsilon's (exploration vs exploitation) also the same for each policy iteration k

            v_samples = np.array([np.zeros(len(times))] * I)
            c_samples = np.array([np.zeros(shape=(len(times), len(capacities)))] * I)

            for i in np.arange(I):
                # line 3
                r_sample_i = np.zeros(len(times))  # will become v_sample
                c_sample_i = np.zeros(shape=(len(times), len(capacities)), dtype=int)

                # line 5
                c = copy.deepcopy(capacities)  # (starting capacity at time 0)

                for t in times:
                    # line 7  (starting capacity at time t)
                    c_sample_i[t] = c

                    # line 8-11  (adjust bid price)
                    pis[c_sample_i[t] == 0] = np.inf
                    pis[c_sample_i[t] > 0] = pi_all[k - 1][t][c_sample_i[t] > 0]

                    # line 12  (epsilon greedy strategy)
                    offer_set = determine_offer_tuple(pis, epsilon[k], revenues, A,
                                                      arrival_probabilities, preference_weights,
                                                      preferences_no_purchase)

                    # line 13  (simulate sales)
                    sold = simulate_sales(offer_set, customer_stream[i][t], sales_stream[i][t],
                                          arrival_probabilities, preference_weights, preferences_no_purchase)

                    # line 14
                    try:
                        r_sample_i[t] = revenues[sold]
                        c -= A[:, sold]
                    except IndexError:
                        # no product was sold
                        pass

                # line 16-18
                v_samples[i] = np.cumsum(r_sample_i[::-1])[::-1]
                c_samples[i] = c_sample_i

            # line 20
            theta_all[k], pi_all[k] = update_parameters(v_samples, c_samples, theta_all[k - 1], pi_all[k - 1], k)

            value_result[str(capacities)][str(preferences_no_purchase)][k] = v_samples
            capacities_result[str(capacities)][str(preferences_no_purchase)][k] = c_samples

        theta_result[str(capacities)][str(preferences_no_purchase)] = theta_all
        pi_result[str(capacities)][str(preferences_no_purchase)] = pi_all

# %%
# write result of calculations
with open(newpath + "\\thetaAll.data", "wb") as filehandle:
    pickle.dump(theta_result, filehandle)

with open(newpath + "\\piAll.data", "wb") as filehandle:
    pickle.dump(pi_result, filehandle)

with open(newpath + "\\valueAll.data", "wb") as filehandle:
    pickle.dump(value_result, filehandle)

with open(newpath + "\\capacitiesAll.data", "wb") as filehandle:
    pickle.dump(capacities_result, filehandle)

with open(newpath + "\\thetaToUse.data", "wb") as filehandle:
    tmp = {}
    for capacities in var_capacities:
        tmp[str(capacities)] = {}
        for preferences_no_purchase in var_no_purchase_preferences:
            tmp[str(preferences_no_purchase)] = theta_result[str(capacities)][str(preferences_no_purchase)][K]
    pickle.dump(tmp, filehandle)

with open(newpath + "\\piResult.data", "wb") as filehandle:
    tmp = {}
    for capacities in var_capacities:
        tmp[str(capacities)] = {}
        for preferences_no_purchase in var_no_purchase_preferences:
            tmp[str(preferences_no_purchase)] = pi_result[str(capacities)][str(preferences_no_purchase)][K]
    pickle.dump(tmp, filehandle)

# %%
wrapup(logfile, time_start, newpath)
