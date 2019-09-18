"""
This script will calculate the Approximate Policy Iteration with linear value function and store it in a large dataframe
as specified in 0_settings.csv.
"""

from B_helper import *

#%%
result_folder = "C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\exampleStefan-True-APILinearMultiLeg-190917-1341"

#%%
a = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
b = pd.read_csv(result_folder+"\\0_settings.csv", delimiter="\t", header=None)
if not(a.equals(b)):
    raise ValueError("SettingFiles don't coincide.")

logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("APILinearMultiLeg-Visualization")

capacities = var_capacities[0]
preferences_no_purchase = var_no_purchase_preferences[0]


# %%
# write result of calculations
with open(result_folder+"\\thetaAll.data", "rb") as filehandle:
    theta_result = pickle.load(filehandle)

with open(result_folder+"\\piAll.data", "rb") as filehandle:
    pi_result = pickle.load(filehandle)

with open(result_folder+"\\valueAll.data", "rb") as filehandle:
    value_result = pickle.load(filehandle)

with open(result_folder+"\\capacitiesAll.data", "rb") as filehandle:
    capacities_result = pickle.load(filehandle)

with open(result_folder+"\\thetaToUse.data", "rb") as filehandle:
     theta_to_use = pickle.load(filehandle)

with open(result_folder+"\\piToUse.data", "rb") as filehandle:
    pi_to_use = pickle.load(filehandle)


#%% products sold
with open(result_folder+"\\plotProducts.data", "rb") as filehandle:
    p = pickle.load(filehandle)
    
#%%
fig, ax = plt.subplots()
for i in products:
    ax.plot(np.arange(K)+1, p[i], label="Product "+str(i+1))
ax.plot(np.arange(K)+1, p[i+1], label="No Purchase")
ax.legend(bbox_to_anchor=(0, -.35, 1, .102), loc="lower left",
          ncol=3, mode="expand")
plt.xticks(np.append([1], np.arange(5, K+1, 5)))
# fig.savefig(result_folder+"\\plotProducts.pdf", bbox_inches="tight")
plt.show()

    
#%% values at start
with open(result_folder+"\\plotValues.data", "rb") as filehandle:
    v = pickle.load(filehandle)
    
#%%
v.apply(sum, axis=1)

np.mean(v, axis=1)

plt.plot(np.arange(K)+1, np.mean(v, axis=1), label="Average value at start")
plt.legend(loc="lower right")
plt.xticks(np.append([1], np.arange(5, K+1, 5)))
plt.savefig(result_folder+"\\plotValue.pdf", bbox_inches="tight")
plt.show()

