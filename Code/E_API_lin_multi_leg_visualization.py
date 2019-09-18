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
    ax.plot(np.arange(K)+1, p[i], label="Product "+str(i+1), linewidth=2.0)
ax.plot(np.arange(K)+1, p[i+1], label="No Purchase", linewidth=2.0)
ax.legend(bbox_to_anchor=(0, -.3, 1, .102), loc="lower left",
          ncol=3, mode="expand")
plt.xticks(np.append([1], np.arange(5, K+1, 5)))
fig.savefig(result_folder+"\\plotProducts1.pdf", bbox_inches="tight")
plt.show()


#%%
for i in products:
    plt.bar(np.arange(K)+1, p[i], label="Product "+str(i+1))
plt.legend(bbox_to_anchor=(0, -.23, 1, .102), loc="lower left",
          ncol=3, mode="expand")
plt.xticks(np.append([1], np.arange(5, K + 1, 5)))
plt.savefig(result_folder+"\\plotProducts2.pdf", bbox_inches="tight")
plt.show()

#%% values at start
v = pd.DataFrame(value_result[str(capacities)][str(preferences_no_purchase)][:, :, 0])
v = v.iloc[1:, :]
    
#%%
v.apply(sum, axis=1)

np.mean(v, axis=1)

plt.plot(np.arange(K)+1, np.mean(v, axis=1), label="Average value at start")
plt.legend(loc="lower right")
plt.xticks(np.append([1], np.arange(5, K+1, 5)))
plt.savefig(result_folder+"\\plotValue.pdf", bbox_inches="tight")
plt.show()

#%% time sold out (not needed yet, as never sold out in exampleStefan)
with open(result_folder+"\\soldOut.data", "rb") as filehandle:
    sold_out = pickle.load(filehandle)


#%% capacities remaining
c = capacities_result[str(capacities)][str(preferences_no_purchase)]

# TODO change to the above
# capacity_last_t = c[:, :, -1, :]
capacity_last_t = c[:, :, -2, :]

capacity_last_t_average = capacity_last_t.sum(axis=1)
capacity_last_t_average = capacity_last_t_average[1:, :]

#%%
def plot_remaining_capacity(capacity_last_t_average, ks):
    bar_width = 1 / (capacity_last_t_average.shape[-1] + 1)
    r = 1.0 * np.arange(len(ks))
    plt.bar(r, capacity_last_t_average[ks, 0], label="Resource 1", width=bar_width, edgecolor="white")
    for h in resources[1:]:
        r += bar_width
        plt.bar(r, capacity_last_t_average[ks, h], label="Resource " + str(h + 1), width=bar_width, edgecolor="white")
    plt.xticks(r - 1.5 * bar_width, ks + 1)
    plt.ylim([0, 1.1*capacity_last_t_average.max()])
    plt.legend(bbox_to_anchor=(0, -.16, 1, .102), loc="lower left",
               ncol=4, mode="expand")
    plt.savefig(result_folder + "\\remainingCapacity" + str(ks[0]) + ".pdf", bbox_inches="tight")
    plt.show()

#%%
ks = np.arange(0, 10)
plot_remaining_capacity(capacity_last_t_average, ks)

#%%
ks = np.arange(50, 60)
plot_remaining_capacity(capacity_last_t_average, ks)

#%% offersets
with open(result_folder+"\\offersetsOffered.data", "rb") as filehandle:
    offer_sets = pickle.load(filehandle)

label_relevant = [str((products+1)[a==1]) for a in get_offer_sets_all(products)[offer_sets.sum(axis=0)>0]]

offer_sets_relevant = offer_sets[1:, offer_sets.sum(axis=0)>0]

df = pd.DataFrame(offer_sets_relevant, columns=label_relevant, index=(np.arange(K)+1), dtype=int)

erg_latex = open(result_folder+"\\offersetsOffered.txt", "w+")  # write and create (if not there)
print(df.to_latex(), file=erg_latex)
erg_latex.close()

