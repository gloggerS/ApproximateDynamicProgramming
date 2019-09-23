"""
This script compares various policies
"""

from B_helper import *
import seaborn as sns

#%%
res_folder_ES = "C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\exampleStefan-True-ES-evaluation-190923-1012"
res_folder_CDLP = "C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\exampleStefan-True-CDLP-evaluation-190923-1210"
res_folder_API_last = "C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\exampleStefan-True-APILinearMultiLeg-evaluation-190923-1415"
res_folder_API_secondlast = "C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\exampleStefan-True-APILinearMultiLeg-2-evaluation-190923-1712"

res_folders = [res_folder_ES, res_folder_CDLP, res_folder_API_last, res_folder_API_secondlast]
labels = ["ES", "CDLP", "API60", "API59"]

#%%
# ensure that all setting files are the same and we have the setting parameters at hand
a = pd.read_csv("0_settings.csv", delimiter="\t", header=None)
for f in res_folders:
    b = pd.read_csv(f+"\\0_settings.csv", delimiter="\t", header=None)
    if not (a.equals(b)):
        raise ValueError("SettingFiles don't coincide.")

logfile, newpath, var_capacities, var_no_purchase_preferences, resources, products, revenues, A, \
        customer_segments, preference_weights, arrival_probabilities, times, T, time_start,\
        epsilon, exponential_smoothing,\
        K, online_K, I \
    = setup_testing("ALL-comparison")

capacities = var_capacities[0]
no_purchase_preference = var_no_purchase_preferences[0]

#%% general formulas
def get_all_values(dat_name):
    ret = {}
    for i in np.arange(len(labels)):
        with open(res_folders[i] + "\\" + dat_name + ".data", "rb") as filehandle:
            df = pickle.load(filehandle)
            df = df.drop(0, axis=1)
            df.columns = [labels[i]]
        ret[i] = df

    dat = ret[0]
    for i in np.arange(len(labels) - 1) + 1:
        dat[ret[i].columns[0]] = ret[i]

    return dat

def get_all_dict(dat_name, capacities, no_purchase_preference):
    ret = {}
    for i in np.arange(len(labels)):
        with open(res_folders[i] + "\\" + dat_name + ".data", "rb") as filehandle:
            df = pickle.load(filehandle)
            df = df['' + str(capacities) + '-' + str(no_purchase_preference)]
        ret[labels[i]] = df

    return ret



# %% Create graphs
#%% values
v = get_all_values("value_final")

#%%
v_plot = pd.melt(v, var_name="Policy", value_name="Value")
sns.boxplot(x=v_plot.Policy, y=v_plot.Value)
plt.savefig(newpath + "\\" + "values-Boxplot" +".pdf", bbox_inches="tight")
plt.show()

#%%
sns.violinplot(x=v_plot.Policy, y=v_plot.Value)
plt.savefig(newpath + "\\" + "values-Violinplot" +".pdf", bbox_inches="tight")
plt.show()


#%%
erg_latex = open(newpath+"\\values-Summary.txt", "w+")  # write and create (if not there)
print(np.round(v.describe(), 2).to_latex(), file=erg_latex)
erg_latex.close()
np.round(v.describe(), 2)

#%%
# rows = ["Mean", "Max", "Min", "sd"]
# rows_function = [np.mean, np.max, np.min, np.std]
# columns = [value_final.columns[1]]
# df_value_summary = pd.DataFrame(np.zeros((len(rows), len(columns))), index=rows, columns=columns)
#
# for col in columns:
#     for i in np.arange(len(rows)):
#         df_value_summary.loc[rows[i], col] = rows_function[i](value_final.loc[:, col])
#
#
# df_value_summary

#%% products
p = get_all_dict("products_all", capacities, no_purchase_preference)

label_product = ["No Purchase", *(products+1)]

p_all = []
for i in np.arange(len(labels)):
    a = p[labels[i]]
    dat = pd.DataFrame(labels[i], index=np.arange(len(a)), columns=["Policy"])
    for j in np.arange(len(products) + 1):
        dat[j] = np.sum(a == j, axis=1)
    dat = dat.melt(id_vars=["Policy"], var_name="Product", value_name="Purchases")
    p_all.append(dat)
p_plot = pd.concat(p_all)


order = [len(products), *products]

fig, ax = plt.subplots()
sns.barplot(x="Product", y="Purchases", hue="Policy", data=p_plot, order=order)
ax.set_xticklabels(label_product)
plt.savefig(newpath + "\\" + "products-Barplot" +".pdf", bbox_inches="tight")
plt.show()


#%% products
# # https://matplotlib.org/3.1.1/gallery/statistics/barchart_demo.html#sphx-glr-gallery-statistics-barchart-demo-py
# p = products_all['' + str(capacities) + '-' + str(preferences_no_purchase)]
#
# p_plot = np.zeros(len(products)+1)
# for i in np.arange(len(products)+1):
#     p_plot[i] = sum(sum(p==i))
#
# label = ["No Purchase", *["Prod. "+str(i+1) for i in products]]
# x = np.arange(len(products) + 1)
# fig, ax = plt.subplots()
# ax.bar(x, p_plot)
# ax.yaxis.grid(True, linestyle="--", color="grey", alpha=.25)
# ax.set_xticks(x)
# ax.set_xticklabels(label)
# plt.show()

#%% offersets
o = get_all_dict("offersets_all", capacities, no_purchase_preference)

o_all = []
for i in np.arange(len(labels)):
    a = o[labels[i]]
    dat = pd.DataFrame(labels[i], index=np.arange(len(a)), columns=["Policy"])
    for j in np.arange(len(get_offer_sets_all(products)) + 1):
        dat[j] = np.sum(a == j, axis=1)
    o_all.append(dat)
o_plot = pd.concat(o_all)

#%%
o_plot = o_plot.set_index("Policy")
offer_sets_relevant = o_plot.columns[np.sum(o_plot, axis=0) > 0]
o_plot = o_plot.loc[:, offer_sets_relevant]
o_plot["Policy"] = o_plot.index

o_plot = o_plot.melt(id_vars="Policy", var_name="Offerset", value_name="Number of times offered")

# fig, ax = plt.subplots()
sns.barplot(x="Offerset", y="Number of times offered", hue="Policy", data=o_plot)
# ax.set_xticklabels(label_product)
plt.savefig(newpath + "\\" + "offersets-Barplot" +".pdf", bbox_inches="tight")
plt.show()

#%%
os_dict_reverse = {k:tuple(v) for k,v in enumerate(get_offer_sets_all(products))}

o_table = pd.DataFrame(offer_sets_relevant, columns=["Offerset Index"])
o_table["Offerset Tuple"] = ""
o_table["Offerset Items"] = ""
for i in o_table.index:
    tup = os_dict_reverse[o_table.loc[i, "Offerset Index"]]
    o_table.loc[i, "Offerset Tuple"] = tup
    o_table.loc[i, "Offerset Items"] = re.sub("\]", "}", re.sub("\[", "{", re.sub(" ", ",", str(products[np.array(tup)==1]+1))))


erg_latex = open(newpath+"\\offersets-Overview.txt", "w+")  # write and create (if not there)
print(o_table.to_latex(index=False), file=erg_latex)
erg_latex.close()
o_table

#%% remaining capacity left over
r = get_all_dict("capacities_final", capacities, no_purchase_preference)

r_all = []
for i in np.arange(len(labels)):
    a = r[labels[i]]
    dat = pd.DataFrame(labels[i], index=np.arange(len(a)), columns=["Policy"])
    for i in a.columns:
        dat[i] = a[i]
    r_all.append(dat)
r_plot = pd.concat(r_all)

r_plot = r_plot.melt(id_vars="Policy", var_name="Resource", value_name="Capacity left over")

#%%
label_resources = resources+1


fig, ax = plt.subplots()
sns.barplot(x="Resource", y="Capacity left over", hue="Policy", data=r_plot)
ax.set_xticklabels(label_resources)
plt.savefig(newpath + "\\" + "resources-Barplot" +".pdf", bbox_inches="tight")
plt.show()

#%%
wrapup(logfile, time_start, newpath)


