"""
This script visualizes a given multi_leg_visualization
"""

from B_helper import *
from mpl_toolkits.mplot3d import Axes3D

#%%
result_folder = "C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\exampleStefan-True-ES-MultiLeg-190827-1516"
no_purchase_preference = np.array([2, 2, 1, 2])
capacity_max = np.array([8, 4, 4, 8])
#%%
with open(result_folder+"\\totalresults.data", "rb") as file:
    data_raw = pickle.load(file)

#%%
data_raw = data_raw[str(no_purchase_preference)]
data = pd.DataFrame(data_raw[0]).assign(t=0)

for t in (np.array(range(len(data_raw)-1))+1):
    data = data.append(pd.DataFrame(data_raw[t]).assign(t=t))

data = data.set_index("t", append=True)
for i in np.array(range(len(data.index.names)-1))[::-1]:
    data = data.swaplevel(i+1, i)

data = data["value"]

#%%
idx = pd.IndexSlice
df = data.loc[idx[:, :, capacity_max[1], capacity_max[2], capacity_max[3]]]
df = df.reset_index()



#%%
df = data.loc[(slice(None), slice(None), capacity_max[1], capacity_max[2], capacity_max[3]), "value"]
df = df.reset_index()
df = df.loc[:, ["t", "c0", "value"]]

#%%
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot_trisurf(df["t"], df["c0"], df["value"], cmap=plt.cm.viridis)
plt.show()
