from simulationBasic import *

#%%

capacities = np.arange(24)+1

value_exact = np.zeros_like(capacities)
for capacity in capacities:
    value_exact[capacity-1] = value_expected(capacity, 39)[0]

#%%
plt.close()
plt.plot(capacities, value_exact)

#%%
