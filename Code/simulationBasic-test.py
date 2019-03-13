from simulationBasic import *

# %% OVERALL PARAMETERS
numProducts = 4
products = np.arange(numProducts) + 1  # only real products (starting from 1)
revenues = np.array([1000, 800, 600, 400])  # only real products

numPeriods = 10

customer_segments_num = 1
arrivalProbability = 0.8
preference_weights = np.array([0.4, 0.8, 1.2, 1.6])

varNoPurchasePreferences = np.array([1, 2, 3])
varCapacity = np.arange(40, 120, 20)
# %% LOCAL PARAMETERS


# %% JUST TEMPORARY
preference_no_purchase = 2
capacity = 6
offer_set = np.array([1, 0, 1, 1])


#%% Test - sample_path
dfResult = sample_path(numPeriods, arrivalProbability, capacity, offer_set, revenues)

x = -dfResult.index
y = np.cumsum(dfResult['revenue'])
plt.plot(x, y)

#%% Test - customer weight
probs = customer_choice_individual(offer_set)

#%% Test - value expected
start_time = time.time()
print(value_expected(3, 3))
print(time.time() - start_time)

#%% Figure 2 aus Koch
capacities = np.arange(24)+1

value_exact = np.zeros_like(capacities)
for capacity in capacities:
    value_exact[capacity-1] = value_expected(capacity, 39)[0]

plt.close()
plt.plot(capacities, value_exact)
plt.show()

#%% Ergebnis Tabelle 1 aus Koch
value_expected(40, 399)