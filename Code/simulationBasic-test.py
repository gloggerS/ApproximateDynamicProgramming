from simulationBasic import *


#%% Test - history
dfResult = history(numPeriods, arrivalProbability, capacity, offer_set, revenues)

x = -dfResult.index
y = np.cumsum(dfResult['revenue'])
plt.plot(x, y)

#%% Test - customer weight
probs = customer_choice_individual(offer_set)

#%% Test - value expected
start_time = time.time()
print(value_expected(3, 3))
print(time.time() - start_time)
