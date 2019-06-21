from simulationBasic import *

#%% Test - sample_path
dfResult = sample_path(T, arrival_probability, capacity, revenues)

x = dfResult.index
y = np.cumsum(dfResult['revenue'])
plt.plot(x, y)
plt.show()

#%% Test - customer weight
probs = customer_choice_individual(offer_set)
probs

#%% Test - value expected
start_time = time.time()
print(value_expected(40, 0))
print(time.time() - start_time)


