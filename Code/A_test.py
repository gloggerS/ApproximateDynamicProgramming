"""
This script is for entering data. It is checked and included in the "database".
"""

import numpy as np
import pandas as pd
import pickle

# # example = "singleLegFlight"
# example = "threeParallelFlights"
# # example = "example0"
# # example = "example for Greedy Heuristic"
# # example = "example parallel flights"
# # example = "efficient sets"

#%%
np.random.seed(123)

test = [np.random.random(400) for _ in range(5000)]

f = open("0-test_customer.data", "wb")
pickle.dump(test, f)
f.close()

#%%
test = [np.random.random(400) for _ in range(5000)]

f = open("0-test_sales.data", "wb")
pickle.dump(test, f)
f.close()

#%%