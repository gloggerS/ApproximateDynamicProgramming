n = 6
products = np.arange(n)
revenues = np.array([400, 800, 500, 1000, 300, 600])

T = 300
times = np.arange(T)

L = 4
customer_segments = np.arange(L)
arrival_probabilities = np.array([0.1, 0.15, 0.2, 0.05])
preference_weights = np.array([[0, 5, 0, 10, 0, 1],
                               [5, 0, 1, 0, 10, 0],
                               [10, 8, 6, 4, 3, 1],
                               [8, 10, 4, 6, 1, 3]])

var_no_purchase_preferences = np.array([[1, 5, 5, 1],
                                        [1, 10, 5, 1],
                                        [5, 20, 10, 5]])
preference_no_purchase = var_no_purchase_preferences[0]

m = 3
resources = np.arange(m)

base_capacity = np.array([30, 50, 40])
delta = np.arange(0.4, 1.21, 0.2)
var_capacities = np.zeros((len(delta), len(base_capacity)))
for i in np.arange(len(delta)):
    var_capacities[i] = delta[i]*base_capacity
capacities = var_capacities[0]

# capacity demand matrix A (rows: resources, cols: products)
# a_ij = 1 if resource i is used by product j
A = np.array([[1, 1, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1]])