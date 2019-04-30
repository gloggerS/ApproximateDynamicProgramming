def print_top(df, p=0.9):
    df2 = df.loc[df.loc[:, "val"] > p * max(df.loc[:, "val"]), :]
    return df2.sort_values(["val"], ascending=False)

capacities_remaining = np.array([0,0,1])
pi = np.array([0, 1134.55, 500])
t = 27
beta = 1

dataName = "example0"

resources, \
products, revenues, A, \
customer_segments, preference_weights, arrival_probabilities, \
times = get_data_without_variations(dataName)
lam = sum(arrival_probabilities)

val_akt = 0
index_max = 0

offer_sets_all = get_offer_sets_all(products)
offer_sets_all = pd.DataFrame(offer_sets_all)
df2 = pd.DataFrame({"purchase_rates":[[0]]*offer_sets_all.__len__()})
displacement_costs = displacement_costs_vector(capacities_remaining, preference_no_purchase, t + 1, pi, beta)

for index, offer_array in offer_sets_all.iterrows():
    # check if it makes sense to consider the offer_array
    # it doesn't make sense, if product cannot be sold due to capacity reasons
    alright = True
    for j in products:
        if offer_array[j] > 0 and any(capacities_remaining - A[:, j] < 0):
            alright = False

    if alright:
        val_new = 0.0
        purchase_rate = purchase_rate_vector(tuple(offer_array), preference_weights,
                                             preference_no_purchase, arrival_probabilities)
        for j in products:
            if offer_array[j] > 0:
                val_new += purchase_rate[j] * \
                           (revenues[j] - sum(displacement_costs * A[:, j]))
        val_new = lam * val_new

        if val_new > val_akt:
            index_max = copy.copy(index)
            val_akt = copy.deepcopy(val_new)

        offer_sets_all.loc[index, "val"] = val_new
        df2.loc[index, "purchase_rates"] = [purchase_rate]
    else:
        offer_sets_all.loc[index, "val"] = 0.0


# offer_sets_all["purchase_rates"] = df2["purchase_rates"]

q = print_top(offer_sets_all)
q
