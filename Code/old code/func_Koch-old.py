
def arrival(num_periods, arrival_probability):
    """
    Calculates the sample path.

    :param num_periods:
    :param arrival_probability:
    :return: Vector with arrival of customers.
    """
    return bernoulli.rvs(size=num_periods, p=arrival_probability)


def sample_path(num_periods, arrival_probability, capacity, revenues):
    """

    Over one complete booking horizon with *num_period* periods and a total *capacity*, the selling sample_path is
    recorded. A customer comes with *arrival_probability* and has given *preferenceWeights* and *noPurchasePreferences*.

    Helpers
    ----
    customerArrived :
        the ID of the customer segment that has arrived (used for customer preferences later on)

    :param num_periods:
    :param arrival_probability:
    :param capacity:
    :param revenues:
    :return: data frame with columns (time, capacity (at start), customer arrived, product sold, revenue)
    *customerArrived*: ID of
    *customerPreferences*: for each customer segment stores the preferences to determine which product will be bought
    """

    index = np.arange(num_periods + 1)  # first row is a dummy (for nice for loop)
    columns = ['capacityStart', 'customerArrived', 'productSold', 'revenue', 'capacityEnd']

    df_sample_path = pd.DataFrame(index=index, columns=columns)
    df_sample_path = df_sample_path.fillna(0)

    df_sample_path.loc[0, 'capacityStart'] = df_sample_path.loc[0, 'capacityEnd'] = capacity
    df_sample_path.loc[1:num_periods, 'customerArrived'] = arrival(num_periods, arrival_probability)

    revenues_with_no_purchase = np.insert(revenues, 0, 0)
    products_with_no_purchase = np.arange(revenues_with_no_purchase.size)

    for t in np.delete(index, 0):  # start in second row (without actually deleting row)
        if df_sample_path.loc[t, 'customerArrived'] == 1:
            if df_sample_path.loc[t - 1, 'capacityEnd'] == 0:
                break
            # A customer has arrived and we have capacity.

            df_sample_path.loc[t, 'capacityStart'] = df_sample_path.loc[t - 1, 'capacityEnd']

            offer_set_tuple = value_expected(df_sample_path.loc[t, 'capacityStart'], t)[1]
            customer_probabilities = customer_choice_individual(offer_set_tuple)

            df_sample_path.loc[t, 'productSold'] = np.random.choice(products_with_no_purchase, size=1,
                                                                    p=customer_probabilities)

            df_sample_path.loc[t, 'revenue'] = revenues_with_no_purchase[df_sample_path.loc[t, 'productSold']]

            if df_sample_path.loc[t, 'productSold'] != 0:
                df_sample_path.loc[t, 'capacityEnd'] = df_sample_path.loc[t, 'capacityStart'] - 1
            else:
                df_sample_path.loc[t, 'capacityEnd'] = df_sample_path.loc[t, 'capacityStart']
        else:
            # no customer arrived
            df_sample_path.loc[t, 'capacityEnd'] = df_sample_path.loc[t, 'capacityStart'] = \
                df_sample_path.loc[t - 1, 'capacityEnd']

    return df_sample_path
