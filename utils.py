import numpy as np


def get_group_regrets(returns, groups, num_groups, products,
                      use_avg_regret=True):
    """
    Computes the total or average regret of each group for a collection of
    products. The `returns` vector contains the consumer returns sorted in
    non-decreasing order, the `groups` vector gives the group index for each
    consumer in `range(num_groups)`, and the `products` vector contains the
    indices of the consumers defining each product. I.e., the return of the
    `i`th product is `returns[products[i]]`. The product indices must be in
    non-decreasing order. If `use_avg_regret=True` then returns the average
    regret for each group, otherwise the total regret.
    """
    check_group_arguments(groups, num_groups)
    check_products_argument(returns, products)
    group_regrets = np.zeros(num_groups)
    group_sizes = np.zeros(num_groups)
    p_ix = len(products)-1
    for c_ix in range(len(returns)-1, -1, -1):
        g_ix = groups[c_ix]
        group_sizes[g_ix] += 1
        while returns[products[p_ix]] > returns[c_ix]:
            p_ix -= 1
        group_regrets[g_ix] += returns[c_ix] - returns[products[p_ix]]
    if use_avg_regret:
        group_regrets /= group_sizes
    return group_regrets


# --- Type Checking ---

def check_returns_argument(returns):
    """
    Asserts that returns is a numpy array sorted in non-decreasing order.
    """
    assert type(returns) is np.ndarray, "returns must be a numpy array"
    # np.diff computes the vector [returns[i+1] - returns[i] for i]. The returns
    # vector is in sorted order only when all of these differences are >= 0.
    assert all(np.diff(returns) >= 0), \
        "returns must be sorted in non-decreasing order"


def check_group_arguments(groups, num_groups):
    """
    Asserts that groups is an integer numpy array with entries in
    {0,...,num_groups - 1}.
    """
    assert type(groups) is np.ndarray and groups.dtype == int, \
        "groups must be a numpy array with dtype == int"
    assert all([g in range(num_groups) for g in groups]), \
        "group indices must be in {0, ..., num_groups-1}."


def check_products_argument(returns, products):
    """
    Asserts that products is an integer numpy array with non-decreasing entries
    in {0, ..., len(returns)-1}.
    """
    assert type(products) is np.ndarray and products.dtype == int, \
        "products must be a numpy array with dtype == int"
    assert all([p in range(len(returns)) for p in products]), \
        "product indices must be in {0, ..., len(returns)-1}"
    assert all(np.diff(products) >= 0), \
        "product indices must be sorted in non-decreasing order"
