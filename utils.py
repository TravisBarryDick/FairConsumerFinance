import numpy as np


def get_group_regrets(returns, groups, num_groups, products,
                      use_avg_regret=True):
    """
    Computes the total or average regret of each group for a collection of
    products. The `returns` vector contains the consumer returns sorted in
    non-decreasing order, the `groups` vector gives the group index for each
    consumer in `range(num_groups)`, and the `products` vector contains the
    indices of the consumers defining each product. I.e., the return of the
    `i`th product is `returns[products[i]]`. If `use_avg_regret=True` then
    returns the average regret for each group, otherwise the total regret.
    """

    group_regrets = np.zeros(num_groups)
    group_sizes = np.zeros(num_groups)
    p_ix = len(products)-1
    for c_ix in range(len(returns)-1, 0, -1):
        g_ix = groups[c_ix]
        group_sizes[g_ix] += 1
        while returns[products[p_ix]] > returns[c_ix]:
            p_ix -= 1
        group_regrets[g_ix] += returns[c_ix] - returns[products[p_ix]]
    if use_avg_regret:
        group_regrets /= group_sizes
    return group_regrets
