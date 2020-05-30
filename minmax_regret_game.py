import numpy as np

from exponential_weights import ExponentialWeights
from regret_minimization_dp import weighted_regret_minimizing_products
from utils import get_group_regrets, check_returns_argument, \
                  check_group_arguments


def minmax_regret_game(returns, groups, num_groups, num_prods, T,
                       use_avg_regret=True, step_size=None):
    """
    The inputs `returns`, `groups`, and `num_groups` describe a collection of
    consumers, where `num_groups` is the number of groups and, for consumer `i`,
    `returns[i]` is the return of their bespoke portfolio, `groups[i]` their
    group index in `range(num_groups)`. It is assumed that consumers are sorted
    by returns so that `returns[i] <= returns[i+1]` for all i. If `step_size`
    is provided, that step size is used, otherwise the `step_size` is computed
    from a theoretical bound (and may lead to very slow convergence).

    Simulates a two-player game between:
    1. An auditor that assigns weights to each group to maximize total or
       average group-weighted regret. (ExponentialWeights)
    2. A product player that chooses products to minimize the total or average
       group-weighted regret. (Best Response)

    If `use_avg_regret` is set to `True`, then the regret of each group is
    computed as the average regret of its members. Otherwise, the total regret
    is used.

    The `num_prods` products chosen by the product player on each round are the
    indices of `num_prods` consumers (who define the returns of the
    corresponding products).

    Returns three arrays:
    1. `products`, where `products[t,:]` contain the products chosen by the
       product player on round `t` of the game.
    2. `group_weights`, where `group_weights[t,:]` contain the group weights
       chosen by the auditor on round `t` of the game.
    3. `group_regrets`, where `group_regrets[t,i]` contains the total or
        average regret for consumers in group `i` using `products[t,:]`.

    The uniform distributions over the rows of `products` and `group_weights`
    provide an approximate minmax equilibrium for the game. In particular, the
    uniform distribution over the rows of `products` gives a mixed set of
    products that minimizes expected total or average regret of the most
    regreftul group for a set of `num_prods` products sampled from the
    distribution.
    """
    check_returns_argument(returns)
    check_group_arguments(groups, num_groups)
    num_consumers = len(returns)
    group_sizes = np.array([sum(groups == i) for i in range(num_groups)])

    # Helper function that converts weights over the groups to weights over the
    # consumers. Each consumer's weight is a copy of their group's weight.
    def group_to_consumer_weights(group_weights):
        weights = np.zeros(num_consumers)
        for g in range(0, num_groups):
            weights[groups == g] = group_weights[g]
            if use_avg_regret and group_sizes[g] > 0:
                weights[groups == g] /= group_sizes[g]
        return weights

    # If the step_size wasn't specified, try to compute a reasonable one from
    # the theoretical bound
    if step_size is None:
        max_group_regrets = np.array([np.sum(returns[groups == i])
                                      for i in range(num_groups)])
        if use_avg_regret:
            max_group_regrets / group_sizes
        max_group_regret = np.max(max_group_regrets)
        step_size = np.sqrt(8 * np.log(num_groups) / T) / max_group_regret
        print(f"No step_size specified, using step_size = {step_size}")

    group_player = ExponentialWeights(num_groups, step_size)

    # Allocate numpy arrays to store the products, weights, and regrets from
    # each round of the game.
    products = np.zeros((T, num_prods), dtype=np.int)
    group_weights = np.zeros((T, num_groups))
    group_regrets = np.zeros((T, num_groups))

    # Simulate the game
    for t in range(T):
        group_weights[t, :] = group_player.get_distribution()
        consumer_weights = group_to_consumer_weights(group_weights[t, :])
        products[t, :] = weighted_regret_minimizing_products(
            returns, consumer_weights, num_prods)[1]
        group_regrets[t, :] = get_group_regrets(returns, groups, num_groups,
                                                products[t, :],
                                                use_avg_regret=use_avg_regret)
        # Note: We negate the vector group_regrets[t, :] because the group
        # player is trying to maximize weighted-group regret, so high group
        # regret corresponds to low loss for the group player.
        group_player.update(-group_regrets[t, :])

    return products, group_weights, group_regrets
