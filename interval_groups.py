import time
from minmax_regret_ilp import minmax_regret_ilp_wrapper
import numpy as np

from utils import check_returns_argument, check_group_arguments, \
                  get_group_regrets


def total_return(creturns, a, b):
    if a == 0:
        return creturns[b-1]
    else:
        return creturns[b-1] - creturns[a-1]


def ivl_regret(creturns, a, b, x):
    return total_return(creturns, a, b) - (b-a)*total_return(creturns, x, x+1)


def opt_products(creturns, firewall_return, n_prods, memo=None):
    if memo is None:
        memo = {}
    n_consumers = len(creturns)
    # Only solve this subproblem if it's not stored in the memo table
    if (n_consumers, n_prods) not in memo.keys():
        # If there are no consumers left, regret is zero and we use no products
        if n_consumers == 0:
            best_regret = 0
            best_prods = np.array([], dtype=int)
        # If there are no products left, everyone is assigned to the firewall
        elif n_prods <= 0:
            best_regret = creturns[-1]/n_consumers - firewall_return
            best_prods = np.array([], dtype=int)
        # Otherwise, guess the riskiest product and recurse
        else:
            best_prods = np.array([], dtype=int)
            best_regret = float("inf")
            # Try all possible highest risk products
            for x in range(n_consumers):
                # Compute the optimal products for the subproblem
                prods, regret = opt_products(creturns[0:x], firewall_return,
                                             n_prods-1, memo)
                # compute total regret over combined consumers
                regret = x*regret + ivl_regret(creturns, x, n_consumers, x)
                # divide by number of combined consumers
                regret /= n_consumers
                if regret < best_regret:
                    best_regret = regret
                    best_prods = np.concatenate((prods, [x]))
        # Store the computed solution in the memo table
        memo[(n_consumers, n_prods)] = (best_prods, best_regret)
    # Return the solution from the table
    return memo[(n_consumers, n_prods)]


def minmax_regret_interval_groups(returns, groups, num_groups, num_prods,
                                  epsilon=1e-10):
    check_returns_argument(returns)
    check_group_arguments(groups, num_groups)

    num_consumers = len(returns)
    group_starts = [np.searchsorted(groups, h) for h in range(num_groups)]
    group_starts.append(num_consumers)

    def solve_group(h, fw, max_prods, target_regret):
        memo = {}
        creturns = np.cumsum(returns[group_starts[h]:group_starts[h+1]])
        n_consumers = len(creturns)
        best_k = max_prods + 1
        best_prods = np.array([], dtype=int)
        # If we can use no products, return that solution
        if n_consumers == 0 or creturns[-1]/n_consumers - fw <= target_regret:
            return np.array([], dtype=int)
        # Otherwise, try all possible largest products
        for x in range(n_consumers):
            for k in range(1, best_k+1):
                prods, regret = opt_products(creturns[0:x], fw, k-1, memo)
                # compute total regret over combined consumers
                regret = x*regret + ivl_regret(creturns, x, n_consumers, x)
                # divide by number of combined consumers
                regret /= n_consumers
                if regret <= target_regret:
                    if k <= best_k:
                        best_k = k
                        best_prods = np.concatenate((prods, [x]))
        if best_k > max_prods:
            return None
        else:
            return best_prods + group_starts[h]

    def solve_satisfiability(target_regret):
        group_products = []
        remaining_prods = num_prods
        firewall_return = float("-inf")
        for h in range(num_groups):
            prods = solve_group(h, firewall_return, remaining_prods,
                                target_regret)
            if prods is None:
                return None  # return None if infeasible
            else:
                group_products.append(prods)
                remaining_prods -= len(prods)
                if len(prods) > 0:
                    firewall_return = returns[prods[-1]]
        return np.concatenate(group_products)

    ub = np.sum(returns) / num_consumers
    lb = 0
    while ub - lb > epsilon:
        mp = (ub + lb) / 2
        sol = solve_satisfiability(mp)
        if sol is None:
            lb = mp
        else:
            ub = mp

    prods = solve_satisfiability(ub)
    regrets = get_group_regrets(returns, groups, num_groups, prods)
    max_regret = np.max(regrets)

    return max_regret, regrets, prods


# --- Testing ---


def test_random(n_consumers, n_groups, n_prods):
    returns = np.sort(np.random.rand(n_consumers))
    groups = np.sort(np.random.randint(0, n_groups, n_consumers))

    dp_tic = time.time()
    dp_mr, dp_rs, dp_ps = minmax_regret_interval_groups(
        returns, groups, n_groups, n_prods)
    dp_time = time.time() - dp_tic
    print("dp:")
    print(f"  runtime = {dp_time} sec")
    print(f"  prods = {dp_ps}")
    print(f"  max_regret = {dp_mr}")
    print(f"  all regrets = {dp_rs}")

    ilp_tic = time.time()
    ilp_mr, ilp_rs, ilp_ps = minmax_regret_ilp_wrapper(
        returns, groups, n_groups, n_prods)
    ilp_time = time.time() - ilp_tic

    print("ILP:")
    print(f"  runtime = {ilp_time} sec")
    print(f"  prods = {ilp_ps}")
    print(f"  max_regret = {ilp_mr}")
    print(f"  all regrets = {ilp_rs}")


if __name__ == "__main__":
    for i in range(20):
        print("--------")
        test_random(50, 5, 5)
