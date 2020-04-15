import numpy as np
import math


def interval_partition(n, k, loss_fn):
    """
    Finds a loss-minimizing partition of the integers {0, ..., n-1} into k
    non-empty intervals. Each interval in the partition incurs a loss, and the
    goal is to minimize the total loss of all k intervals. The input `loss_fn`
    function defines the loss for intervals in the partition: for any i < j,
    the loss of an interval {i, ..., j-1} is given by `loss(i,j)`. Returns a
    numpy array of break points `b` satisfying 0 = b[0] < ... < b[k] = n+1 so
    that the optimal intervals are given by {b[i], ..., b[i+1]-1} for i in
    {0, ..., k-1}.
    """
    # loss_table[N, K] is the loss of optimally partitioning the integers
    # {0, ..., N-1} using K intervals.
    loss_table = np.empty((n+1, k+1))
    # last_break_table[N, K] stores the starting index of the final interval in
    # the optimal partitioning achieving the loss in loss_table[N, K]
    last_break_table = np.empty((n+1, k+1), np.int64)
    # solve[N, K] is set to true once we have computed the corresponding
    # entries of loss_table and last_break_table
    solved = np.zeros((n+1, k+1), np.bool)

    # Helper function that recursively applies the recurrence relations to
    # solve all subproblems
    def solve_recursively(N, K):
        # If we have already solved this subproblem, just return.
        if solved[N, K]:
            return
        # base case: if we only have one interval then there is no choice
        if K == 1:
            loss_table[N, K] = loss_fn(0, N)
            last_break_table[N, K] = 0
        # inductive case: try all possible starting points `b` for the final
        # interval and combine them with optimal partitionings of the remaining
        # prefix of integers using K-1 intervals. Remember the best one.
        else:
            loss_table[N, K] = math.inf
            for b in range(K-1, N):
                solve_recursively(b, K-1)
                loss = loss_fn(b, N) + loss_table[b, K-1]
                if loss < loss_table[N, K]:
                    loss_table[N, K] = loss
                    last_break_table[N, K] = b
        # Record that we've solved this subproblem
        solved[N, K] = True

    # Helper function to reconstruct the optimal products from the filled in
    # tables
    def reconstruct_solution(N, K):
        # If K == 1 the only interval must have started at index 0
        if K == 1:
            return [0]
        # Otherwise, we look up the last breakpoint and recurse
        else:
            # If K == k+1 the last breakpoint points past the end of the list
            if K == k+1:
                b = N
            # Otherwise, we look up the last breakpoint in the last_break_table
            else:
                b = last_break_table[N, K]
            # Given the last break, we know that the optimal solution must have
            # included the optimal partition of prefix {0, ..., b-1} using K-1
            # intervals. Find the optimal partition for that prefix and append
            # b to it.
            rest = reconstruct_solution(b, K-1)
            rest.append(b)
            return rest

    # use the recursive helper function to solve all subproblems
    solve_recursively(n, k)
    # reconstruct the optimal break points and convert to a numpy array
    optimal_breaks = np.array(reconstruct_solution(n, k+1))

    return loss_table[n, k], optimal_breaks


def regret_minimizing_products(returns, k):
    """
    Finds a set of k products that minimizes regret when each consumer is
    assigned to the highest return product that does not exceed their risk
    threshold. Takes as input a sorted numpy array of returns of length n
    containing the return for the bespoke portfolio for each consumer and the
    number k of products to choose. Returns indices of the k consumers whose
    risk limits define the regret-minimizing products, as well as the total
    regret.
    """
    # It is equivalent to find the set of k products that maximize return, and
    # slightly simpler to code. Therefore, we use interval_partition to compute
    # the return maximizing products and then slightly massage the output.

    # The following function computes the loss of an interval to be the
    # negative total return when consumers {i, ..., j-1} are assigned to a
    # product defined by consumer i's risk limit.
    def loss_fn(i, j):
        return -returns[i]*(j-i)

    # Calling interval_partition with the above loss function returns a
    # collection of break points minmizing total loss (i.e., maximizing return)
    loss, breaks = interval_partition(len(returns), k, loss_fn)

    # The regret is the total bespoke return plus the loss from
    # interval_partition
    regret = np.sum(returns) + loss

    # The indices of the optimal products are just the first k elements of the
    # breakpoints array returned by interval_partition
    products = breaks[0:k]

    return regret, products
