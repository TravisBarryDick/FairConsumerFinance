import numpy as np
import math
import os
import sys
from numba import jit
import numba
os.chdir('/Users/Hadi/Dropbox/Research/Fair Consumer Finance/python_code/Hadi')
sys.path.append(os.path.abspath('../'))
from utils import check_returns_argument


@jit(nopython=True)
#def regret_minimizing_products_jit(returns: numba.types.float64[:],
#                                   num_prods: int,
#                                   weights: numba.types.float64[:]):
def regret_minimizing_products_jit(returns,
                                       num_prods,
                                       weights):
    num_consumers = len(returns)

    cweights = np.cumsum(weights)

    # subprob_return[N,k] is the optimal return for consumers {0, ..., N-1} with
    # k products. The index k = 0 is not used.
    subprob_return = np.zeros((num_consumers, num_prods + 1))

    # subprob_product[N,k] is the last product used in the optimal solution for
    # consumers {0, ..., N-1} with k products. The index k = 0 is not used
    subprob_product = np.empty((num_consumers, num_prods + 1), dtype=np.uint16)

    # Base Case:
    for N in range(num_consumers):
        subprob_return[N, 1] = returns[0] * cweights[N]
        subprob_product[N, 1] = 0

    # Inductive Case
    for N in range(num_consumers):  # n = 0, ..., n_consumers-1
        for k in range(2, num_prods + 1):  # k = 1, ..., num_prods
            subprob_return[N, k] = 0
            for z in range(k - 1, N + 1):
                z_return = returns[z] * (cweights[N] - cweights[z - 1])
                total_return = z_return + subprob_return[z - 1, k - 1]
                if total_return >= subprob_return[N, k]:
                    subprob_return[N, k] = total_return
                    subprob_product[N, k] = z

    # Get optimal products
    opt_products = np.empty(num_prods, dtype=np.uint16)
    subprob_k = num_prods
    subprob_N = num_consumers - 1
    for k in range(num_prods - 1, -1, -1):
        opt_products[k] = subprob_product[subprob_N, subprob_k]
        subprob_k -= 1
        subprob_N = opt_products[k] - 1

    regret = np.sum(returns * weights) - \
             subprob_return[num_consumers - 1, num_prods]

    return regret, opt_products
print(regret_minimizing_products_jit.inspect_types())

def interval_partition(n, k, loss_fn,use_iterative = False,returns=None, weights = None):

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
        #print(last_break_table)
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
    """
    def reconstruct_solution_it(k,n):
        breaks = np.empty(k+1)
        breaks[k] = n
        breaks[k-1] = last_break_table[k-1,n-1]
        print(breaks)
        print(last_break_table)
        for K in range(k-2,-1,-1):
            idx = int(last_break_table[K,int(breaks[int(K+1)])-1])
            breaks[K] = last_break_table[K, idx]
        return(breaks)"""
    def reconstruct_solution_it(k,n):
        breaks = np.empty(k+1,dtype=np.uint16)
        #breaks = np.empty(k, dtype=np.uint16)
        breaks[k] = n
        breaks[k-1] = last_break_table[k-1,n-1]
        print(breaks)
        old_k = k-1
        old_n = n
        for K in range(k-2,-1,-1):
            k_this_problem = old_k - 1
            N_this_problem = last_break_table[old_k-1, old_n-1]
            #print(k_this_problem,N_this_problem)
            breaks[K] = last_break_table[k_this_problem-1,N_this_problem-1]
            old_k = k_this_problem
            old_n = N_this_problem
        """
        for K in range(k-2,-1,-1):
            print(K)
            old_n = breaks[K+1]

            n_remaining = last_break_table[(K+1)-1,old_n-1]-1
            #n_remaining = old_n-1
            #n_remaining = int(last_break_table[K-1,int(breaks[int(K)+1])])
            print(n_remaining)
            #breaks[K] = last_break_table[K-1, last_break_table[K-1, breaks[K + 1] - 1-1]]
            n_remaining = last_break_table[K-1, last_break_table[K-1, breaks[K+1]-1]-1]
            breaks[K] = last_break_table[K-1, n_remaining-1]
        print(breaks)"""
        return(breaks)

    if use_iterative:
        #loss_table, last_break_table = solve_iterative(returns, num_prods=k,weights=weights)
        #optimal_breaks=np.array(reconstruct_solution_it(k,n))
        regret,opt_prods = solve_iterative(returns,num_prods=k, weights =weights)
        #return loss_table[k-1,n-1], optimal_breaks
        return(regret,opt_prods)
    else:
    # use the recursive helper function to solve all subproblems
        solve_recursively(n, k)
     #   print(last_break_table)
    # reconstruct the optimal break points and convert to a numpy array
    optimal_breaks = np.array(reconstruct_solution(n, k+1))
    #print(optimal_breaks)
    return loss_table[n, k], optimal_breaks


def regret_minimizing_products(returns, k, use_avg_regret=True):
    """
    Finds a set of k products that minimizes regret when each consumer is
    assigned to the highest return product that does not exceed their risk
    threshold. Takes as input a sorted numpy array of returns of length n
    containing the return for the bespoke portfolio for each consumer and the
    number k of products to choose. Returns indices of the k consumers whose
    risk limits define the regret-minimizing products, as well as the total
    regret.
    """
    check_returns_argument(returns)
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
    if use_avg_regret:
        regret /= len(returns)

    # The indices of the optimal products are just the first k elements of the
    # breakpoints array returned by interval_partition
    products = breaks[0:k]

    return regret, products


def weighted_regret_minimizing_products(returns, weights, k,use_iterative=False):
    """
    Same as regret_minimizing_products except each consumer has a real-valued
    and the goal is to minimize the total weighted regret.
    """
    # As in regret_minimizing_products, it is equivalent and slightly simpler
    # to find the set of products that maximize weighted return.

    # The following loss function computes the negative total weighted return
    # for consumers {i, ..., j-1} when assigned to a product defined by
    # consumer i. Uses a pre-computed cumulative sum of the weights so that it
    # costs O(1) time per evaluation.
    check_returns_argument(returns)
    weights_csum = np.cumsum(weights)
    def loss_fn(i, j):
        if i == 0:
            return -returns[0] * weights_csum[j-1]
        else:
            return -returns[i] * (weights_csum[j-1] - weights_csum[i-1])

    # Call interval_partition to find the products that maximize revenue.

    loss, breaks = interval_partition(len(returns), k, loss_fn,use_iterative=use_iterative,returns = returns, weights = weights)

    # Calculate the regret and extract the set of products.
    regret = np.dot(weights, returns) + loss
    products = breaks[0:k]
    #print(products)
    return regret, products
