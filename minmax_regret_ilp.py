import cvxpy as cp
import numpy as np

def minmax_regret_ilp(users, num_prods):

    # this is preliminary

    # for convenience here, i've assumed `users` is a list of pairs 
    # of the form (return_threshold, group_membership)

    # extract some useful values
    n = len(users)
    groups = set([u[1] for u in users])
    group_regrets = {g : 0 for g in groups}


    # B is the regret of the most regretful group -- the thing we want to minimize
    B = cp.Variable()


    # X is the assingment variable, X[i,j] == 1 iff user i is assigned to the product defined by j
    X = cp.Variable((n,n), boolean = True)

    # y is the product variable. y[k] == 1 iff k defines a product
    y = cp.Variable(n, boolean = True)

    # we want to minimize the max regret B
    objective = cp.Minimize(B)

    constraints = []

    # constraint B to be positive (probably unnecessary)
    constraints.append((B>=0))

    # each user is assigned to exactly one product -- rows of X sum to 1
    constraints.append((cp.sum(X,axis=1) == 1))

    # we have to pick a certain number of products
    constraints.append( cp.sum(y) == num_prods   )


    for i in range(n):
        for j in range(n):

            # we cant assign user i to product j if product j isnt in our set of products
            constraints.append((X[i,j] <= y[j])) 

            # if j's desired return is larger than i's
            # we cant give j's product to i
            if users[j][0] > users[i][0]:
                constraints.append( ( X[i,j] == 0 )  )

            # the regret of user i is the difference between his desired return
            # and the return he gets from his assigned product
            g = users[i][1]
            group_regrets[g] += (users[i][0]-users[j][0]) * X[i,j]


    # the regret of each group must be less than B
    for r in group_regrets.values():
        constraints.append(( r<=B ))


    prob = cp.Problem(objective,constraints)

    prob.solve()

    return B.value, np.around(X.value), np.around(y.value), [v.value for v in group_regrets.values()]