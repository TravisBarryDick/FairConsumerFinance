def minmax_regret_ilp(users, num_prods):

    # this is preliminary, we're seeing if the solver can handle travis' ilp formulation

    # if this works, it should be cleaned up

    # for convenience here, i've assumed `users` is a list of pairs 
    # of the form (return_threshold, group_membership)

    
    n = len(users)
    B = cp.Variable()

    groups = set([u[1] for u in users])
    group_regrets = {g:0 for g in groups}
    # X is the assingment variable. each person gets 
    # one product (sum of each row is 1)
    X = cp.Variable((n,n), boolean = True)
    y = cp.Variable(n, boolean = True)


    objective = cp.Minimize(B)

    constraints = []
    constraints.append((B>=0))
    constraints.append((cp.sum(X,axis=1) == 1))

    constraints.append( cp.sum(y) == num_prods   )


    for i in range(n):
        for j in range(n):


            constraints.append((X[i,j] <= y[j])) 

    for i in range(n):
        for j in range(n):
            # if j's desired return is larger than i's
            # we cant give j's product to i
            if users[j][0] > users[i][0]:
                constraints.append( ( X[i,j] == 0 )  )





    for i in range(n):
        for j in range(n):
            # the regret of user i is the difference between his desired return
            # and the return he gets from his assigned product
            g = users[i][1]
            group_regrets[g] += (users[i][0]-users[j][0]) * X[i,j]


    for r in group_regrets.values():
        constraints.append(( r<=B ))

    prob = cp.Problem(objective,constraints)

    return X,y,B,group_regrets,prob


u = sorted([[random.random()*5,random.choice(range(2))] for i in range(5) ])

mri = minmax_regret_ilp