import cvxpy as cp


def max_return_constrain_variance(vals, covars, risk_limit, budget, long_only=False):

    n_assets = vals.shape[0]

    # maybe sanity-check the shapes of the inputs?

    weights = cp.Variable(n_assets)

    obj = cp.Maximize(weights.T @ vals)
    constrs = [(cp.sum(weights) <= budget), (cp.quad_form(weights, covars) <= risk_limit)]
    if long_only:
        constrs.append(weights >= 0)

    prob = cp.Problem(obj, constrs)
    prob.solve()
    return weights.value, prob.value


def min_risk_constrain_returns(vals, covars, return_threshold, budget, long_only=False):

    n_assets = vals.shape[0]

    # maybe sanity-check the shapes of the inputs?

    weights = cp.Variable(n_assets)

    obj = cp.Minimize(cp.quad_form(weights, covars))
    constrs = [(cp.sum(weights) <= budget), (weights.T @ vals >= return_threshold)]
    if long_only:
        constrs.append(weights >= 0)

    prob = cp.Problem(obj, constrs)
    prob.solve()

    return weights.value, prob.value
