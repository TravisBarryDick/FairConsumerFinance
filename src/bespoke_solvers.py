import cvxpy as cp

def max_return_constrain_variance(vals, covars, risk_limit, budget, long_only=False):
    """Takes in the values and covariances of a set of assets as well as a
       user's risk limit and budget and returns the optimal mixture of assets.

       Optionally only solves where non-negative quantites of each asset are
       included.

    Parameters
    ----------
    vals : list or list-like
        A sequence of the expected returns of each asset.
    covars : 2d array
        The covariance matrix of the asset returns.
    risk_limit : float
        The maximum tolerated variance for the optimal portfolio.
    budget : float
        The maximum total quantity of assets to include.
    long_only : bool, optional
        If `True`, only purchases non-negative quantities of all assets, 
        otherwise, short-sales (i.e. negative quantities) are permitted.

    Returns
    -------
    1d array
        The amount of each asset purchased.
    float
        The expected return of the optimal portfolio.

    """

    n_assets = vals.shape[0]

    # maybe sanity-check the shapes of the inputs?

    weights = cp.Variable(n_assets)

    obj = cp.Maximize(weights.T @ vals)
    constrs = [(cp.sum(weights) <= budget),
               (cp.quad_form(weights, covars) <= risk_limit)]
    if long_only:
        constrs.append(weights >= 0)

    prob = cp.Problem(obj, constrs)
    prob.solve()
    return weights.value, prob.value


def min_risk_constrain_returns(vals, covars, return_threshold, budget,
                               long_only=False):

    """Takes in the values and covariances of a set of assets as well as a
       user's minimum demanded return and budget and returns the optimal
       mixture of assets.

       Optionally only solves where non-negative quantites of each asset are
       included.

    Parameters
    ----------
    vals : list or list-like
        A sequence of the expected returns of each asset.
    covars : 2d array
        The covariance matrix of the asset returns.
    risk_limit : float
        The maximum tolerated variance for the optimal portfolio.
    budget : float
        The maximum total quantity of assets to include.
    long_only : bool, optional
        If `True`, only purchases non-negative quantities of all assets,
        otherwise, short-sales (i.e. negative quantities) are permitted.

    Returns
    -------
    1d array
        The amount of each asset purchased.
    float
        The expected return of the optimal portfolio.

    """


    n_assets = vals.shape[0]

    # maybe sanity-check the shapes of the inputs?

    weights = cp.Variable(n_assets)

    obj = cp.Minimize(cp.quad_form(weights, covars))
    constrs = [(cp.sum(weights) <= budget),
               (weights.T @ vals >= return_threshold)]
    if long_only:
        constrs.append(weights >= 0)

    prob = cp.Problem(obj, constrs)
    prob.solve()

    return weights.value, prob.value
