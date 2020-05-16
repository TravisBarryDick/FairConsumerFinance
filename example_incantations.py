import numpy as np

import bespoke_solvers
import minmax_regret_game
import minmax_regret_ilp
import regret_minimization_dp

################
# Loading Data #
################

# Note: I removed the column and row labels from the data Michael provided
#       because I couldn't figure out how to get numpy to ignore 1 row and 1
#       column. We should probably use pandas dataframes or something like that.

asset_covars = np.genfromtxt("data/cov_nolabels.csv", delimiter=",")
asset_returns = np.genfromtxt("data/returns_nolabels.csv", delimiter=",")

###############################
# Computing Bespoke Solutions #
###############################

asset_weights, portfolio_return = bespoke_solvers.max_return_constrain_variance(
    asset_returns, asset_covars, 0.0001, 1, long_only=True)

# portfolio_return = 0.0008552207953564784
# asset_weights = (rounded to 3 decimal points)
# array([0.168, 0.   , 0.   , 0.   , 0.068, 0.   , 0.   , 0.   , 0.   ,
#        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
#        0.   , 0.037, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.132,
#        0.   , 0.   , 0.   , 0.   , 0.138, 0.   , 0.   , 0.005, 0.   ,
#        0.   , 0.   , 0.   , 0.   , 0.   , 0.096, 0.   , 0.   , 0.   ,
#        0.   , 0.   , 0.   , 0.   , 0.   ])

asset_weights, portfolio_risk = bespoke_solvers.min_risk_constrain_returns(
    asset_returns, asset_covars, 0.00085522, 1, long_only=True)

# portfiolio_risk = 9.999981370091702e-05 (pretty much 0.0001)
# asset_weights = (rounded to 3 decimal points)
# array([0.168, 0.   , 0.   , 0.   , 0.068, 0.   , 0.   , 0.   , 0.   ,
#        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
#        0.   , 0.037, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.132,
#        0.   , 0.   , 0.   , 0.   , 0.138, 0.   , 0.   , 0.005, 0.   ,
#        0.   , 0.   , 0.   , 0.   , 0.   , 0.096, 0.   , 0.   , 0.   ,
#        0.   , 0.   , 0.   , 0.   , 0.   ])

#############################################
# Computing Returns for Synthetic Consumers #
#############################################

# Some synthetic consumer risk limits
consumer_risks = np.array([1e-4, 2e-4, 3e-4, 4e-4, 1])


# Simple function that computes the return for a given risk threshold (using
# the global asset_returns and asset_covars data). Only defined this to make
# the next lines a little less daunting :)
def bespoke_return(risk):
    w, r = bespoke_solvers.max_return_constrain_variance(
        asset_returns, asset_covars, risk, 1.0, long_only=True)
    return r


consumer_returns = np.array([bespoke_return(r) for r in consumer_risks])
# consumer_returns =
# array([0.00085522, 0.00120946, 0.00146318, 0.00160231, 0.0019195 ])

# Note: we could have typed these returns in by hand. I just wanted to use the
#       solvers

######################################
# Regret Minimizing Dynamic Programs #
######################################

regret, products = regret_minimization_dp.regret_minimizing_products(
    consumer_returns, 2)

# regret = 0.0001899403327051819
# products = array([0, 2])

# Note: the products array gives the indices of the consumers that define the
#       products. We are forced to have a product at index 0 so that the lowest
#       risk consumer has a suitable product. This solution puts the movable
#       product at consumer 2, which serves consumers 2, 3, and 4.

regret, products = regret_minimization_dp.weighted_regret_minimizing_products(
    consumer_returns, np.array([0, 0, 0, 0, 2]), 2)
# regret = 0.0
# products = array([0, 4])

# Note: this is the same as the above problem, except we put 0 weight on all
#       consumers except for a weight of 2 on the riskiest consumer. The
#       solution uses its one movable product to serve that consumer.

# Note: we really only need the weighted version of the DP for the two-player
#       game

#########################
# ILP For MinMax Regret #
#########################

# We use risk-based groups with the 3 lowest risk consumers in group 0 and the
# 2 highest risk consumers in group 1.
consumer_groups = np.array([0, 0, 0, 1, 1])
num_groups = 2

regret, products = minmax_regret_ilp.minmax_regret_ilp_wrapper(
    consumer_returns, consumer_groups, num_groups, 3)
# regret = 0.00031718630921291033
# products = array([0, 1, 3])

# Note: The 3 producst that minimize overall regret are array([0, 2, 4]), so
#       this is an instance where the minmax products are not the same as the
#       overall regret minimizing products


#####################################
# Two Player Game for MinMax Regret #
#####################################

# play the two-player game for 5 rounds and get the products and weights chosen
# on each round, together with the total regret of each group. 5 rounds is a
# bit too short to see much interesting stuff, but I wanted to keep the
# printouts short.
products, group_weights, group_regrets = minmax_regret_game.minmax_regret_game(
    consumer_returns, consumer_groups, num_groups, 3, 5)

# products =
# array([[0, 2, 4],
#        [0, 2, 4],
#        [0, 2, 4],
#        [0, 2, 4],
#        [0, 1, 2]])

# group_weights =
# array([[0.5       , 0.5       ],
#        [0.51604751, 0.48395249],
#        [0.532062  , 0.467938  ],
#        [0.54801071, 0.45198929],
#        [0.56386141, 0.43613859]])

# group_regrets =
# array([[0.00035424, 0.00013914],
#        [0.00035424, 0.00013914],
#        [0.00035424, 0.00013914],
#        [0.00035424, 0.00013914],
#        [0.        , 0.00059546]])
