from bespoke_solvers import max_return_constrain_variance
import numpy as np
import sys

covars = np.genfromtxt(sys.argv[1], delimiter=",")
vals = np.genfromtxt(sys.argv[2])
risk = float(sys.argv[3])

weights, obj = max_return_constrain_variance(vals, covars, risk, 1, True)

print("Objective value: ", obj)
print("Total weight: ", np.sum(weights))
print("Weights", weights)
