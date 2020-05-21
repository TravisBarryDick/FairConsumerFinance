# subroutine for computing products via greedy algo, ignoring groups

import numpy as np
import math
import utils

def sparsify(returns,products,epsilon):

  sparse = products
  rets = returns[sparse]
  delta = np.diff(rets)
  done = (min(delta) > epsilon)
  while (done == 0):
    idx = np.where(delta <= epsilon) 
    for i in idx:
      sparse[i+1] = 0 
    sparse = np.unique(sparse)
    rets = returns[sparse]
    delta = np.diff(rets)
    done = (min(delta) > epsilon)
  return sparse 

  # for i in np.arange(0,len(sparse)-1,1):
    # if (returns[sparse[i+1]]-returns[sparse[i]] <= epsilon):
      # sparse[i+1] = 0
  # sparse = np.unique(sparse)
  # return sparse 
