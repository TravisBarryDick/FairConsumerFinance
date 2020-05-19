# subroutine for computing products via greedy algo, ignoring groups

import numpy as np
import math
import utils

def greedy_products(returns,k):

  n_consumers = len(returns)
  groups = np.zeros(n_consumers,dtype=int)	# hack to use group regret subroutine, everyone in one group
  products = np.zeros(k,dtype=int)
  products[0] = 0				# first product is lowest risk/ret consumer

  for j in np.arange(1,k,1):	# fill in remaining products
    reglist = np.zeros(n_consumers)		# to keep track of regret of adding each consumer as next prod
    for i in np.arange(0,n_consumers,1):
      testprods = products
      testprods[j] = i				# try adding each consumer as next product
      testprods = np.sort(testprods)
      group_regrets = utils.get_group_regrets(returns,groups,1,testprods,True)	# see how good this product is
      thisreg = group_regrets[0]
      reglist[i] = thisreg
    idx = np.argmin(reglist)			# take best next product
    products[j] = idx
  products = np.sort(products)
  group_regrets = utils.get_group_regrets(returns,groups,1,products,True)
  regret = group_regrets[0]
  return regret, products
