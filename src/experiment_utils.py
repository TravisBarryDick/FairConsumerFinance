import numpy as np
import pandas as pd
import sys
import os
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import bespoke_solvers
import minmax_regret_game
import minmax_regret_ilp
import greedy
import regret_minimization_dp
from utils import get_group_regrets
from utils import check_group_arguments
from utils import check_products_argument





def bespoke_return(risk):
    w, r = bespoke_solvers.max_return_constrain_variance(
        asset_returns, asset_covars, risk, 1.0, long_only=True)
    return r


def calc_theoretical_max_regret(returns,groups,ngroups):
    return(np.max(np.array([np.sum(returns[groups == i]) for i in range(ngroups)])))

def calc_theroetical_stepsize(max_group_regret, ngroups, T):
    stepsize = np.sqrt(8 * np.log(ngroups) / T) / max_group_regret
    return(stepsize)

def calc_stepsize(stepsize_mult,returns,groups,ngroups,T):
    max_group_regret = calc_theoretical_max_regret(returns,groups,ngroups)
    theory_ss = calc_theroetical_stepsize(max_group_regret,ngroups,T)
    return(stepsize_mult*theory_ss)

def getregret(groupsizes,groupregrets):
  Z = np.sum(groupsizes)
  regret = np.sum(np.multiply(np.divide(groupsizes,Z),groupregrets))
  return regret

def get_grr_vary_mean(n_consumers,mean_low,mean_dif, std_low, std_dif,  p_high=0.5 ):
    means = [mean_low, mean_low+mean_dif]
    stds = [std_low, std_dif]
    consumer_groups = np.random.binomial(1,p_high,n_consumers)
    consumer_risks = np.array([np.random.normal(loc=means[consumer_groups[i]], scale=stds[consumer_groups[i]]) for i in range(n_consumers)])
    consumer_risks[consumer_risks<0]=0
    consumer_risks = [round(risk,6) for risk in consumer_risks]
    consumer_returns = np.array([bespoke_return(r) for r in consumer_risks])
    return(consumer_groups,consumer_risks, consumer_returns)

def get_grr_vary_mean_multigroup(n_consumers,mean_low,mean_dif, std_low, std_dif,  ps,ngroups ):
    means = [mean_low + i * mean_dif for i in range(ngroups)]
    stds = [std_low + i * std_dif for i in range(ngroups)]
    consumer_groups = random.choices(range(ngroups), weights=ps, k=n_consumers)
    consumer_risks = np.array([np.random.normal(loc=means[consumer_groups[i]], scale=stds[consumer_groups[i]]) for i in range(n_consumers)])
    consumer_risks[consumer_risks<0]=0
    consumer_risks = [round(risk,6) for risk in consumer_risks]
    consumer_returns = np.array([bespoke_return(r) for r in consumer_risks])
    return(consumer_groups,consumer_risks, consumer_returns)

def get_grr_vary_mean_rare(n_consumers, mean_low, mean_dif, std_low, std_dif, probs, n_normal_groups, prob_exp, lambda_exp):
    means = [mean_low + i * mean_dif for i in range(n_normal_groups)]
    stds = [std_low + i * std_dif for i in range(n_normal_groups)]
    consumer_groups = random.choices(range(n_normal_groups+1), weights=probs+[prob_exp], k=n_consumers)
    consumer_risks = []
    for consumer in enumerate(consumer_groups):
        if consumer==n_normal_groups:
            consumer_risks.append(np.random.normal(loc=means[consumer_groups[consumer]],scale=stds[consumer_groups[consumer]]))
        else:
            consumer_risks.append(np.random.exponential(lambda_exp)*means[0])
    consumer_risks = np.array(consumer_risks)
    consumer_risks[consumer_risks < 0] = 0
    consumer_risks = [round(risk,6) for risk in consumer_risks]
    consumer_returns = np.array([bespoke_return(r) for r in consumer_risks])
    return (consumer_groups, consumer_risks, consumer_returns)

def get_asset_ret_cov(asset_scale=252):
    asset_covars = np.genfromtxt("data/cov_nolabels.csv", delimiter=",")
    asset_returns = np.genfromtxt("data/returns_nolabels.csv", delimiter=",")
    asset_covars = asset_covars*(asset_scale)
    asset_returns = asset_returns*asset_scale
    return(asset_covars,asset_returns)


def sort_data(groups,risks,returns):
    df = pd.DataFrame({'groups':groups, 'risks':risks, 'returns':returns})
    df = df.sort_values('returns')
    return(np.array(df.groups),np.array(df.risks),np.array(df.returns))


def do_game(returns,groups,ngroups,groupsizes, n_prod,stepsize,T):
    # do game

    all_products_mm, group_weights, all_group_regrets_mm = minmax_regret_game.minmax_regret_game(
        returns=returns,
        groups=groups,
        num_groups=ngroups,
        num_prods=n_prod,
        T=T,
        use_avg_regret=True,
        step_size=stepsize,
        use_it=True)
    products_mm = np.unique(all_products_mm)
    group_regrets_mm = get_group_regrets(returns, groups, ngroups, products_mm)
    regret_mm_max = np.max(group_regrets_mm)
    if len(groupsizes)>len(group_regrets_mm):
        groupsizes = [size for size in groupsizes if size>0]
    group_regrets_mm = np.array(group_regrets_mm)
    groupsizes = np.array(groupsizes)

    regret_mm_pop = getregret(groupsizes, group_regrets_mm)
    regret_mm_dif = np.max(group_regrets_mm) - np.min(group_regrets_mm)
    results = {'R_max':regret_mm_max,'R_pop':regret_mm_pop,'R_dif':regret_mm_dif}
    return (results, products_mm)
def do_greedy(returns,groups,n_groups,groupsizes, n_prod):
    regret_greedy, products_greedy = greedy.greedy_products(returns, n_prod)
    groupsizes = [size for size in groupsizes if size > 0]
    group_regrets_greedy = get_group_regrets(returns, groups, n_groups, products_greedy, True)
    regret_greedy_pop = getregret(groupsizes, group_regrets_greedy)
    regret_greedy_max = max(group_regrets_greedy)
    regret_greedy_dif = max(group_regrets_greedy) - min(group_regrets_greedy)
    results = {'R_max':regret_greedy_max,'R_pop':regret_greedy_pop,'R_dif':regret_greedy_dif}
    return(results, products_greedy)
def do_greedy_mm(returns, groups,n_groups,groupsizes, n_prod):

    group_regrets_greedy_mm, products_greedy_mm = greedy.greedyminmax(returns, groups, n_groups, n_prod)
    groupsizes = [size for size in groupsizes if size>0]
    regret_greedy_mm_pop = getregret(groupsizes, group_regrets_greedy_mm)
    regret_greedy_mm_max = max(group_regrets_greedy_mm)
    regret_greedy_mm_dif = max(group_regrets_greedy_mm) - min(group_regrets_greedy_mm)
    results = {'R_max':regret_greedy_mm_max,'R_pop':regret_greedy_mm_pop,'R_dif':regret_greedy_mm_dif}
    return(results, products_greedy_mm)
def do_sparsify(returns,groups, n_groups, groupsizes, n_prod, slack, products_mm):
    sparse_products = sparsify_2_wrapper(returns, products_mm, n_prod + slack)
    groupsizes = [size for size in groupsizes if size>0]
    sparse_group_regret = get_group_regrets(returns, groups, n_groups, sparse_products)
    sparse_regret_max = max(sparse_group_regret)
    sparse_regret_pop = getregret(groupsizes, sparse_group_regret)
    sparse_regret_dif = max(sparse_group_regret) - min(sparse_group_regret)
    results = {'R_max':sparse_regret_max,'R_pop':sparse_regret_pop,'R_dif':sparse_regret_dif}
    return(results,sparse_products)

def calc_kset_regret(returns,groups,numgroups,kset):
    kset = np.array(kset)
    max_regret = np.max(get_group_regrets(returns,groups,numgroups,kset))
    return(max_regret)
def calc_exp_regret(kset_series, kset_regrets):
    return( 1/len(kset_series)*np.sum([kset_regrets[tuple(kset)] for kset in kset_series]))

def calc_max_group_E_regret(groups,numgroups, kset_series, kset_regrets, returns):
    all_regrets = [get_group_regrets(returns,groups,numgroups,kset) for kset in kset_series]
    T = len(kset_series)
    group_avg_regrets = 1/T* np.max( [np.sum([all_regrets[t][j]for t in range(T)]) for j in range(numgroups)] )
    return( np.max(group_avg_regrets))
def sparsify_3(returns,products,k):

  sparse = np.unique(np.sort(products))
  rets = returns[sparse]
  delta = np.diff(rets)
  done = (sparse.size <= k)
  while (done == 0):
    idx = np.where(delta > 0)
    mingap = 10.0
    for i in np.nditer(idx):
      if (delta[i] <= mingap):
        mingap = delta[i]
        minidx = i
    sparse[minidx+1] = 0
    sparse = np.unique(sparse)
    rets = returns[sparse]
    delta = np.diff(rets)
    done = (sparse.size <= k)
  return sparse

def sparsify_2_wrapper(returns, products, k):
    return (sparsify_3(returns, products, k))


def get_group_regrets_nonconsumer_prods(returns, groups, num_groups, actual_products, use_avg_regret=True):
    """
    We update the original method to allow for products outside of consumer returns.
    """
    check_group_arguments(groups, num_groups)
    assert type(actual_products) is np.ndarray, \
        "products must be a numpy array "
    assert all(np.diff(actual_products) >= 0), \
        "product indices must be sorted in non-decreasing order"

    group_regrets = np.zeros(num_groups)
    group_sizes = np.zeros(num_groups)
    p_ix = len(actual_products) - 1
    for c_ix in range(len(returns)-1, -1, -1):
        g_ix = groups[c_ix]
        group_sizes[g_ix] += 1
        while actual_products[p_ix] > returns[c_ix]:
            p_ix -= 1
        group_regrets[g_ix] += returns[c_ix] - actual_products[p_ix]
    if use_avg_regret:
        for g in range(num_groups):
            if group_sizes[g] > 0:
                group_regrets[g] /= group_sizes[g]
    return group_regrets

def test_regret_Functions():
    returns = np.array([0.1,0.3,0.4,0.45,0.5])
    groups = np.array([0,1,0,1,2])
    actual_products = np.array([0,0.2,0.4])
    num_groups=3
    out = get_group_regrets_nonconsumer_prods(returns, groups, num_groups, actual_products,
                                        use_avg_regret=True)
    #products = [0,0.2,.0.4]
    expected = np.array([0.05,0.075, 0.1])
    assert(min(np.isclose(expected,out)))
    """new regret function not same as expected"""

    out_old = get_group_regrets(returns,groups,num_groups,np.array([0,2,4,]))
    expected_old = np.array([0,0.125,0])
    assert(min(np.isclose(expected_old,out_old)))
    """old regret function not same as expected"""
def do_DP(returns,groups,groupsizes,n_groups,n_prod, consumer_weights_dp):
    reg, opt_prods = regret_minimization_dp.regret_minimizing_products_jit(returns,n_prod,consumer_weights_dp)
    group_regrets = get_group_regrets(returns,groups,n_groups,opt_prods)
    max_group_regret = np.max(group_regrets)
    pop_regret = getregret(groupsizes, group_regrets)
    regret_dif = np.max(group_regrets)-np.min(group_regrets)
    results = {'R_max': max_group_regret, 'R_pop': pop_regret, 'R_dif': regret_dif}
    return(results,opt_prods)

def do_ILP(returns, groups, ngroups, n_prods, groupsizes, use_avg_regret=True):
    max_regret_ilp, group_regrets_ilp, products_ilp = minmax_regret_ilp.minmax_regret_ilp_wrapper(returns, groups, ngroups, n_prods, use_avg_regret = use_avg_regret, solver_str = solver)
    pop_regret = getregret(groupsizes, group_regrets_ilp)
    regret_dif = np.max(group_regrets_ilp) - np.min(group_regrets_ilp)
    results = {'R_max': max_regret_ilp, 'R_pop': pop_regret, 'R_dif': regret_dif}
    return (results, products_ilp)


from configs import *
random.seed(seed)
asset_covars, asset_returns = get_asset_ret_cov(asset_scale)
