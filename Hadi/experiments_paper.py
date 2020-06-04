"""Experiments for Paper.

# Experiment 1: No regret Convergence time vs Stepsize. (Figure in long version)
# Experiment 2: Performance by ILP, sparsify, greedy, greedy MM (old figure)
# Experiment 3: varying mean difference of normals
# Experiment 4: varying std difference of normals
# Experiment 5: varying mean and std of normal via multiplier
# Experiment 6: Rare group
# Experiment 7: Generalization
# Experiment 8: performance of sparsify (Figure 1)
# Experiment 9: Performance of Algos (Figure 2,3)
# Experiment 10: Product Dependence (+timing) (Figure 4, Figure 5)
# Experiment 11: Consumer Dependence (+timing) (Figure 5)
# Experiment 12: Generalization Final Version (Figure 6)
"""

import numpy as np
import pandas as pd
import sys
import os
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir('/Users/Hadi/Dropbox/Research/Fair Consumer Finance/python_code/Hadi')
sys.path.append(os.path.abspath('../'))
import bespoke_solvers
import minmax_regret_game
import minmax_regret_ilp
import greedy
import greedyminmax
import regret_minimization_dp
from utils import get_group_regrets
from utils import check_group_arguments
from utils import check_products_argument
sys.path.append(os.path.abspath('../mkearns'))
import sparsify as sp
import getregret
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',500)

#from compare_game_ilp import bespoke_return,calc_theoretical_max_regret,calc_theroetical_stepsize,\
   # calc_stepsize,sparsify_3,sparsify_2_wrapper ,getregret, get_grr_vary_mean, get_grr_vary_mean_rare, \
    #get_grr_vary_mean_multigroup,sort_data, do_game,do_greedy,do_greedy_mm,do_sparsify
#from compare_game_ilp import do_ilp
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
    asset_covars = np.genfromtxt("../data/cov_nolabels.csv", delimiter=",")
    asset_returns = np.genfromtxt("../data/returns_nolabels.csv", delimiter=",")
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

    group_regrets_greedy_mm, products_greedy_mm = greedyminmax.greedyminmax(returns, groups, n_groups, n_prod)
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

def calc_max_group_E_regret(groups,numgroups, kset_series, kset_regrets):
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


def get_group_regrets_nonconsumer_prods(returns, groups, num_groups, actual_products,
                      use_avg_regret=True):
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

def do_ILP(returns, groups, ngroups, n_prods, use_avg_regret=True):
    max_regret_ilp, group_regrets_ilp, products_ilp = minmax_regret_ilp.minmax_regret_ilp_wrapper(returns, groups, ngroups, n_prods, use_avg_regret = use_avg_regret, solver_str = 'GUROBI')
    pop_regret = getregret(groupsizes, group_regrets_ilp)
    regret_dif = np.max(group_regrets_ilp) - np.min(group_regrets_ilp)
    results = {'R_max': max_regret_ilp, 'R_pop': pop_regret, 'R_dif': regret_dif}
    return (results, products_ilp)

if __name__=='__main__':

    run_exp_1 = False
    run_exp_2 = False
    run_exp_3 = False
    run_exp_4 = False
    run_exp_5 = False
    run_exp_6 = False
    run_exp_7 = False
    run_exp_8 = False
    run_exp_9 = True
    run_exp_10 = False
    run_exp_11 = False
    run_exp_12 = False

    test_regret_Functions()
    seed = 10
    random.seed(seed)
    asset_scale = 252
    asset_covars,asset_returns = get_asset_ret_cov(asset_scale)

    try:
        if run_exp_1:
            print('Running Experiment 1')

            ###### first experiment: performance. Convergence time and stepsize dependendence.
            #groups,risks,returns = get_grr_vary_mean_multigroup(50,0.02, 0.01,0.002,0.001,1/3*np.array([1,1,1]),3)
            groups,risks,returns = get_grr_vary_mean_multigroup(50,0.001, 0.0001,0.005,0.0005,1/3*np.array([1,1,1]),3)
            groups,risks,returns = sort_data(groups,risks,returns)
            np.unique(groups,return_counts=True)
            ngroups=len(np.unique(groups))
            val_counts = dict(zip(np.unique(groups, return_counts=True)[0], np.unique(groups, return_counts=True)[1]))
            groupsizes = [val_counts[g] if g in val_counts.keys() else 0 for g in range(ngroups)]
            stepsize_mults = [1, 10,100,1000,10000]
            T = 1000
            n_prod = 5
            results = {'t':[], 'Exp_R':[], 'Stepsize Multiplier':[]}
            for step_mult  in stepsize_mults:
                stepsize = calc_stepsize(step_mult, returns, groups, ngroups, T)
                all_products_mm, group_weights, all_group_regrets_mm = minmax_regret_game.minmax_regret_game(
                    returns=returns,
                    groups=groups,
                    num_groups=ngroups,
                    num_prods=n_prod,
                    T=T,
                    use_avg_regret=True,
                    step_size=stepsize,
                    use_it=True)
                ksets = list(set([tuple(kset) for kset in all_products_mm]))
                kset_regret = {kset: calc_kset_regret(returns, groups, ngroups, kset) for kset in ksets}
                #exp_regrets = [calc_exp_regret(all_products_mm[0:t+1], kset_regret) for t in range(T)]
                exp_regrets = [calc_max_group_E_regret(groups,ngroups, all_products_mm[0:t+1],kset_regret)for t in range(T)]
                for t in range(T):
                    results['t'].append(t+1)
                    results['Exp_R'].append(exp_regrets[t])
                    results['Stepsize Multiplier'].append('M'+str(step_mult))

            results_df = pd.DataFrame(results)

            fig = plt.figure(figsize=(12,9))
            sns.lineplot('t', 'Exp_R', hue='Stepsize Multiplier', data=results_df)
            plt.ylabel('Max Expected Group Regret')
            plt.title('No Regret Game Max Expected Group Regret')
            plt.savefig('Convergence_stepsize_2.png')
            plt.close(fig)
    except:
        print('Exp 2 failed')

    try:
        if run_exp_2:
            print('Running Experiment 2')

            ### Experiment 2 - Performance by ILP, sparsify, greedy, greedy MM

            seed = 10
            random.seed(seed)
            n_instances=5
            results = {'Prods':[],'Prod_des':[], 'R_group':[],'R_pop':[], 'Algo':[],'Instance':[]}
            for j in range(n_instances):
                groups,risks,returns = get_grr_vary_mean_multigroup(50,0.02, 0.01,0.002,0.001,1/3*np.array([1,1,1]),3)
                groups,risks,returns = sort_data(groups,risks,returns)
                np.unique(groups,return_counts=True)
                ngroups=len(np.unique(groups))
                val_counts = dict(zip(np.unique(groups, return_counts=True)[0], np.unique(groups, return_counts=True)[1]))
                groupsizes = [val_counts[g] if g in val_counts.keys() else 0 for g in range(ngroups)]
                stepsize_mult = 1000
                T = 2000
                n_prods = range(1,11)

                for p in n_prods:
                    max_regret_ilp, group_regrets_ilp, products_ilp = minmax_regret_ilp.minmax_regret_ilp_wrapper(
                        returns, groups, ngroups, p, use_avg_regret=True, solver_str='GUROBI')
                    results['Prods'].append(p)
                    results['Prod_des'].append(p)
                    results['R_group'].append(np.max(group_regrets_ilp))
                    results['R_pop'].append(getregret(groupsizes,group_regrets_ilp))
                    results['Algo'].append('ILP')
                    results['Instance'].append(j)

                for p in n_prods:
                    results_greedy,prods_greedy = do_greedy(returns,groups,ngroups,groupsizes,p)
                    results['Prods'].append(p)
                    results['Prod_des'].append(p)
                    results['R_group'].append(results_greedy['R_max'])
                    results['R_pop'].append(results_greedy['R_pop'])
                    results['Algo'].append('Greedy')
                    results['Instance'].append(j)

                for p in n_prods:
                    results_greedy_mm,prods_greedy_mm = do_greedy_mm(returns,groups,ngroups,groupsizes,p)
                    results['Prods'].append(p)
                    results['Prod_des'].append(p)
                    results['R_group'].append(results_greedy_mm['R_max'])
                    results['R_pop'].append(results_greedy_mm['R_pop'])
                    results['Algo'].append('GreedyMM')
                    results['Instance'].append(j)

                for p in n_prods:
                    stepsize = calc_stepsize(step_mult, returns, groups, ngroups, T)
                    results_game, prods_game = do_game(returns,groups,ngroups,groupsizes,p,stepsize,T)
                    results['Prods'].append(len(prods_game))
                    results['Prod_des'].append(p)
                    results['R_group'].append(results_game['R_max'])
                    results['R_pop'].append(results_game['R_pop'])
                    results['Algo'].append('NoRegret')
                    results['Instance'].append(j)

                    results_slack_1,slack_1_prod = do_sparsify(returns, groups, ngroups, groupsizes, p, 1, prods_game)
                    results['Prods'].append(p+1)
                    results['Prod_des'].append(p)
                    results['R_group'].append(results_slack_1['R_max'])
                    results['R_pop'].append(results_slack_1['R_pop'])
                    results['Algo'].append('SparseP+1')
                    results['Instance'].append(j)

                    results_slack_2 ,slack_2_prod = do_sparsify(returns, groups, ngroups, groupsizes, p, 2, prods_game)
                    results['Prods'].append(p+2)
                    results['Prod_des'].append(p)
                    results['R_group'].append(results_slack_2['R_max'])
                    results['R_pop'].append(results_slack_2['R_pop'])
                    results['Algo'].append('SparseP+2')
                    results['Instance'].append(j)

            results_df = pd.DataFrame(results)

            fig = plt.figure(figsize=(20,10))
            sns.scatterplot('Prods','R_group',hue='Algo',data=results_df[results_df.Algo != 'NoRegret'])
            sns.scatterplot('Prods','R_group',data=results_df[results_df.Algo=='NoRegret'],color='black',marker='P',s=100,label='NoRegret')
            plt.xlabel('Products Used')
            plt.ylabel('Max Group Regret Ex-Post')
            plt.title('No Regret Game Max Expected Group Regret')
            plt.savefig('Ex Post Regret by Products scatter.png')
            plt.close(fig)

            fig = plt.figure(figsize=(20,10))
            sns.lineplot('Prods','R_group',hue='Algo',data=results_df[results_df.Algo != 'NoRegret'])
            sns.scatterplot('Prods','R_group',data=results_df[results_df.Algo=='NoRegret'],color='black',marker='P',s=100,label='NoRegret')
            plt.xlabel('Products Used')
            plt.ylabel('Max Group Regret Ex-Post')
            plt.title('No Regret Game Max Expected Group Regret')
            plt.savefig('Ex Post Regret by Products lineplot.png')
            plt.close(fig)

            fig = plt.figure(figsize=(20,10))
            sns.scatterplot('Prods','R_group',hue='Algo',data=results_df[~results_df.Algo.isin(['NoRegret','Greedy','GreedyMM'])])
            sns.scatterplot('Prods','R_group',data=results_df[results_df.Algo=='NoRegret'],color='black',marker='P',s=100,label='NoRegret')
            plt.xlabel('Products Used')
            plt.ylabel('Max Group Regret Ex-Post')
            plt.title('No Regret Game Max Expected Group Regret')
            plt.savefig('Ex Post Regret by Products cleaner scatter.png')
            plt.close(fig)

            fig = plt.figure(figsize=(20,10))
            sns.lineplot('Prods','R_group',hue='Algo',data=results_df[~results_df.Algo.isin(['NoRegret','Greedy','GreedyMM'])])
            sns.scatterplot('Prods','R_group',data=results_df[results_df.Algo=='NoRegret'],color='black',marker='P',s=100,label='NoRegret')
            plt.xlabel('Products Used')
            plt.ylabel('Max Group Regret Ex-Post')
            plt.title('No Regret Game Max Expected Group Regret')
            plt.savefig('Ex Post Regret by Products cleaner lineplot.png')
            plt.close(fig)

            fig = plt.figure(figsize=(20,10))
            sns.lineplot('Prods','R_group',hue='Algo',data=results_df[~results_df.Algo.isin(['NoRegret','Greedy','GreedyMM'])],ci=None)
            sns.scatterplot('Prods','R_group',data=results_df[results_df.Algo=='NoRegret'],color='black',marker='P',s=100,label='NoRegret')
            plt.xlabel('Products Used')
            plt.ylabel('Max Group Regret Ex-Post')
            plt.title('No Regret Game Max Expected Group Regret')
            plt.savefig('Ex Post Regret by Products cleaner lineplot no err.png')
            plt.close(fig)


            plt.clf()
            fig = plt.figure(figsize=(20,10))
            sns.lineplot('Prod_des','R_group',hue='Algo',data=results_df)
            plt.xlabel('Desired Products')
            plt.ylabel('Max Group Regret Ex-Post')
            plt.title('No Regret Game Max Expected Group Regret')
            plt.savefig('Ex Post Regret by Desired Products cleaner lineplot.png')
            plt.close(fig)
    except:
        print('Exp 2 failed')

    if run_exp_3:
        print('Running Experiment 3')

        # Experiment  3: varying mean difference of normals

        product_choices = [3,5,10]
        mean_difs = np.linspace(0.005,0.05,10)
        seed = 10
        random.seed(seed)
        n_instances=5
        results_3 = {'Prods':[],'Prod_des':[], 'R_group':[],'R_pop':[], 'Algo':[],'Instance':[], 'Mean_dif':[]}
        for dif in mean_difs:
            for j in range(n_instances):
                groups, risks, returns = get_grr_vary_mean_multigroup(50, 0.02, dif, 0.002, 0,
                                                                      1 / 3 * np.array([1, 1, 1]), 3)
                groups, risks, returns = sort_data(groups, risks, returns)
                np.unique(groups, return_counts=True)
                ngroups = len(np.unique(groups))
                val_counts = dict(zip(np.unique(groups, return_counts=True)[0], np.unique(groups, return_counts=True)[1]))
                groupsizes = [val_counts[g] if g in val_counts.keys() else 0 for g in range(ngroups)]
                T = 2000
                stepsize_mult = 1000

                for p in product_choices:

                    max_regret_ilp, group_regrets_ilp, products_ilp = minmax_regret_ilp.minmax_regret_ilp_wrapper(
                        returns, groups, ngroups, p, use_avg_regret=True, solver_str='GUROBI')
                    results_3['Prods'].append(p)
                    results_3['Prod_des'].append(p)
                    results_3['R_group'].append(np.max(group_regrets_ilp))
                    results_3['R_pop'].append(getregret(groupsizes,group_regrets_ilp))
                    results_3['Algo'].append('ILP')
                    results_3['Instance'].append(j)
                    results_3['Mean_dif'].append(dif)

                for p in product_choices:
                    results_greedy,prods_greedy = do_greedy(returns,groups,ngroups,groupsizes,p)
                    results_3['Prods'].append(p)
                    results_3['Prod_des'].append(p)
                    results_3['R_group'].append(results_greedy['R_max'])
                    results_3['R_pop'].append(results_greedy['R_pop'])
                    results_3['Algo'].append('Greedy')
                    results_3['Instance'].append(j)
                    results_3['Mean_dif'].append(dif)

                for p in product_choices:
                    results_greedy_mm,prods_greedy_mm = do_greedy_mm(returns,groups,ngroups,groupsizes,p)
                    results_3['Prods'].append(p)
                    results_3['Prod_des'].append(p)
                    results_3['R_group'].append(results_greedy_mm['R_max'])
                    results_3['R_pop'].append(results_greedy_mm['R_pop'])
                    results_3['Algo'].append('GreedyMM')
                    results_3['Instance'].append(j)
                    results_3['Mean_dif'].append(dif)

                for p in product_choices:
                    stepsize = calc_stepsize(stepsize_mult, returns, groups, ngroups, T)
                    results_game, prods_game = do_game(returns,groups,ngroups,groupsizes,p,stepsize,T)
                    results_3['Prods'].append(len(prods_game))
                    results_3['Prod_des'].append(p)
                    results_3['R_group'].append(results_game['R_max'])
                    results_3['R_pop'].append(results_game['R_pop'])
                    results_3['Algo'].append('NoRegret')
                    results_3['Instance'].append(j)
                    results_3['Mean_dif'].append(dif)

                    results_slack_1,slack_1_prod = do_sparsify(returns, groups, ngroups, groupsizes, p, 1, prods_game)
                    results_3['Prods'].append(p+1)
                    results_3['Prod_des'].append(p)
                    results_3['R_group'].append(results_slack_1['R_max'])
                    results_3['R_pop'].append(results_slack_1['R_pop'])
                    results_3['Algo'].append('SparseP+1')
                    results_3['Instance'].append(j)
                    results_3['Mean_dif'].append(dif)

                    results_slack_2 ,slack_2_prod = do_sparsify(returns, groups, ngroups, groupsizes, p, 2, prods_game)
                    results_3['Prods'].append(p+2)
                    results_3['Prod_des'].append(p)
                    results_3['R_group'].append(results_slack_2['R_max'])
                    results_3['R_pop'].append(results_slack_2['R_pop'])
                    results_3['Algo'].append('SparseP+2')
                    results_3['Instance'].append(j)
                    results_3['Mean_dif'].append(dif)

            df_3 = pd.DataFrame(results_3)
            fig = plt.figure(figsize=(20,10))
            sns.lineplot('Mean_dif','R_group',hue='Algo',style='Prod_des', data=df_3[~(df_3.Algo.isin(['NoRegret','SparseP+2']))])
            plt.title('Regret vs Difference in Means of distribution')
            plt.xlabel('Difference in Mean')
            plt.ylabel('Ex Post Max Group Regret')
            plt.savefig('Regret vs Dif in means.png')

            plt.close(fig)
            fig = plt.figure(figsize=(20,10))
            sns.lineplot('Mean_dif','R_group',hue='Algo',style='Prod_des', data=df_3[~(df_3.Algo.isin(['NoRegret','GreedyMM','SparseP+2']))])
            plt.title('Regret vs Difference in Means of distribution')
            plt.xlabel('Difference in Std. Deviation')
            plt.ylabel('Ex Post Max Group Regret')
            plt.savefig('Regret vs Dif in means no gmm.png')
            plt.close(fig)

            plt.close(fig)
            fig = plt.figure(figsize=(20,10))
            sns.lineplot('Mean_dif','R_group',hue='Algo',style='Prod_des', data=df_3[(df_3.Prod_des==3) & ~(df_3.Algo.isin(['NoRegret','GreedyMM','SparseP+2']))])
            plt.title('Regret vs Difference in Means of distribution')
            plt.xlabel('Difference in Std. Deviation')
            plt.ylabel('Ex Post Max Group Regret')
            plt.savefig('Regret vs Dif in means no gmm 3 prod.png')
            plt.close(fig)

            plt.close(fig)
            fig = plt.figure(figsize=(20,10))
            sns.lineplot('Mean_dif','R_group',hue='Algo',style='Prod_des', data=df_3[(df_3.Prod_des==10) & ~(df_3.Algo.isin(['NoRegret','GreedyMM','SparseP+2']))])
            plt.title('Regret vs Difference in Means of distribution')
            plt.xlabel('Difference in Std. Deviation')
            plt.ylabel('Ex Post Max Group Regret')
            plt.savefig('Regret vs Dif in means no gmm 10 prod.png')
            plt.close(fig)


    if run_exp_4:
        print('Running Experiment 4')

        # Experiment 4: varying std difference of normals

        product_choices = [3, 5, 10]
        std_dif = np.linspace(0.001, 0.01, 10)
        seed = 10
        random.seed(seed)
        n_instances = 5
        results_4 = {'Prods': [], 'Prod_des': [], 'R_group': [], 'R_pop': [], 'Algo': [], 'Instance': [],
                     'Std_dif': []}
        for dif in std_dif:
            for j in range(n_instances):
                groups, risks, returns = get_grr_vary_mean_multigroup(50, 0.02, 0, 0.002, dif,
                                                                      1 / 3 * np.array([1, 1, 1]), 3)
                groups, risks, returns = sort_data(groups, risks, returns)
                np.unique(groups, return_counts=True)
                ngroups = len(np.unique(groups))
                val_counts = dict(
                    zip(np.unique(groups, return_counts=True)[0], np.unique(groups, return_counts=True)[1]))
                groupsizes = [val_counts[g] if g in val_counts.keys() else 0 for g in range(ngroups)]
                T = 2000
                stepsize_mult = 1000

                for p in product_choices:
                    max_regret_ilp, group_regrets_ilp, products_ilp = minmax_regret_ilp.minmax_regret_ilp_wrapper(
                        returns, groups, ngroups, p, use_avg_regret=True, solver_str='GUROBI')
                    results_4['Prods'].append(p)
                    results_4['Prod_des'].append(p)
                    results_4['R_group'].append(np.max(group_regrets_ilp))
                    results_4['R_pop'].append(getregret(groupsizes, group_regrets_ilp))
                    results_4['Algo'].append('ILP')
                    results_4['Instance'].append(j)
                    results_4['Std_dif'].append(dif)

                for p in product_choices:
                    results_greedy, prods_greedy = do_greedy(returns, groups, ngroups, groupsizes, p)
                    results_4['Prods'].append(p)
                    results_4['Prod_des'].append(p)
                    results_4['R_group'].append(results_greedy['R_max'])
                    results_4['R_pop'].append(results_greedy['R_pop'])
                    results_4['Algo'].append('Greedy')
                    results_4['Instance'].append(j)
                    results_4['Std_dif'].append(dif)

                for p in product_choices:
                    results_greedy_mm, prods_greedy_mm = do_greedy_mm(returns, groups, ngroups, groupsizes, p)
                    results_4['Prods'].append(p)
                    results_4['Prod_des'].append(p)
                    results_4['R_group'].append(results_greedy_mm['R_max'])
                    results_4['R_pop'].append(results_greedy_mm['R_pop'])
                    results_4['Algo'].append('GreedyMM')
                    results_4['Instance'].append(j)
                    results_4['Std_dif'].append(dif)

                for p in product_choices:
                    stepsize = calc_stepsize(stepsize_mult, returns, groups, ngroups, T)
                    results_game, prods_game = do_game(returns, groups, ngroups, groupsizes, p, stepsize, T)
                    results_4['Prods'].append(len(prods_game))
                    results_4['Prod_des'].append(p)
                    results_4['R_group'].append(results_game['R_max'])
                    results_4['R_pop'].append(results_game['R_pop'])
                    results_4['Algo'].append('NoRegret')
                    results_4['Instance'].append(j)
                    results_4['Std_dif'].append(dif)

                    results_slack_1, slack_1_prod = do_sparsify(returns, groups, ngroups, groupsizes, p, 1, prods_game)
                    results_4['Prods'].append(p + 1)
                    results_4['Prod_des'].append(p)
                    results_4['R_group'].append(results_slack_1['R_max'])
                    results_4['R_pop'].append(results_slack_1['R_pop'])
                    results_4['Algo'].append('SparseP+1')
                    results_4['Instance'].append(j)
                    results_4['Std_dif'].append(dif)

                    results_slack_2, slack_2_prod = do_sparsify(returns, groups, ngroups, groupsizes, p, 2, prods_game)
                    results_4['Prods'].append(p + 2)
                    results_4['Prod_des'].append(p)
                    results_4['R_group'].append(results_slack_2['R_max'])
                    results_4['R_pop'].append(results_slack_2['R_pop'])
                    results_4['Algo'].append('SparseP+2')
                    results_4['Instance'].append(j)
                    results_4['Std_dif'].append(dif)
            df_4 = pd.DataFrame(results_4)
            fig = plt.figure(figsize=(20, 10))
            sns.lineplot('Std_dif', 'R_group', hue='Algo', style='Prod_des',data=df_4[~(df_4.Algo.isin(['NoRegret', 'SparseP+2']))])
            plt.xlabel('Difference in Std. Deviation')
            plt.ylabel('Ex Post Max Group Regret')
            plt.title('Regret vs Difference in Std of distribution')
            plt.savefig('Regret v stdev .png')
            plt.close(fig)

            fig = plt.figure(figsize=(20, 10))
            sns.lineplot('Std_dif', 'R_group', hue='Algo', style='Prod_des',data=df_4[~(df_4.Algo.isin(['NoRegret','GreedyMM', 'SparseP+2']))])
            plt.xlabel('Difference in Std. Deviation')
            plt.ylabel('Ex Post Max Group Regret')
            plt.title('Regret vs Difference in Std of distribution')
            plt.savefig('Regret v stdev no gmm.png')
            plt.close(fig)

            fig = plt.figure(figsize=(20, 10))
            sns.lineplot('Std_dif', 'R_group', hue='Algo',data=df_4[(df_4.Prod_des==10) & ~(df_4.Algo.isin(['NoRegret','GreedyMM', 'SparseP+2']))])
            plt.xlabel('Difference in Std. Deviation')
            plt.ylabel('Ex Post Max Group Regret')
            plt.title('Regret vs Difference in Std of distribution')
            plt.savefig('Regret v stdev no gmm 10prod .png')
            plt.close(fig)

            fig = plt.figure(figsize=(20, 10))
            sns.lineplot('Std_dif', 'R_group', hue='Algo',data=df_4[(df_4.Prod_des==3) & ~(df_4.Algo.isin(['NoRegret','GreedyMM', 'SparseP+2']))])
            plt.xlabel('Difference in Std. Deviation')
            plt.ylabel('Ex Post Max Group Regret')
            plt.title('Regret vs Difference in Std of distribution')
            plt.savefig('Regret v stdev no gmm 3 prod .png')
            plt.close(fig)
    if run_exp_5:
        print('Running Experiment 5')

        # Experiment 5: varying normal mult

        product_choices = [3, 5, 10]
        dist_mults = 0.5*np.linspace(2, 12, 11)
        seed = 10
        random.seed(seed)
        n_instances = 5
        results_5 = {'Prods': [], 'Prod_des': [], 'R_group': [], 'R_pop': [], 'Algo': [], 'Instance': [],
                     'Mult': []}
        for dist_mult in dist_mults:
            for j in range(n_instances):
                groups, risks, returns = get_grr_vary_mean_multigroup(50, 0.02, dist_mult*0.02 , 0.002, dist_mult*0.002,
                                                                      1 / 3 * np.array([1, 1, 1]), 3)
                groups, risks, returns = sort_data(groups, risks, returns)
                np.unique(groups, return_counts=True)
                ngroups = len(np.unique(groups))
                val_counts = dict(
                    zip(np.unique(groups, return_counts=True)[0], np.unique(groups, return_counts=True)[1]))
                groupsizes = [val_counts[g] if g in val_counts.keys() else 0 for g in range(ngroups)]
                T = 2000
                stepsize_mult = 1000

                for p in product_choices:
                    max_regret_ilp, group_regrets_ilp, products_ilp = minmax_regret_ilp.minmax_regret_ilp_wrapper(
                        returns, groups, ngroups, p, use_avg_regret=True, solver_str='GUROBI')
                    results_5['Prods'].append(p)
                    results_5['Prod_des'].append(p)
                    results_5['R_group'].append(np.max(group_regrets_ilp))
                    results_5['R_pop'].append(getregret(groupsizes, group_regrets_ilp))
                    results_5['Algo'].append('ILP')
                    results_5['Instance'].append(j)
                    results_5['Mult'].append(dist_mult)

                for p in product_choices:
                    results_greedy, prods_greedy = do_greedy(returns, groups, ngroups, groupsizes, p)
                    results_5['Prods'].append(p)
                    results_5['Prod_des'].append(p)
                    results_5['R_group'].append(results_greedy['R_max'])
                    results_5['R_pop'].append(results_greedy['R_pop'])
                    results_5['Algo'].append('Greedy')
                    results_5['Instance'].append(j)
                    results_5['Mult'].append(dist_mult)

                for p in product_choices:
                    results_greedy_mm, prods_greedy_mm = do_greedy_mm(returns, groups, ngroups, groupsizes, p)
                    results_5['Prods'].append(p)
                    results_5['Prod_des'].append(p)
                    results_5['R_group'].append(results_greedy_mm['R_max'])
                    results_5['R_pop'].append(results_greedy_mm['R_pop'])
                    results_5['Algo'].append('GreedyMM')
                    results_5['Instance'].append(j)
                    results_5['Mult'].append(dist_mult)

                for p in product_choices:
                    stepsize = calc_stepsize(stepsize_mult, returns, groups, ngroups, T)
                    results_game, prods_game = do_game(returns, groups, ngroups, groupsizes, p, stepsize, T)
                    results_5['Prods'].append(len(prods_game))
                    results_5['Prod_des'].append(p)
                    results_5['R_group'].append(results_game['R_max'])
                    results_5['R_pop'].append(results_game['R_pop'])
                    results_5['Algo'].append('NoRegret')
                    results_5['Instance'].append(j)
                    results_5['Mult'].append(dist_mult)

                    results_slack_1, slack_1_prod = do_sparsify(returns, groups, ngroups, groupsizes, p, 1,
                                                                prods_game)
                    results_5['Prods'].append(p + 1)
                    results_5['Prod_des'].append(p)
                    results_5['R_group'].append(results_slack_1['R_max'])
                    results_5['R_pop'].append(results_slack_1['R_pop'])
                    results_5['Algo'].append('SparseP+1')
                    results_5['Instance'].append(j)
                    results_5['Mult'].append(dist_mult)

                    results_slack_2, slack_2_prod = do_sparsify(returns, groups, ngroups, groupsizes, p, 2,
                                                                prods_game)
                    results_5['Prods'].append(p + 2)
                    results_5['Prod_des'].append(p)
                    results_5['R_group'].append(results_slack_2['R_max'])
                    results_5['R_pop'].append(results_slack_2['R_pop'])
                    results_5['Algo'].append('SparseP+2')
                    results_5['Instance'].append(j)
                    results_5['Mult'].append(dist_mult)
        df_5 = pd.DataFrame(results_5)
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Mult', 'R_group', hue='Algo', style='Prod_des',data=df_5[~(df_5.Algo.isin(['NoRegret','SparseP+2']))])
        plt.xlabel('Distribution Multiplier')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Distribution Multiplier')
        plt.savefig('Regret v distribution multiplier.png')
        plt.close(fig)

        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Mult', 'R_group', hue='Algo', style='Prod_des',data=df_5[~(df_5.Algo.isin(['NoRegret','GreedyMM','SparseP+2']))])
        plt.xlabel('Distribution Multiplier')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Distribution Multiplier')
        plt.savefig('Regret v distribution multiplier no gmm.png')
        plt.close(fig)

        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Mult', 'R_group', hue='Algo', style='Prod_des',data=df_5[(df_5.Prod_des==3) & ~(df_5.Algo.isin(['NoRegret','GreedyMM','SparseP+2']))])
        plt.xlabel('Distribution Multiplier')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Distribution Multiplier')
        plt.savefig('Regret v distribution multiplier no gmm 3 prod.png')
        plt.close(fig)
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Mult', 'R_group', hue='Algo', style='Prod_des',
                     data=df_5[(df_5.Prod_des == 10) & ~(df_5.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
        plt.xlabel('Distribution Multiplier')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Distribution Multiplier')
        plt.savefig('Regret v distribution multiplier no gmm 10 prod.png')
        plt.close(fig)
    if run_exp_6:
        print('Running Experiment 6')

        # Experiment 6: Rare group

        product_choices = [3, 5, 10]
        lambs = np.linspace(0.1, 1,10)
        seed = 10
        random.seed(seed)
        n_instances = 5
        stepsize_mult = 2000
        results_6 = {'Prods': [], 'Prod_des': [], 'R_group': [], 'R_pop': [], 'Algo': [], 'Instance': [],
                     'Lambda': []}
        for lamb in lambs:
            for j in range(n_instances):
                groups, risks, returns = get_grr_vary_mean_rare(50, 0.02, 0.01 , 0.002, 0.001,[0.5,0.4],2, 0.1, lamb)
                groups, risks, returns = sort_data(groups, risks, returns)
                np.unique(groups, return_counts=True)
                ngroups = len(np.unique(groups))
                val_counts = dict(
                    zip(np.unique(groups, return_counts=True)[0], np.unique(groups, return_counts=True)[1]))
                groupsizes = [val_counts[g] if g in val_counts.keys() else 0 for g in range(ngroups)]
                T = 2000
                stepsize_mult = 5000

                for p in product_choices:
                    max_regret_ilp, group_regrets_ilp, products_ilp = minmax_regret_ilp.minmax_regret_ilp_wrapper(returns, groups, ngroups, p, use_avg_regret=True, solver_str='GUROBI')
                    results_6['Prods'].append(p)
                    results_6['Prod_des'].append(p)
                    results_6['R_group'].append(np.max(group_regrets_ilp))
                    results_6['R_pop'].append(getregret(groupsizes, group_regrets_ilp))
                    results_6['Algo'].append('ILP')
                    results_6['Instance'].append(j)
                    results_6['Lambda'].append(lamb)

                for p in product_choices:
                    results_greedy, prods_greedy = do_greedy(returns, groups, ngroups, groupsizes, p)
                    results_6['Prods'].append(p)
                    results_6['Prod_des'].append(p)
                    results_6['R_group'].append(results_greedy['R_max'])
                    results_6['R_pop'].append(results_greedy['R_pop'])
                    results_6['Algo'].append('Greedy')
                    results_6['Instance'].append(j)
                    results_6['Lambda'].append(lamb)

                for p in product_choices:
                    results_greedy_mm, prods_greedy_mm = do_greedy_mm(returns, groups, ngroups, groupsizes, p)
                    results_6['Prods'].append(p)
                    results_6['Prod_des'].append(p)
                    results_6['R_group'].append(results_greedy_mm['R_max'])
                    results_6['R_pop'].append(results_greedy_mm['R_pop'])
                    results_6['Algo'].append('GreedyMM')
                    results_6['Instance'].append(j)
                    results_6['Lambda'].append(lamb)

                for p in product_choices:
                    stepsize = calc_stepsize(stepsize_mult, returns, groups, ngroups, T)
                    results_game, prods_game = do_game(returns, groups, ngroups, groupsizes, p, stepsize, T)
                    results_6['Prods'].append(len(prods_game))
                    results_6['Prod_des'].append(p)
                    results_6['R_group'].append(results_game['R_max'])
                    results_6['R_pop'].append(results_game['R_pop'])
                    results_6['Algo'].append('NoRegret')
                    results_6['Instance'].append(j)
                    results_6['Lambda'].append(lamb)

                    results_slack_1, slack_1_prod = do_sparsify(returns, groups, ngroups, groupsizes, p, 1,
                                                                prods_game)
                    results_6['Prods'].append(p + 1)
                    results_6['Prod_des'].append(p)
                    results_6['R_group'].append(results_slack_1['R_max'])
                    results_6['R_pop'].append(results_slack_1['R_pop'])
                    results_6['Algo'].append('SparseP+1')
                    results_6['Instance'].append(j)
                    results_6['Lambda'].append(lamb)

                    results_slack_2, slack_2_prod = do_sparsify(returns, groups, ngroups, groupsizes, p, 2,
                                                                prods_game)
                    results_6['Prods'].append(p + 2)
                    results_6['Prod_des'].append(p)
                    results_6['R_group'].append(results_slack_2['R_max'])
                    results_6['R_pop'].append(results_slack_2['R_pop'])
                    results_6['Algo'].append('SparseP+2')
                    results_6['Instance'].append(j)
                    results_6['Lambda'].append(lamb)

        df_6 = pd.DataFrame(results_6)
        df_6 = df_6.rename({'Lambda':'Beta'},axis=1)
        df_6['Lambda'] = df_6.Beta**(-1)
        fig = plt.figure(figsize=(20, 10))

        sns.lineplot('Lambda', 'R_pop', hue='Algo', style='Prod_des',data=df_6[~(df_6.Algo.isin(['NoRegret', 'SparseP+2']))])
        plt.xlabel('Lambda for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare.png')

        plt.close(fig)
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Lambda', 'R_pop', hue='Algo', style='Prod_des',data=df_6[~(df_6.Algo.isin(['NoRegret','GreedyMM' ,'SparseP+2']))])
        plt.xlabel('Lambda for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare no gmm.png')
        plt.close(fig)

        plt.close(fig)
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Lambda', 'R_pop', hue='Algo',data=df_6[(df_6.Prod_des==10) & ~(df_6.Algo.isin(['NoRegret','GreedyMM' ,'SparseP+2']))])
        plt.xlabel('Lambda for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare no gmm 10prod.png')
        plt.close(fig)

        plt.close(fig)
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Lambda', 'R_pop', hue='Algo',data=df_6[(df_6.Prod_des==3) & ~(df_6.Algo.isin(['NoRegret','GreedyMM' ,'SparseP+2']))])
        plt.xlabel('Lambda for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare no gmm 3 prod.png')
        plt.close(fig)

        fig = plt.figure(figsize=(20, 10))

        sns.lineplot('Beta', 'R_pop', hue='Algo', style='Prod_des',
                     data=df_6[~(df_6.Algo.isin(['NoRegret', 'SparseP+2']))])
        plt.xlabel('Beta for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare beta.png')

        plt.close(fig)
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Beta', 'R_pop', hue='Algo', style='Prod_des',
                     data=df_6[~(df_6.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
        plt.xlabel('Beta for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare no gmm beta.png')
        plt.close(fig)

        plt.close(fig)
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Beta', 'R_pop', hue='Algo',
                     data=df_6[(df_6.Prod_des == 10) & ~(df_6.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
        plt.xlabel('Beta for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare no gmm 10prod beta.png')
        plt.close(fig)

        plt.close(fig)
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Beta', 'R_pop', hue='Algo',
                     data=df_6[(df_6.Prod_des == 3) & ~(df_6.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
        plt.xlabel('Beta for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare no gmm 3 prod beta.png')
        plt.close(fig)


        plt.close(fig)
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Lambda', 'R_pop', hue='Algo',data=df_6[(df_6.Prod_des==3) & ~(df_6.Algo.isin(['NoRegret','GreedyMM' ,'SparseP+2']))])
        plt.xlabel('Lambda for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare no gmm 3 prod.png')
        plt.close(fig)

        fig = plt.figure(figsize=(20, 10))

        sns.lineplot('Beta', 'R_pop', hue='Algo', style='Prod_des',
                     data=df_6[~(df_6.Algo.isin(['NoRegret', 'SparseP+2']))])
        plt.xlabel('Beta for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare beta.png')

        plt.close(fig)
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Beta', 'R_pop', hue='Algo', style='Prod_des',
                     data=df_6[~(df_6.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
        plt.xlabel('Beta for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare no gmm beta.png')
        plt.close(fig)

        plt.close(fig)
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Beta', 'R_pop', hue='Algo',
                     data=df_6[(df_6.Prod_des == 10) & ~(df_6.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
        plt.xlabel('Beta for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare no gmm 10prod beta.png')
        plt.close(fig)

        plt.close(fig)
        fig = plt.figure(figsize=(20, 10))
        sns.lineplot('Beta', 'R_pop', hue='Algo',
                     data=df_6[(df_6.Prod_des == 3) & ~(df_6.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
        plt.xlabel('Beta for Rare Group')
        plt.ylabel('Ex Post Max Group Regret')
        plt.title('Regret vs Rare Group mean')
        plt.savefig('Regret rare no gmm 3 prod beta.png')
        plt.close(fig)


    if run_exp_7:
        print('Running Experiment 7')

        # Experiment 7: Generalization
        product_choices = [3, 5, 10]
        seed = 10
        random.seed(seed)
        results_7 = {'Prods': [], 'Prod_des': [], 'R_group_in': [], 'R_pop_in': [], 'R_group_out': [], 'R_pop_out': [], 'Algo': [], 'Instance': [],'N_cons_in': []}
        n_consumer_choices = np.linspace(20,200,19)
        n_instances = 5
        T=500
        stepsize_mult=10000
        for j in range(n_instances):
            out_groups,out_risks,out_returns = get_grr_vary_mean_multigroup(50,0.02, 0.01,0.002,0.001,1/3*np.array([1,1,1]),3)
            out_groups,out_risks,out_returns = sort_data(out_groups,out_risks,out_returns)
            out_ngroups = len(np.unique(out_groups))
            out_val_counts = dict(
                zip(np.unique(out_groups, return_counts=True)[0], np.unique(out_groups, return_counts=True)[1]))
            out_groupsizes = [out_val_counts[g] if g in out_val_counts.keys() else 0 for g in range(out_ngroups)]

            for n_cons in n_consumer_choices:
                in_groups,in_risks,in_returns = get_grr_vary_mean_multigroup(int(n_cons),0.02, 0.01,0.002,0.001,1/3*np.array([1,1,1]),3)
                in_groups,in_risks,in_returns = sort_data(in_groups,in_risks,in_returns)
                np.unique(in_groups,return_counts=True)
                in_ngroups=len(np.unique(in_groups))
                in_val_counts = dict(zip(np.unique(in_groups, return_counts=True)[0], np.unique(in_groups, return_counts=True)[1]))
                in_groupsizes = [in_val_counts[g] if g in in_val_counts.keys() else 0 for g in range(in_ngroups)]
                #stepsize_mult = 10000
                #T = 200

                for prod in product_choices:
                    stepsize = calc_stepsize(stepsize_mult, in_returns, in_groups, in_ngroups, T)
                    results_game, prods_game = do_game(in_returns, in_groups, in_ngroups, in_groupsizes, prod, stepsize, T)
                    #group_regrets_mm_out = get_group_regrets(out_returns, out_groups, out_ngroups, prods_game,
                         #             use_avg_regret=True)
                    actual_prods = in_returns[prods_game]
                    if 0 not in actual_prods:
                        actual_prods = np.insert(actual_prods,0,0)
                    group_regrets_mm_out = get_group_regrets_nonconsumer_prods(out_returns, out_groups, out_ngroups, actual_prods, use_avg_regret=True)
                    regret_out_max = np.max(group_regrets_mm_out)
                    if len(out_groupsizes) > len(group_regrets_mm_out):
                        out_groupsizes = [size for size in out_groupsizes if size > 0]
                    group_regrets_mm_out = np.array(group_regrets_mm_out)
                    out_groupsizes = np.array(out_groupsizes)
                    regret_out_pop = getregret(out_groupsizes, group_regrets_mm_out)

                    results_7['Prods'].append(len(prods_game))
                    results_7['Prod_des'].append(prod)
                    results_7['R_group_in'].append(results_game['R_max'])
                    results_7['R_pop_in'].append(results_game['R_pop'])
                    results_7['Algo'].append('NoRegret')
                    results_7['Instance'].append(j)
                    results_7['R_group_out'].append(regret_out_max)
                    results_7['R_pop_out'].append(regret_out_pop)
                    results_7['N_cons_in'].append(n_cons)


                    results_slack_1, slack_1_prod = do_sparsify(in_returns, in_groups, in_ngroups, in_groupsizes, prod, 1,
                                                                prods_game)

                    actual_prods = in_returns[slack_1_prod]

                    if 0 not in actual_prods:
                        actual_prods = np.insert(actual_prods,0,0)

                    group_regrets_sp1_out = get_group_regrets_nonconsumer_prods(out_returns, out_groups, out_ngroups, actual_prods, use_avg_regret=True)

                    regret_max_out_sp1 = np.max(group_regrets_sp1_out)
                    if len(out_groupsizes) > len(group_regrets_sp1_out):
                        out_groupsizes = [size for size in out_groupsizes if size > 0]
                    group_regrets_sp1_out = np.array(group_regrets_sp1_out)
                    out_groupsizes = np.array(out_groupsizes)
                    regret_pop_out_sp1 = getregret(out_groupsizes, group_regrets_mm_out)

                    results_7['Prods'].append(len(slack_1_prod))
                    results_7['Prod_des'].append(prod)
                    results_7['R_group_in'].append(results_slack_1['R_max'])
                    results_7['R_pop_in'].append(results_slack_1['R_pop'])
                    results_7['Algo'].append('SparseP+1')
                    results_7['Instance'].append(j)
                    results_7['R_group_out'].append(regret_max_out_sp1)
                    results_7['R_pop_out'].append(regret_pop_out_sp1)
                    results_7['N_cons_in'].append(n_cons)
        df_7 = pd.DataFrame(results_7)

        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
        sns.lineplot('N_cons_in', 'value', hue='Algo', style='variable',data=pd.DataFrame(melted_7).reset_index(),ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization')
        plt.savefig('generalization.png')

        plt.clf()
        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
        sns.lineplot('N_cons_in', 'value', style='variable',data=pd.DataFrame(melted_7[(melted_7.Algo=='NoRegret') &( melted_7.Prod_des==10)]).reset_index(),ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization - 10 products desired')
        plt.savefig('generalization 10.png')

        plt.clf()
        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
        sns.lineplot('N_cons_in', 'value', style='variable',data=pd.DataFrame(melted_7[(melted_7.Algo=='NoRegret') &( melted_7.Prod_des==3)]).reset_index(),ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization - 3 products desired')
        plt.savefig('generalization 3.png')

        plt.clf()
        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
        sns.lineplot('N_cons_in', 'value', style='variable',data=pd.DataFrame(melted_7[(melted_7.Algo=='NoRegret') &( melted_7.Prod_des==5)]).reset_index(),ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization - 5 products desired')
        plt.savefig('generalization 5.png')

        plt.clf()
        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
        sns.lineplot('N_cons_in', 'value', style='variable',
                     data=pd.DataFrame(melted_7[(melted_7.Algo == 'SparseP+1') & (melted_7.Prod_des == 10)]).reset_index(),
                     ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization - 10 products desired (Sparse P+1)')
        plt.savefig('generalization 10 sparse.png')

        plt.clf()
        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
        sns.lineplot('N_cons_in', 'value', style='variable',
                     data=pd.DataFrame(melted_7[(melted_7.Algo == 'SparseP+1') & (melted_7.Prod_des == 3)]).reset_index(),
                     ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization - 3 products desired (Sparse P+1)')
        plt.savefig('generalization 3 sparse .png')

        plt.clf()
        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
        sns.lineplot('N_cons_in', 'value', style='variable',
                     data=pd.DataFrame(melted_7[(melted_7.Algo == 'SparseP+1') & (melted_7.Prod_des == 5)]).reset_index(),
                     ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization - 5 products desired (Sparse P+1)')
        plt.savefig('generalization 5 sparse.png')

        plt.clf()
        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
        sns.lineplot('N_cons_in', 'value', style='variable',
                     data=pd.DataFrame(melted_7[(melted_7.Algo == 'NoRegret') & (melted_7.Prod_des == 10)]).reset_index(),
                     ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization - 10 products desired, Population Regret')
        plt.savefig('generalization 10 pop.png')

        plt.clf()
        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
        sns.lineplot('N_cons_in', 'value', style='variable',
                     data=pd.DataFrame(melted_7[(melted_7.Algo == 'NoRegret') & (melted_7.Prod_des == 3)]).reset_index(),
                     ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization - 3 products desired, Population Regret')
        plt.savefig('generalization 3 pop.png')

        plt.clf()
        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
        sns.lineplot('N_cons_in', 'value', style='variable',
                     data=pd.DataFrame(melted_7[(melted_7.Algo == 'NoRegret') & (melted_7.Prod_des == 5)]).reset_index(),
                     ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization - 5 products desired, Population Regret')
        plt.savefig('generalization 5 pop.png')

        plt.clf()
        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
        sns.lineplot('N_cons_in', 'value', style='variable',
                     data=pd.DataFrame(melted_7[(melted_7.Algo == 'SparseP+1') & (melted_7.Prod_des == 10)]).reset_index(),
                     ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization - 10 products desired (Sparse P+1, Population Regret)')
        plt.savefig('generalization 10 sparse pop.png')

        plt.clf()
        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
        sns.lineplot('N_cons_in', 'value', style='variable',
                     data=pd.DataFrame(melted_7[(melted_7.Algo == 'SparseP+1') & (melted_7.Prod_des == 3)]).reset_index(),
                     ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization - 3 products desired (Sparse P+1, Population Regret)')
        plt.savefig('generalization 3 sparse pop.png')

        plt.clf()
        melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
        sns.lineplot('N_cons_in', 'value', style='variable',
                     data=pd.DataFrame(melted_7[(melted_7.Algo == 'SparseP+1') & (melted_7.Prod_des == 5)]).reset_index(),
                     ci=None)
        plt.ylabel('Regret')
        plt.xlabel('Number of Consumers in Sample')
        plt.title('Generalization - 5 products desired (Sparse P+1, Population Regret)')
        plt.savefig('generalization 5 sparse pop.png')



    if run_exp_8:
        print('Running Experiment 8')

        ###experiment 8: performance of sparsify
        seed = 10
        random.seed(seed)
        results_8 = {'R_pop':[],'R_group':[],'R_dif':[],'subinst':[],'Additional Products':[]}
        n_instances = 10
        T= 200
        stepsize_mult=10000
        n_cons= 50
        mean_lower = 0.02
        mean_dif = 0.01
        std_lower=.002
        std_dif= 0.001
        ps = 1 / 3 * np.array([1, 1, 1])
        ngroups = 3
        init_products = 5
        max_ad_products = 10
        for inst in range(n_instances):
            groups,risks,returns = get_grr_vary_mean_multigroup(n_cons,mean_lower, mean_dif,std_lower,std_dif,ps,ngroups)
            groups,risks,returns = sort_data(groups,risks,returns)
            np.unique(groups,return_counts=True)
            ngroups=len(np.unique(groups))
            val_counts = dict(zip(np.unique(groups, return_counts=True)[0], np.unique(groups, return_counts=True)[1]))
            groupsizes = [val_counts[g] if g in val_counts.keys() else 0 for g in range(ngroups)]
            stepsize = calc_stepsize(stepsize_mult, returns, groups, ngroups, T)
            results_game, prods_game = do_game(returns,groups,ngroups,groupsizes,init_products,stepsize,T)
            results_8['R_pop'].append(results_game['R_pop'])
            results_8['R_group'].append(results_game['R_max'])
            results_8['R_dif'].append(results_game['R_dif'])
            results_8['subinst'].append(inst)
            results_8['Additional Products'].append(-1)

            for k in range(0,max_ad_products+1):
                #as an aside, check to make sure do_sparsify is workign correctly. returns same results if slack==len(prods_game)-init_prods.
                results_sparsify, prods_sparsify = do_sparsify(returns,groups,ngroups,groupsizes,init_products,k,prods_game)
                results_8['R_pop'].append(results_sparsify['R_pop'])
                results_8['R_group'].append(results_sparsify['R_max'])
                results_8['R_dif'].append(results_sparsify['R_dif'])
                results_8['subinst'].append(inst)
                results_8['Additional Products'].append(k)

        df_8 = pd.DataFrame(results_8)
        df_8.rename({'R_group': 'Max Group Avg Regret','R_pop': 'Population Avg Regret'},axis=1,inplace=True)
        plt.clf()
        sns.lineplot('Additional Products', 'Max Group Avg Regret',data=df_8[df_8['Additional Products']>-1])
        plt.plot([i for i in range(11)],unsparse_group ,'b--', label='Max Group Regret Unsparsified')
        plt.legend()
        plt.title('Group Regret as Sparsifier allowed more products')
        plt.savefig('Sparsification Group Regret.png')
        plt.clf()
        melted_8 = pd.melt(df_8, id_vars=['Additional Products'], value_vars=['Max Group Avg Regret', 'Population Avg Regret'])
        sns.lineplot('Additional Products','value',hue='variable',data=melted_8[melted_8['Additional Products']>-1])
        unsparse_group = [df_8[df_8['Additional Products'] == -1]['Max Group Avg Regret'].mean() for i in range(11)]
        unsparse_pop = [df_8[df_8['Additional Products'] == -1]['Population Avg Regret'].mean() for i in range(11)]
        plt.plot([i for i in range(11)],unsparse_group ,'b--', label='Max Group Regret Unsparsified')
        plt.plot([i for i in range(11)],unsparse_pop ,'r--', label='Pop Regret Unsparsified')
        plt.legend()
        plt.title('Regret as Sparsifier allowed more products')
        plt.savefig('Sparisfication group pop regret.png')
        plt.clf()

    try:
        if run_exp_9:
            print('Running Experiment 9')

            ## Experiment 9 - performance
            seed = 10
            random.seed(seed)
            results_9 = {'R_pop':[],'R_group':[],'R_dif':[],'subinst':[],'Products Used':[],'Algo':[],'Time (seconds)':[]}
            n_instances = 100
            T= 500
            stepsize_mult=1000
            n_cons= 50
            mean_lower = 0.02
            mean_dif = 0.01
            std_lower=.002
            std_dif= 0.0001
            #mean_lower = 0.001
            #mean_dif = 0.005
            #std_lower = 0.0001
            #std_dif = 0.0005
            ps = 1 / 3 * np.array([1, 1, 1])
            ngroups = 3
            products = 5
            consumer_weights_dp = 1/n_cons*np.array([1 for i in range(n_cons)])
            def append_row_9(results,algo, prod_used, subinst,time_taken):
                results_9['R_group'].append(results['R_max'])
                results_9['R_pop'].append(results['R_pop'])
                results_9['R_dif'].append(results['R_dif'])
                results_9['subinst'].append(subinst)
                results_9['Algo'].append(algo)
                results_9['Products Used'].append(prod_used)
                results_9['Time (seconds)'].append(time_taken)
            for instance in range(n_instances):
                start_time_instance = time.time()
                print('instance: '+str(instance))
                groups,risks,returns = get_grr_vary_mean_multigroup(n_cons,mean_lower, mean_dif,std_lower,std_dif,ps,ngroups)
                groups,risks,returns = sort_data(groups,risks,returns)
                np.unique(groups,return_counts=True)
                ngroups=3
                val_counts = dict(zip(np.unique(groups, return_counts=True)[0], np.unique(groups, return_counts=True)[1]))
                groupsizes = [val_counts[g] if g in val_counts.keys() else 0 for g in range(ngroups)]
                stepsize = calc_stepsize(stepsize_mult, returns, groups, ngroups, T)

                start = time.time()
                results_game, prods_game = do_game(returns,groups,ngroups,groupsizes,products,stepsize,T)
                end = time.time()
                time_game = end-start

                start = time.time()
                results_sparsify_0, prods_sparsify_0 = do_sparsify(returns,groups,ngroups,groupsizes,products,0,prods_game)
                end = time.time()
                time_sp_0 = time_game+  end - start

                start = time.time()
                results_sparsify_1, prods_sparsify_1 = do_sparsify(returns,groups,ngroups,groupsizes,products,1,prods_game)
                end = time.time()
                time_sp_1 = time_game+  end - start

                start = time.time()
                results_sparsify_2, prods_sparsify_2 = do_sparsify(returns,groups,ngroups,groupsizes,products,2,prods_game)
                end = time.time()
                time_sp_2 = time_game+  end - start

                start = time.time()
                results_sparsify_3, prods_sparsify_3 = do_sparsify(returns,groups,ngroups,groupsizes,products,3,prods_game)
                end = time.time()
                time_sp_3 = time_game+  end - start

                start = time.time()
                results_sparsify_4, prods_sparsify_4 = do_sparsify(returns,groups,ngroups,groupsizes,products,4,prods_game)
                end = time.time()
                time_sp_4 = time_game+  end - start

                start = time.time()
                results_DP, prods_DP = do_DP(returns,groups,groupsizes, ngroups,products,consumer_weights_dp)
                end = time.time()
                time_DP = end - start

                start = time.time()
                results_ILP, prods_ILP = do_ILP(returns,groups,ngroups,products,use_avg_regret=True)
                end = time.time()
                time_ILP = end - start

                start = time.time()
                results_greedy, prods_greedy = do_greedy(returns,groups,ngroups,groupsizes,products)
                end = time.time()
                time_greedy = end - start

                append_row_9(results_game, 'No Regret', len(prods_game), instance,time_game)
                append_row_9(results_sparsify_0, 'Sparse+0', len(prods_sparsify_0),instance,time_sp_0)
                append_row_9(results_sparsify_1, 'Sparse+1', len(prods_sparsify_1),instance,time_sp_1)
                append_row_9(results_sparsify_2, 'Sparse+2', len(prods_sparsify_2),instance,time_sp_2)
                append_row_9(results_sparsify_3, 'Sparse+3', len(prods_sparsify_3),instance,time_sp_3)
                append_row_9(results_sparsify_4, 'Sparse+4', len(prods_sparsify_4),instance,time_sp_4)
                append_row_9(results_DP, 'DP', len(prods_DP),instance,time_DP)
                append_row_9(results_ILP, 'ILP', len(prods_ILP),instance,time_ILP)
                append_row_9(results_greedy, 'Greedy', len(prods_greedy),instance,time_greedy)
                end_time_instance = time.time()
                instance_time = end_time_instance-start_time_instance
                print(instance_time)
            df_9 = pd.DataFrame(results_9)
            df_9.rename({'R_pop':'Population Regret', 'R_group':'Group Regret'}, axis=1,inplace=True)
            plt.clf()
            fig = plt.figure(figsize=(8,6))
            ax = sns.barplot(x="Algo", y="Population Regret", data=df_9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            plt.savefig('performance_population.png')
            plt.clf()
            fig = plt.figure(figsize=(8,6))
            ax = sns.barplot(x="Algo", y="Group Regret", data=df_9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            plt.savefig('performance_group.png')
            plt.clf()
            fig = plt.figure(figsize=(8,6))
            ax = sns.barplot(x="Algo", y="Population Regret", data=df_9[df_9.Algo.isin(['Sparse+2','DP','ILP','Greedy'])])
            plt.savefig('performance_population_4.png')
            plt.clf()
            fig = plt.figure(figsize=(8,6))
            ax = sns.barplot(x="Algo", y="Group Regret", data=df_9[df_9.Algo.isin(['Sparse+2','DP','ILP','Greedy'])])
            plt.savefig('performance_group_4.png')
            plt.clf()
            plt.clf()
            fig = plt.figure(figsize=(8,6))
            ax = sns.barplot(x="Algo", y="Time (seconds)", data=df_9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            plt.savefig('time_taken.png')
            fig = plt.figure(figsize=(8,6))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            plt.clf()
            plt.clf()
            ax = sns.barplot(x="Algo", y="Time (seconds)", data=df_9[df_9.Algo.isin(['Sparse+2','DP','ILP','Greedy'])])
            plt.savefig('time_taken_4.png')
            plt.clf()
            df_9.groupby(['Algo'])['Time (seconds)'].mean().to_csv('exp 9 times.csv')
    except:
        print ('Experiment 9 failed')






    if run_exp_10:
        print('Running Experiment 10')
        ## Experiment 10 - Product Dependence
        seed = 10
        random.seed(seed)
        results_10 = {'R_pop':[],'R_group':[],'R_dif':[],'subinst':[],'Products Desired':[],'Products Used':[],'Algo':[],'Time (seconds)':[]}
        n_instances = 10
        T= 200
        stepsize_mult=10000
        n_cons= 50
        mean_lower = 0.02
        mean_dif = 0.01
        std_lower=.002
        std_dif= 0.001
        ps = 1 / 3 * np.array([1, 1, 1])
        ngroups = 3
        all_products = 20
        consumer_weights_dp = 1/n_cons*np.array([1 for i in range(n_cons)])
        def append_row_10(results,algo, prod_used,prod_des, subinst,time_taken):
            results_10['R_group'].append(results['R_max'])
            results_10['R_pop'].append(results['R_pop'])
            results_10['R_dif'].append(results['R_dif'])
            results_10['subinst'].append(subinst)
            results_10['Algo'].append(algo)
            results_10['Products Used'].append(prod_used)
            results_10['Products Desired'].append(prod_des)
            results_10['Time (seconds)'].append(time_taken)

        for instance in range(n_instances):
            groups,risks,returns = get_grr_vary_mean_multigroup(n_cons,mean_lower, mean_dif,std_lower,std_dif,ps,ngroups)
            groups,risks,returns = sort_data(groups,risks,returns)
            np.unique(groups,return_counts=True)
            ngroups=3
            val_counts = dict(zip(np.unique(groups, return_counts=True)[0], np.unique(groups, return_counts=True)[1]))
            groupsizes = [val_counts[g] if g in val_counts.keys() else 0 for g in range(ngroups)]
            for products in range(1, all_products+1):

                stepsize = calc_stepsize(stepsize_mult, returns, groups, ngroups, T)

                start = time.time()
                results_game, prods_game = do_game(returns,groups,ngroups,groupsizes,products,stepsize,T)
                end = time.time()
                time_game = end - start

                start = time.time()
                results_sparsify_0, prods_sparsify_0 = do_sparsify(returns,groups,ngroups,groupsizes,products,0,prods_game)
                end = time.time()
                time_sp_0= time_game + end - start

                start = time.time()
                results_sparsify_1, prods_sparsify_1 = do_sparsify(returns,groups,ngroups,groupsizes,products,1,prods_game)
                end = time.time()
                time_sp_1= time_game + end - start


                start = time.time()
                results_sparsify_2, prods_sparsify_2 = do_sparsify(returns,groups,ngroups,groupsizes,products,2,prods_game)
                end = time.time()
                time_sp_2= time_game + end - start


                start = time.time()
                results_DP, prods_DP = do_DP(returns,groups,groupsizes, ngroups,products,consumer_weights_dp)
                end = time.time()
                time_DP= end - start


                start = time.time()
                results_ILP, prods_ILP = do_ILP(returns,groups,ngroups,products,use_avg_regret=True)
                end = time.time()
                time_ILP = end - start


                start = time.time()
                results_greedy, prods_greedy = do_greedy(returns,groups,ngroups,groupsizes,products)
                end = time.time()
                time_greedy =  end - start



                append_row_10(results_game, 'No Regret', len(prods_game),products,  instance,time_game)
                append_row_10(results_sparsify_0, 'Sparse+0', len(prods_sparsify_0),products, instance,time_sp_0)
                append_row_10(results_sparsify_1, 'Sparse+1', len(prods_sparsify_1),products, instance,time_sp_1)
                append_row_10(results_sparsify_2, 'Sparse+1', len(prods_sparsify_2),products,instance,time_sp_2)
                append_row_10(results_DP, 'DP', len(prods_DP),products, instance,time_DP)
                append_row_10(results_ILP, 'ILP', len(prods_ILP),products, instance,time_ILP)
                append_row_10(results_greedy, 'Greedy', len(prods_greedy),products, instance,time_greedy)

        df_10 = pd.DataFrame(results_10)
        df_10.rename({'R_pop':'Population Regret', 'R_group':'Group Regret'}, axis=1,inplace=True)
        plt.clf()
        ax = sns.lineplot(x="Products Used", hue='Algo', y="Population Regret", data=df_10[df_10.Algo.isin(['DP','ILP','Greedy'])])
        df_10_sp1 = df_10[df_10.Algo=='Sparse+1'].groupby('Products Used')['Population Regret'].mean().reset_index()
        sns.scatterplot(x="Products Used", color='black', y="Population Regret", data=df_10_sp1)
        plt.title('Population Regret vs Products Used')
        plt.savefig('pop_regret_by_prod_population.png')
        plt.clf()

        ax = sns.lineplot(x="Products Used", hue='Algo', y="Group Regret", data=df_10[df_10.Algo.isin(['DP','ILP','Greedy'])])
        df_10_sp1 = df_10[df_10.Algo=='Sparse+1'].groupby('Products Used')['Group Regret'].mean().reset_index()
        sns.scatterplot(x="Products Used", color='black', y="Group Regret", data=df_10_sp1)
        plt.title('Population Regret vs Products Used')
        plt.savefig('group_regret_by_prod_population.png')
        plt.clf()

        plt.clf()
        ax = sns.lineplot(x='Products Used', hue='Algo', y='Time (seconds)', data = df_10[df_10.Algo.isin(['DP','ILP','Greedy'])])
        df_10_time  = df_10[df_10.Algo=='Sparse+1'].groupby('Products Used')['Time (seconds)'].mean().reset_index()
        sns.scatterplot(x="Products Used", color='black', y="Group Regret", data=df_10_sp1)
        plt.savefig('time by products.png')

    try:
        if run_exp_11:
            print('Running Experiment 11')
            ## Experiment 11 - Consumer Dependence
            seed = 10
            random.seed(seed)
            results_11 = {'R_pop':[],'R_group':[],'R_dif':[],'subinst':[],'Products Desired':[],'Products Used':[],'Algo':[],'Time (seconds)':[],'N_cons':[]}
            n_instances = 5
            T= 500
            stepsize_mult=1000
            n_cons= 50
            #mean_lower = 0.02
            #mean_dif = 0.01
            #std_lower=.002
            #std_dif= 0.001
            mean_lower = 0.001
            mean_dif = 0.005
            std_lower=.0001
            std_dif= 0.0005

            all_cons_choices = np.linspace(10,50,9 )
            ps = 1 / 3 * np.array([1, 1, 1])
            ngroups = 3
            products = 5
            def append_row_11(results,algo, prod_used,prod_des, subinst,time_taken,ncons):
                results_11['R_group'].append(results['R_max'])
                results_11['R_pop'].append(results['R_pop'])
                results_11['R_dif'].append(results['R_dif'])
                results_11['subinst'].append(subinst)
                results_11['Algo'].append(algo)
                results_11['Products Used'].append(prod_used)
                results_11['Products Desired'].append(prod_des)
                results_11['Time (seconds)'].append(time_taken)
                results_11['N_cons'].append(ncons)

            for n_cons in all_cons_choices:
                n_cons = int(n_cons)
                consumer_weights_dp = 1 / n_cons * np.array([1 for i in range(n_cons)])
                for instance in range(n_instances):
                    groups,risks,returns = get_grr_vary_mean_multigroup(n_cons,mean_lower, mean_dif,std_lower,std_dif,ps,ngroups)
                    groups,risks,returns = sort_data(groups,risks,returns)
                    np.unique(groups,return_counts=True)
                    ngroups=3
                    val_counts = dict(zip(np.unique(groups, return_counts=True)[0], np.unique(groups, return_counts=True)[1]))
                    groupsizes = [val_counts[g] if g in val_counts.keys() else 0 for g in range(ngroups)]

                    stepsize = calc_stepsize(stepsize_mult, returns, groups, ngroups, T)

                    start = time.time()
                    results_game, prods_game = do_game(returns,groups,ngroups,groupsizes,products,stepsize,T)
                    end = time.time()
                    time_game = end - start

                    start = time.time()
                    results_sparsify_0, prods_sparsify_0 = do_sparsify(returns,groups,ngroups,groupsizes,products,0,prods_game)
                    end = time.time()
                    time_sp_0= time_game + end - start

                    start = time.time()
                    results_sparsify_1, prods_sparsify_1 = do_sparsify(returns,groups,ngroups,groupsizes,products,1,prods_game)
                    end = time.time()
                    time_sp_1= time_game + end - start


                    start = time.time()
                    results_sparsify_2, prods_sparsify_2 = do_sparsify(returns,groups,ngroups,groupsizes,products,2,prods_game)
                    end = time.time()
                    time_sp_2= time_game + end - start


                    start = time.time()
                    results_DP, prods_DP = do_DP(returns,groups,groupsizes, ngroups,products,consumer_weights_dp)
                    end = time.time()
                    time_DP= end - start


                    start = time.time()
                    results_ILP, prods_ILP = do_ILP(returns,groups,ngroups,products,use_avg_regret=True)
                    end = time.time()
                    time_ILP = end - start


                    start = time.time()
                    results_greedy, prods_greedy = do_greedy(returns,groups,ngroups,groupsizes,products)
                    end = time.time()
                    time_greedy =  end - start


                    append_row_11(results_game, 'No Regret', len(prods_game),products,  instance,time_game,n_cons)
                    append_row_11(results_sparsify_0, 'Sparse+0', len(prods_sparsify_0),products, instance,time_sp_0,n_cons)
                    append_row_11(results_sparsify_1, 'Sparse+1', len(prods_sparsify_1),products, instance,time_sp_1,n_cons)
                    append_row_11(results_sparsify_2, 'Sparse+2', len(prods_sparsify_2),products,instance,time_sp_2,n_cons)
                    append_row_11(results_DP, 'DP', len(prods_DP),products, instance,time_DP,n_cons)
                    append_row_11(results_ILP, 'ILP', len(prods_ILP),products, instance,time_ILP,n_cons)
                    append_row_11(results_greedy, 'Greedy', len(prods_greedy),products, instance,time_greedy,n_cons)
                    append_row_11(results_greedy, 'Greedy', len(prods_greedy),products, instance,time_greedy,n_cons)
            plt.clf()
            ax = sns.lineplot(x='N_cons', hue='Algo', y='Time (seconds)', data = df_11[df_11.Algo.isin(['ILP','Greedy','No Regret'])])
            plt.title('Time vs Number of consumers')
            plt.savefig('time vs consumers.png')


            df_11 = pd.DataFrame(results_11)
            df_11.rename({'R_pop':'Population Regret', 'R_group':'Group Regret'}, axis=1,inplace=True)
            plt.clf()
            ax = sns.lineplot(x="N_cons", hue='Algo', y="Population Regret", data=df_11[df_11.Algo.isin(['DP','ILP','Greedy'])])
            df_11_sp1 = df_11[df_11.Algo=='Sparse+1'].groupby('Products Used')['Population Regret'].mean().reset_index()
            sns.scatterplot(x="Products Used", color='black', y="Population Regret", data=df_11_sp1)
            plt.title('Population Regret vs Products Used')
            plt.savefig('pop_regret_by_prod_population.png')
            plt.clf()

            ax = sns.lineplot(x="N_cons", hue='Algo', y="Group Regret", data=df_11[df_11.Algo.isin(['DP','ILP','Greedy'])])
            df_11_sp1 = df_11[df_11.Algo=='Sparse+1'].groupby('Products Used')['Group Regret'].mean().reset_index()
            sns.scatterplot(x="Products Used", color='black', y="Group Regret", data=df_11_sp1)
            plt.title('Population Regret vs Products Used')
            plt.savefig('group_regret_by_cons_population.png')
            plt.clf()

            plt.clf()
            ax = sns.lineplot(x='N_cons', hue='Algo', y='Time (seconds)', data = df_11[df_11.Algo.isin(['ILP','Greedy','No Regret'])])
            df_11_time  = df_11[df_11.Algo=='Sparse+1'].groupby('Products Used')['Time (seconds)'].mean().reset_index()
            sns.scatterplot(x="N_cons", color='black', y="Time (seconds)", data=df_11[df_11.Algo=='No Regret'],ax=ax)
            plt.title('Time vs Number of consumers')
            plt.savefig('time vs consumers.png')
    except:
        print('Error Experiment 11')


    try:
        if run_exp_12:
            print('Running Experiment 12')
            ## Experiment 12 - final generalization
            seed = 10
            random.seed(seed)
            results_12 = {'Prods': [], 'Prod_des': [], 'R_group_in': [], 'R_pop_in': [], 'R_group_out': [], 'R_pop_out': [],
                         'Algo': [], 'Instance': [], 'N_cons_in': []}
            #n_consumer_choices = np.linspace(20, 600, 30)
            n_consumer_choices = np.linspace(500, 800,16)

            max_instances = 100
            prod = 5
            test_size = 5000
            T = 500
            stepsize_mult = 1000
            out_groups, out_risks, out_returns = get_grr_vary_mean_multigroup(test_size, 0.02, 0.01, 0.002, 0.001,
                                                                              1 / 3 * np.array([1, 1, 1]), 3)
            out_groups, out_risks, out_returns = get_grr_vary_mean_multigroup(test_size, 0.001, 0.005, 0.0001, 0.0005,
                                                                              1 / 3 * np.array([1, 1, 1]), 3)
            out_groups, out_risks, out_returns = sort_data(out_groups, out_risks, out_returns)
            out_ngroups = len(np.unique(out_groups))
            out_val_counts = dict(
                zip(np.unique(out_groups, return_counts=True)[0], np.unique(out_groups, return_counts=True)[1]))
            out_groupsizes = [out_val_counts[g] if g in out_val_counts.keys() else 0 for g in range(out_ngroups)]
            print('generated test set')
            for j in range(max_instances):
                print('instance: '+str(j))
                start_time = time.time()
                for n_cons in n_consumer_choices:
                    print(n_cons)
                    start_cons = time.time()


                    #in_groups, in_risks, in_returns = get_grr_vary_mean_multigroup(int(n_cons), 0.02, 0.01, 0.002, 0.001,
                     #                                                              1 / 3 * np.array([1, 1, 1]), 3)
                    in_groups, in_risks, in_returns = get_grr_vary_mean_multigroup(int(n_cons), 0.001, 0.005, 0.0001,
                                                                                   0.0005,
                                                                                   1 / 3 * np.array([1, 1, 1]), 3)

                    in_groups, in_risks, in_returns = sort_data(in_groups, in_risks, in_returns)
                    np.unique(in_groups, return_counts=True)
                    in_ngroups = len(np.unique(in_groups))
                    in_val_counts = dict(
                        zip(np.unique(in_groups, return_counts=True)[0], np.unique(in_groups, return_counts=True)[1]))
                    in_groupsizes = [in_val_counts[g] if g in in_val_counts.keys() else 0 for g in range(in_ngroups)]

                    stepsize = calc_stepsize(stepsize_mult, in_returns, in_groups, in_ngroups, T)
                    results_game, prods_game = do_game(in_returns, in_groups, in_ngroups, in_groupsizes, prod, stepsize, T)
                    actual_prods = in_returns[prods_game]
                    if 0 not in actual_prods:
                        actual_prods = np.insert(actual_prods, 0, 0)
                    group_regrets_mm_out = get_group_regrets_nonconsumer_prods(out_returns, out_groups, out_ngroups,
                                                                               actual_prods, use_avg_regret=True)
                    regret_out_max = np.max(group_regrets_mm_out)
                    if len(out_groupsizes) > len(group_regrets_mm_out):
                        out_groupsizes = [size for size in out_groupsizes if size > 0]
                    group_regrets_mm_out = np.array(group_regrets_mm_out)
                    out_groupsizes = np.array(out_groupsizes)
                    regret_out_pop = getregret(out_groupsizes, group_regrets_mm_out)

                    results_12['Prods'].append(len(prods_game))
                    results_12['Prod_des'].append(5)
                    results_12['R_group_in'].append(results_game['R_max'])
                    results_12['R_pop_in'].append(results_game['R_pop'])
                    results_12['Algo'].append('NoRegret')
                    results_12['Instance'].append(j)
                    results_12['R_group_out'].append(regret_out_max)
                    results_12['R_pop_out'].append(regret_out_pop)
                    results_12['N_cons_in'].append(n_cons)

                    results_slack_4, slack_4_prod = do_sparsify(in_returns, in_groups, in_ngroups, in_groupsizes, prod, 4,
                                                                prods_game)

                    actual_prods = in_returns[slack_4_prod]

                    if 0 not in actual_prods:
                        actual_prods = np.insert(actual_prods, 0, 0)

                    group_regrets_sp4_out = get_group_regrets_nonconsumer_prods(out_returns, out_groups, out_ngroups,
                                                                                actual_prods, use_avg_regret=True)

                    regret_max_out_sp4 = np.max(group_regrets_sp4_out)
                    if len(out_groupsizes) > len(group_regrets_sp4_out):
                        out_groupsizes = [size for size in out_groupsizes if size > 0]
                    group_regrets_sp4_out = np.array(group_regrets_sp4_out)
                    out_groupsizes = np.array(out_groupsizes)
                    regret_pop_out_sp4 = getregret(out_groupsizes, group_regrets_sp4_out)

                    results_12['Prods'].append(len(slack_4_prod))
                    results_12['Prod_des'].append(5)
                    results_12['R_group_in'].append(results_slack_4['R_max'])
                    results_12['R_pop_in'].append(results_slack_4['R_pop'])
                    results_12['Algo'].append('SparseP+4')
                    results_12['Instance'].append(j)
                    results_12['R_group_out'].append(regret_max_out_sp4)
                    results_12['R_pop_out'].append(regret_pop_out_sp4)
                    results_12['N_cons_in'].append(n_cons)
                    end_cons = time.time()
                    print(end_cons-start_cons)
                if j%10==0:
                    df_so_far = pd.DataFrame(results_12)
                    dfsp_4 = df_so_far[df_so_far.Algo=='SparseP+4']
                    means_only = dfsp_4[[col for col in dfsp_4.columns if col != 'Instance']].groupby(['N_cons_in']).mean()
                    dfsp_4.to_csv('sparse_gen_data_aof_'+str(j) +'_instances.csv')
                    means_only.to_csv('sparse_gen_data_aof_'+str(j) +'_instances_means_only.csv')
                    plt.clf()
                    fig = plt.figure(figsize=(8,6))
                    melted_12 = pd.melt(dfsp_4, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
                    melted_12['variable'] = melted_12.variable.map({'R_group_in': 'In-sample Group Regret', 'R_group_out':'Test set Group Regret'})
                    melted_12.rename({'variable': 'Regret_type','value':'Regret','N_cons_in':'Consumer Sample Size'},axis=1, inplace=True)

                    sns.lineplot('Consumer Sample Size', 'Regret', style='Regret_type',
                                 data=pd.DataFrame(melted_12).reset_index(),
                                 ci=None)
                    plt.ylabel('Regret')
                    plt.xlabel('Number of Consumers in Sample')
                    plt.title('Generalization - No Regret Sparsified to P+4')
                    plt.savefig('generalization_sp4_group.png')

                    plt.clf()
                    fig = plt.figure(figsize=(8,6))
                    melted_12 = pd.melt(dfsp_4, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
                    melted_12['variable'] = melted_12.variable.map({'R_pop_in': 'In-sample Population Regret', 'R_pop_out':'Test set Population Regret'})
                    melted_12.rename({'variable': 'Regret_type','value':'Regret','N_cons_in':'Consumer Sample Size'},axis=1, inplace=True)
                    sns.lineplot('Consumer Sample Size', 'Regret', style='Regret_type',
                                 data=pd.DataFrame(melted_12).reset_index(),
                                 ci=None)
                    plt.ylabel('Regret')
                    plt.xlabel('Number of Consumers in Sample')
                    plt.title('Generalization - No Regret Sparsified to P+4')
                    plt.savefig('generalization_sp4_pop.png')

                    plt.close('all')
                end_time = time.time()
                print(end_time-start_time)
        df_12 = pd.DataFrame(results_12)
    except:
        print('Error Experiment 12')

