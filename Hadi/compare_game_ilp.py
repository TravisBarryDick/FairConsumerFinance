import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
os.chdir('/Users/Hadi/Dropbox/Research/Fair Consumer Finance/python_code/Hadi')
sys.path.append(os.path.abspath('../'))
import bespoke_solvers
import minmax_regret_game
import minmax_regret_ilp
import regret_minimization_dp
from utils import get_group_regrets,
sys.path.append(os.path.abspath('../mkearns'))
import sparsify as sp
def bespoke_return(risk):
    w, r = bespoke_solvers.max_return_constrain_variance(
        asset_returns, asset_covars, risk, 1.0, long_only=True)
    return r


def calc_empirical_distribution(products, n_prod_choices):
    counts = [sum([i in products[t] for t in range(len(products))]) for i in range(n_prod_choices)]
    counts = [count / sum(counts) for count in counts]
    prods = [i for i in range(n_prod_choices)]
    return (pd.DataFrame(counts, prods).rename({0: 'product'}, axis=1))

def get_risks_returns_groups(prior_mean = 0.0001, prior_std = 0.00001, ngroups = 5, n_consumers=100, group_std = 0.000001):
    means = np.random.normal(loc=prior_mean, scale=prior_std, size=ngroups)
    consumer_groups = np.random.randint(low=0, high=ngroups, size=n_consumers)
    consumer_risks = np.array([np.random.normal(loc=means[consumer_groups[i]], scale=group_std) for i in range(n_consumers)])
    consumer_returns = np.array([bespoke_return(r) for r in consumer_risks])
    return(consumer_groups, consumer_risks, consumer_returns)
def get_risk_returns_groups(prior_mean =0.01)
def calc_theoretical_max_regret(returns,groups):
    return(np.max(np.array([np.sum(returns[groups == i]) for i in range(ngroups)])))

def calc_theroetical_stepsize(max_group_regret, ngroups, T):
    stepsize = np.sqrt(8 * np.log(ngroups) / T) / max_group_regret
    return(stepsize)

def calc_stepsize(stepsize_mult,returns,groups,ngroups,T):
    mgr = calc_theoretical_max_regret(returns,groups)
    theory_ss = calc_theroetical_stepsize(mgr,ngroups,T)
    return(stepsize_mult*theory_ss)


def run_for_n_prods(n_prods,returns,groups,ngroups, threshold=0, T=100,stepsize_mult=10, use_avg_regret=True ):
    T = 100
    n_prod_choices = len(returns)
    stepsize = calc_stepsize(stepsize_mult, returns,groups,ngroups,T)
    products, group_weights, group_regrets = minmax_regret_game.minmax_regret_game(returns=returns, groups=groups, num_groups=ngroups, num_prods=n_prods, T=T, use_avg_regret=use_avg_regret, step_size= stepsize)
    counts = [sum([i in products[t] for t in range(len(products))]) for i in range(n_prod_choices)]
    counts = [count / sum(counts) for count in counts]
    return ([sum([counts[i] > threshold for i in range(len(counts))]), products, counts,group_weights, group_regrets])

def run_for_n_prods_ilp(n_prods, returns,groups, ngroups,use_avg_regret=True):
    max_regret_ilp, group_regrets_ilp, products_ilp = minmax_regret_ilp.minmax_regret_ilp_wrapper(
        returns, groups, ngroups, n_prods, use_avg_regret=use_avg_regret,solver_str='GUROBI')
    return(max_regret_ilp, group_regrets_ilp, products_ilp)


def sparsify_wrapper(returns,products, epsilon=0.001):
    if len(products)==1:
        return(products)
    else:
        return(sp.sparsify(returns,products,epsilon))
def make_performance_plot(performance_data, filename='Performance'):
    plt.clf()
    plt.plot(performance_data.ilp_nprod, performance_data.ILP_max_regret, label='ILP', color='red')
    plt.plot(performance_data.desired_prod, performance_data.MM_max_regret, label='MM (purported)', color='blue')
    plt.plot(performance_data.desired_prod, performance_data.regret_sparse, label='MM (sparsified)', color='green')

    plt.scatter(performance_data.actual_prods, performance_data.MM_max_regret, label='MM (actual)', color='blue')
    plt.scatter(performance_data.actual_prods, performance_data.regret_sparse, label='MM (sparsified)', color='green')
    plt.scatter(performance_data.n_sparse, performance_data.regret_sparse, label='MM (sparsified)', color='purple')

    plt.ylabel('Max Regret')
    plt.xlabel('Products')
    plt.legend()
    plt.title('Max Regret under ILP vs Union of Minmax Game Solution')
    plt.savefig(filename+'.png')
def make_actual_vs_desired_plot(performance_data,filename = 'actual_vs_sparse_vs_desired')
    plt.plot(performance_data.desired, performance_data.actual,label='Actual Products Used')
    plt.plot(performance_data.desired, performance_data.n_sparse,label='Sparsified Products Used')
    plt.xlabel('N Products Desired')
    plt.ylabel('N Products Used')
    plt.title('Unique Products used in MM game')
    plt.legend()
    plt.savefig(filename+'.png')
def get_asset_ret_cov(asset_scale=252):
    asset_covars = np.genfromtxt("../data/cov_nolabels.csv", delimiter=",")
    asset_returns = np.genfromtxt("../data/returns_nolabels.csv", delimiter=",")
    asset_covars = asset_covars*(asset_scale)
    asset_returns = asset_returns*asset_scale
    return(asset_covars,asset_returns)

if __name__=='__main__':
    asset_scale = 252
    seed = 42
    asset_covars,asset_returns = get_asset_ret_cov(asset_scale)
    np.random.seed(seed)
    prior_mean = 0.05
    prior_std = 0.01
    ngroups = 5
    n_consumers = 100
    group_std = 0.01
    stepsize_multiplier = 10
    groups,risks, returns = get_risks_returns_groups(prior_mean=prior_mean,prior_std=prior_std,ngroups=ngroups,n_consumers=n_consumers,group_std=group_std)
    data = pd.DataFrame({'group': groups, 'risk': risks, 'returns': returns})
    data = data.sort_values('returns')
    risks = np.array(data.risk)
    returns = np.array(data.returns)
    groups = np.array(data.group)


    data.groupby('group').risk.plot(kind='kde')
    plt.clf()
    data.groupby('group').returns.plot.kde()
    plt.clf()
    T = 200
    sup_size, products, counts, group_weights, group_regrets =  run_for_n_prods(5, returns, groups, ngroups, threshold=0, T=T, stepsize_mult=10, use_avg_regret=True)
    products[-10:]
    emp_dist = calc_empirical_distribution(products, len(np.unique(risks)))
    emp_dist.plot(kind='bar')

    print(pd.datetime.now())
    results_mm= [run_for_n_prods(i+1, returns,groups,ngroups, threshold=0, T=100,stepsize_mult=100, use_avg_regret=True) for i in range(20)]
    print(pd.datetime.now())
    pos_support = [result[0] for result in results_mm]
    pd.DataFrame({'N_Products':[i+1 for i in range(len(pos_support))], 'Pos_Support':pos_support}).plot(x='N_Products',y='Pos_Support')
    results_ilp = [run_for_n_prods_ilp(i + 1, returns, groups, ngroups, use_avg_regret=True) for i in range(20)]
    print(pd.datetime.now())

    pd.DataFrame({'N_Products': [i + 1 for i in range(len(pos_support))], 'Pos_Support': pos_support}).plot(
    x='N_Products', y='Pos_Support')

    max_regret_ilp = [results_ilp[i][0] for i in range(len(results_ilp))]
    performance_ilp =  pd.DataFrame({'ilp_nprod': [i for i in range(len(results_ilp))],'ILP_max_regret': max_regret_ilp })
    all_prods_mm = [np.unique([results_mm[i][1]] )for i in range(len(results_mm))]
    max_regret_mm = [np.max(get_group_regrets(returns, groups,ngroups,all_prods_mm[i])) for i in range(len(all_prods_mm))]
    actual_used = [results_mm[i][0] for i in range(len(results_mm))]

    performance_mm = pd.DataFrame({'desired_prod':[i+1 for i in range(len(max_regret_mm))],'actual_prods':actual_used, 'MM_max_regret':max_regret_mm})
    performance = pd.merge(performance_ilp, performance_mm, left_on='ilp_nprod', right_on='desired_prod')

    sparse_products = [sparsify_wrapper( returns, np.unique(results_mm[i][1]), 0.005) for i in range(len(results_mm))]
    n_sparse = [len(sparse_products[i]) for i in range(len(sparse_products))]
    des_prod = [i+1 for i in range(len(sparse_products))]
    actual_prods = [results_mm[i][0] for i in range(len(results_mm))]
    group_regret_sparse = [get_group_regrets(returns, groups, ngroups,sparse_products[i]) for i in range(len(sparse_products))]
    regret_sparse = [np.max(group_regret_sparse[i]) for i in range(len(group_regret_sparse))]
    performance_sparse = pd.DataFrame({'desired':des_prod, 'actual':actual_prods, 'n_sparse': n_sparse, 'regret_sparse': regret_sparse })
    performance = pd.merge(performance, performance_sparse, left_on='desired_prod', right_on='desired')
    assert(np.min([np.min([sparse_products[i][j] in results_mm[i][1] for j in range(len(sparse_products[i]))]) for i in range(len(sparse_products))]) ==True)
    regrets_nonsparse = [np.max(get_group_regrets(returns, groups,ngroups,all_prods_mm[i])) for i in range(len(all_prods_mm))]
    assert(np.max(np.array(max_regret_mm) > np.array(regret_sparse))==0)

    performance.plot(x='actual',y='n_sparse',kind='scatter')
    plt.title('Sparsified Products vs Actual Products')
    plt.savefig('sparse_vs_actual.png')

    performance.plot(x='desired',y='n_sparse',kind='line')
    plt.title('Sparsified Products vs Desired Products')
    plt.savefig('sparse_vs_desired.png')
    plt.clf()
    plt.plot(performance.desired, performance.actual,label='Actual Products Used')
    plt.plot(performance.desired, performance.n_sparse,label='Sparsified Products Used')
    plt.xlabel('N Products Desired')
    plt.ylabel('N Products Used')
    plt.title('Unique Products used in MM game')
    plt.legend()
    plt.savefig('actual_vs_sparse_vs_desired.png')
    performance.plot(x='desired_prod',y='n_sparse')
    plt.savefig('actual_vs_desired.png')




    plt.clf()
    plt.plot(performance.ilp_nprod, performance.ILP_max_regret,label='ILP',color='red')
    plt.plot(performance.desired_prod,performance.MM_max_regret,label='MM (purported)', color='blue')
    plt.plot(performance.desired_prod,performance.regret_sparse,label='MM (sparsified)', color='green')

    plt.scatter(performance.actual_prods, performance.MM_max_regret,label='MM (actual)',color='blue')
    plt.scatter(performance.actual_prods, performance.regret_sparse,label='MM (sparsified)',color='green')
    plt.scatter(performance.n_sparse, performance.regret_sparse,label='MM (sparsified)',color='purple')

    plt.ylabel('Max Regret')
    plt.xlabel('Products')
    plt.legend()
    plt.title('Max Regret under ILP vs Union of Minmax Game Solution')
    plt.savefig('Performance.png')


    make_performance_plot(performance, 'Performance')
