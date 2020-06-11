from experiment_utils import *
def exp_1():
    print('Running Experiment 1')

    ###### first experiment: performance. Convergence time and stepsize dependendence.

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
      exp_regrets = [calc_max_group_E_regret(groups,ngroups, all_products_mm[0:t+1],kset_regret, returns)for t in range(T)]
      for t in range(T):
          results['t'].append(t+1)
          results['Exp_R'].append(exp_regrets[t])
          results['Stepsize Multiplier'].append('M'+str(step_mult))

    results_df = pd.DataFrame(results)

    fig = plt.figure(figsize=(12,9))
    sns.lineplot('t', 'Exp_R', hue='Stepsize Multiplier', data=results_df)
    plt.ylabel('Max Expected Group Regret')
    plt.title('No Regret Game Max Expected Group Regret')
    plt.savefig('results/Convergence_stepsize_2.png')
    plt.close(fig)
