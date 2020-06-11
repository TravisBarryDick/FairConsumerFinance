from experiment_utils import *
def exp_2():
    print('Running Experiment 2')

    ### Experiment 2 - Performance by ILP, sparsify, greedy, greedy MM

    
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
        step_mult = 1000
        T = 2000
        n_prods = range(1,11)

        for p in n_prods:
            max_regret_ilp, group_regrets_ilp, products_ilp = minmax_regret_ilp.minmax_regret_ilp_wrapper(
                returns, groups, ngroups, p, use_avg_regret=True, solver_str=solver)
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
    plt.savefig('results/Ex Post Regret by Products scatter.png')
    plt.close(fig)

    fig = plt.figure(figsize=(20,10))
    sns.lineplot('Prods','R_group',hue='Algo',data=results_df[results_df.Algo != 'NoRegret'])
    sns.scatterplot('Prods','R_group',data=results_df[results_df.Algo=='NoRegret'],color='black',marker='P',s=100,label='NoRegret')
    plt.xlabel('Products Used')
    plt.ylabel('Max Group Regret Ex-Post')
    plt.title('No Regret Game Max Expected Group Regret')
    plt.savefig('results/Ex Post Regret by Products lineplot.png')
    plt.close(fig)

    fig = plt.figure(figsize=(20,10))
    sns.scatterplot('Prods','R_group',hue='Algo',data=results_df[~results_df.Algo.isin(['NoRegret','Greedy','GreedyMM'])])
    sns.scatterplot('Prods','R_group',data=results_df[results_df.Algo=='NoRegret'],color='black',marker='P',s=100,label='NoRegret')
    plt.xlabel('Products Used')
    plt.ylabel('Max Group Regret Ex-Post')
    plt.title('No Regret Game Max Expected Group Regret')
    plt.savefig('results/Ex Post Regret by Products cleaner scatter.png')
    plt.close(fig)

    fig = plt.figure(figsize=(20,10))
    sns.lineplot('Prods','R_group',hue='Algo',data=results_df[~results_df.Algo.isin(['NoRegret','Greedy','GreedyMM'])])
    sns.scatterplot('Prods','R_group',data=results_df[results_df.Algo=='NoRegret'],color='black',marker='P',s=100,label='NoRegret')
    plt.xlabel('Products Used')
    plt.ylabel('Max Group Regret Ex-Post')
    plt.title('No Regret Game Max Expected Group Regret')
    plt.savefig('results/Ex Post Regret by Products cleaner lineplot.png')
    plt.close(fig)

    fig = plt.figure(figsize=(20,10))
    sns.lineplot('Prods','R_group',hue='Algo',data=results_df[~results_df.Algo.isin(['NoRegret','Greedy','GreedyMM'])],ci=None)
    sns.scatterplot('Prods','R_group',data=results_df[results_df.Algo=='NoRegret'],color='black',marker='P',s=100,label='NoRegret')
    plt.xlabel('Products Used')
    plt.ylabel('Max Group Regret Ex-Post')
    plt.title('No Regret Game Max Expected Group Regret')
    plt.savefig('results/Ex Post Regret by Products cleaner lineplot no err.png')
    plt.close(fig)


    plt.clf()
    fig = plt.figure(figsize=(20,10))
    sns.lineplot('Prod_des','R_group',hue='Algo',data=results_df)
    plt.xlabel('Desired Products')
    plt.ylabel('Max Group Regret Ex-Post')
    plt.title('No Regret Game Max Expected Group Regret')
    plt.savefig('results/Ex Post Regret by Desired Products cleaner lineplot.png')
    plt.close(fig)
