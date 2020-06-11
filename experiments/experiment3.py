from experiment_utils import *
def exp_3():
    print('Running Experiment 3')

    # Experiment  3: varying mean difference of normals

    product_choices = [3,5,10]
    mean_difs = np.linspace(0.005,0.05,10)

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
                    returns, groups, ngroups, p, use_avg_regret=True, solver_str=solver )
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
    plt.savefig('results/Regret vs Dif in means.png')
    plt.close(fig)

    fig = plt.figure(figsize=(20,10))
    sns.lineplot('Mean_dif','R_group',hue='Algo',style='Prod_des', data=df_3[~(df_3.Algo.isin(['NoRegret','GreedyMM','SparseP+2']))])
    plt.title('Regret vs Difference in Means of distribution')
    plt.xlabel('Difference in Std. Deviation')
    plt.ylabel('Ex Post Max Group Regret')
    plt.savefig('results/Regret vs Dif in means no gmm.png')
    plt.close(fig)


    fig = plt.figure(figsize=(20,10))
    sns.lineplot('Mean_dif','R_group',hue='Algo',style='Prod_des', data=df_3[(df_3.Prod_des==3) & ~(df_3.Algo.isin(['NoRegret','GreedyMM','SparseP+2']))])
    plt.title('Regret vs Difference in Means of distribution')
    plt.xlabel('Difference in Std. Deviation')
    plt.ylabel('Ex Post Max Group Regret')
    plt.savefig('results/Regret vs Dif in means no gmm 3 prod.png')
    plt.close(fig)


    fig = plt.figure(figsize=(20,10))
    sns.lineplot('Mean_dif','R_group',hue='Algo',style='Prod_des', data=df_3[(df_3.Prod_des==10) & ~(df_3.Algo.isin(['NoRegret','GreedyMM','SparseP+2']))])
    plt.title('Regret vs Difference in Means of distribution')
    plt.xlabel('Difference in Std. Deviation')
    plt.ylabel('Ex Post Max Group Regret')
    plt.savefig('results/Regret vs Dif in means no gmm 10 prod.png')
    plt.close(fig)
