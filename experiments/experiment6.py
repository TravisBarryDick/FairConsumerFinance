from experiment_utils import *
def exp_6():
    print('Running Experiment 6')

    # Experiment 6: Rare group

    product_choices = [3, 5, 10]
    lambs = np.linspace(0.1, 1,2)
    
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
                max_regret_ilp, group_regrets_ilp, products_ilp = minmax_regret_ilp.minmax_regret_ilp_wrapper(returns, groups, ngroups, p, use_avg_regret=True, solver_str = solver)
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
    plt.savefig('results/Regret rare.png')

    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    sns.lineplot('Lambda', 'R_pop', hue='Algo', style='Prod_des',data=df_6[~(df_6.Algo.isin(['NoRegret','GreedyMM' ,'SparseP+2']))])
    plt.xlabel('Lambda for Rare Group')
    plt.ylabel('Ex Post Max Group Regret')
    plt.title('Regret vs Rare Group mean')
    plt.savefig('results/Regret rare no gmm.png')
    plt.close(fig)

    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    sns.lineplot('Lambda', 'R_pop', hue='Algo',data=df_6[(df_6.Prod_des==10) & ~(df_6.Algo.isin(['NoRegret','GreedyMM' ,'SparseP+2']))])
    plt.xlabel('Lambda for Rare Group')
    plt.ylabel('Ex Post Max Group Regret')
    plt.title('Regret vs Rare Group mean')
    plt.savefig('results/Regret rare no gmm 10prod.png')
    plt.close(fig)

    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    sns.lineplot('Lambda', 'R_pop', hue='Algo',data=df_6[(df_6.Prod_des==3) & ~(df_6.Algo.isin(['NoRegret','GreedyMM' ,'SparseP+2']))])
    plt.xlabel('Lambda for Rare Group')
    plt.ylabel('Ex Post Max Group Regret')
    plt.title('Regret vs Rare Group mean')
    plt.savefig('results/Regret rare no gmm 3 prod.png')
    plt.close(fig)

    fig = plt.figure(figsize=(20, 10))

    sns.lineplot('Beta', 'R_pop', hue='Algo', style='Prod_des',
                 data=df_6[~(df_6.Algo.isin(['NoRegret', 'SparseP+2']))])
    plt.xlabel('Beta for Rare Group')
    plt.ylabel('Ex Post Max Group Regret')
    plt.title('Regret vs Rare Group mean')
    plt.savefig('results/Regret rare beta.png')

    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    sns.lineplot('Beta', 'R_pop', hue='Algo', style='Prod_des',
                 data=df_6[~(df_6.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
    plt.xlabel('Beta for Rare Group')
    plt.ylabel('Ex Post Max Group Regret')
    plt.title('Regret vs Rare Group mean')
    plt.savefig('results/Regret rare no gmm beta.png')
    plt.close(fig)

    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    sns.lineplot('Beta', 'R_pop', hue='Algo',
                 data=df_6[(df_6.Prod_des == 10) & ~(df_6.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
    plt.xlabel('Beta for Rare Group')
    plt.ylabel('Ex Post Max Group Regret')
    plt.title('Regret vs Rare Group mean')
    plt.savefig('results/Regret rare no gmm 10prod beta.png')
    plt.close(fig)

    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    sns.lineplot('Beta', 'R_pop', hue='Algo',
                 data=df_6[(df_6.Prod_des == 3) & ~(df_6.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
    plt.xlabel('Beta for Rare Group')
    plt.ylabel('Ex Post Max Group Regret')
    plt.title('Regret vs Rare Group mean')
    plt.savefig('results/Regret rare no gmm 3 prod beta.png')
    plt.close(fig)


    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    sns.lineplot('Lambda', 'R_pop', hue='Algo',data=df_6[(df_6.Prod_des==3) & ~(df_6.Algo.isin(['NoRegret','GreedyMM' ,'SparseP+2']))])
    plt.xlabel('Lambda for Rare Group')
    plt.ylabel('Ex Post Max Group Regret')
    plt.title('Regret vs Rare Group mean')
    plt.savefig('results/Regret rare no gmm 3 prod.png')
    plt.close(fig)

    fig = plt.figure(figsize=(20, 10))

    sns.lineplot('Beta', 'R_pop', hue='Algo', style='Prod_des',
                 data=df_6[~(df_6.Algo.isin(['NoRegret', 'SparseP+2']))])
    plt.xlabel('Beta for Rare Group')
    plt.ylabel('Ex Post Max Group Regret')
    plt.title('Regret vs Rare Group mean')
    plt.savefig('results/Regret rare beta.png')

    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    sns.lineplot('Beta', 'R_pop', hue='Algo', style='Prod_des',
                 data=df_6[~(df_6.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
    plt.xlabel('Beta for Rare Group')
    plt.ylabel('Ex Post Max Group Regret')
    plt.title('Regret vs Rare Group mean')
    plt.savefig('results/Regret rare no gmm beta.png')
    plt.close(fig)

    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    sns.lineplot('Beta', 'R_pop', hue='Algo',
                 data=df_6[(df_6.Prod_des == 10) & ~(df_6.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
    plt.xlabel('Beta for Rare Group')
    plt.ylabel('Ex Post Max Group Regret')
    plt.title('Regret vs Rare Group mean')
    plt.savefig('results/Regret rare no gmm 10prod beta.png')
    plt.close(fig)

    plt.close(fig)
    fig = plt.figure(figsize=(20, 10))
    sns.lineplot('Beta', 'R_pop', hue='Algo',
                 data=df_6[(df_6.Prod_des == 3) & ~(df_6.Algo.isin(['NoRegret', 'GreedyMM', 'SparseP+2']))])
    plt.xlabel('Beta for Rare Group')
    plt.ylabel('Ex Post Max Group Regret')
    plt.title('Regret vs Rare Group mean')
    plt.savefig('results/Regret rare no gmm 3 prod beta.png')
    plt.close(fig)
