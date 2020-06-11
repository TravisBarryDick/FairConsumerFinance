from experiment_utils import *
def exp_7():
    print('Running Experiment 7')

    # Experiment 7: Generalization
    product_choices = [3, 5, 10]
    
    random.seed(seed)
    results_7 = {'Prods': [], 'Prod_des': [], 'R_group_in': [], 'R_pop_in': [], 'R_group_out': [], 'R_pop_out': [], 'Algo': [], 'Instance': [],'N_cons_in': []}
    n_consumer_choices = np.linspace(20,200,19)
    n_instances = 5
    T=500
    stepsize_mult=1000
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
            #stepsize_mult = 1000
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
    plt.savefig('results/generalization.png')

    plt.clf()
    melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
    sns.lineplot('N_cons_in', 'value', style='variable',data=pd.DataFrame(melted_7[(melted_7.Algo=='NoRegret') &( melted_7.Prod_des==10)]).reset_index(),ci=None)
    plt.ylabel('Regret')
    plt.xlabel('Number of Consumers in Sample')
    plt.title('Generalization - 10 products desired')
    plt.savefig('results/generalization 10.png')

    plt.clf()
    melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
    sns.lineplot('N_cons_in', 'value', style='variable',data=pd.DataFrame(melted_7[(melted_7.Algo=='NoRegret') &( melted_7.Prod_des==3)]).reset_index(),ci=None)
    plt.ylabel('Regret')
    plt.xlabel('Number of Consumers in Sample')
    plt.title('Generalization - 3 products desired')
    plt.savefig('results/generalization 3.png')

    plt.clf()
    melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
    sns.lineplot('N_cons_in', 'value', style='variable',data=pd.DataFrame(melted_7[(melted_7.Algo=='NoRegret') &( melted_7.Prod_des==5)]).reset_index(),ci=None)
    plt.ylabel('Regret')
    plt.xlabel('Number of Consumers in Sample')
    plt.title('Generalization - 5 products desired')
    plt.savefig('results/generalization 5.png')

    plt.clf()
    melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
    sns.lineplot('N_cons_in', 'value', style='variable',
                 data=pd.DataFrame(melted_7[(melted_7.Algo == 'SparseP+1') & (melted_7.Prod_des == 10)]).reset_index(),
                 ci=None)
    plt.ylabel('Regret')
    plt.xlabel('Number of Consumers in Sample')
    plt.title('Generalization - 10 products desired (Sparse P+1)')
    plt.savefig('results/generalization 10 sparse.png')

    plt.clf()
    melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
    sns.lineplot('N_cons_in', 'value', style='variable',
                 data=pd.DataFrame(melted_7[(melted_7.Algo == 'SparseP+1') & (melted_7.Prod_des == 3)]).reset_index(),
                 ci=None)
    plt.ylabel('Regret')
    plt.xlabel('Number of Consumers in Sample')
    plt.title('Generalization - 3 products desired (Sparse P+1)')
    plt.savefig('results/generalization 3 sparse .png')

    plt.clf()
    melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_group_in', 'R_group_out'])
    sns.lineplot('N_cons_in', 'value', style='variable',
                 data=pd.DataFrame(melted_7[(melted_7.Algo == 'SparseP+1') & (melted_7.Prod_des == 5)]).reset_index(),
                 ci=None)
    plt.ylabel('Regret')
    plt.xlabel('Number of Consumers in Sample')
    plt.title('Generalization - 5 products desired (Sparse P+1)')
    plt.savefig('results/generalization 5 sparse.png')

    plt.clf()
    melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
    sns.lineplot('N_cons_in', 'value', style='variable',
                 data=pd.DataFrame(melted_7[(melted_7.Algo == 'NoRegret') & (melted_7.Prod_des == 10)]).reset_index(),
                 ci=None)
    plt.ylabel('Regret')
    plt.xlabel('Number of Consumers in Sample')
    plt.title('Generalization - 10 products desired, Population Regret')
    plt.savefig('results/generalization 10 pop.png')

    plt.clf()
    melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
    sns.lineplot('N_cons_in', 'value', style='variable',
                 data=pd.DataFrame(melted_7[(melted_7.Algo == 'NoRegret') & (melted_7.Prod_des == 3)]).reset_index(),
                 ci=None)
    plt.ylabel('Regret')
    plt.xlabel('Number of Consumers in Sample')
    plt.title('Generalization - 3 products desired, Population Regret')
    plt.savefig('results/generalization 3 pop.png')

    plt.clf()
    melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
    sns.lineplot('N_cons_in', 'value', style='variable',
                 data=pd.DataFrame(melted_7[(melted_7.Algo == 'NoRegret') & (melted_7.Prod_des == 5)]).reset_index(),
                 ci=None)
    plt.ylabel('Regret')
    plt.xlabel('Number of Consumers in Sample')
    plt.title('Generalization - 5 products desired, Population Regret')
    plt.savefig('results/generalization 5 pop.png')

    plt.clf()
    melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
    sns.lineplot('N_cons_in', 'value', style='variable',
                 data=pd.DataFrame(melted_7[(melted_7.Algo == 'SparseP+1') & (melted_7.Prod_des == 10)]).reset_index(),
                 ci=None)
    plt.ylabel('Regret')
    plt.xlabel('Number of Consumers in Sample')
    plt.title('Generalization - 10 products desired (Sparse P+1, Population Regret)')
    plt.savefig('results/generalization 10 sparse pop.png')

    plt.clf()
    melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
    sns.lineplot('N_cons_in', 'value', style='variable',
                 data=pd.DataFrame(melted_7[(melted_7.Algo == 'SparseP+1') & (melted_7.Prod_des == 3)]).reset_index(),
                 ci=None)
    plt.ylabel('Regret')
    plt.xlabel('Number of Consumers in Sample')
    plt.title('Generalization - 3 products desired (Sparse P+1, Population Regret)')
    plt.savefig('results/generalization 3 sparse pop.png')

    plt.clf()
    melted_7 = pd.melt(df_7, id_vars=['Algo', 'Prod_des', 'N_cons_in'], value_vars=['R_pop_in', 'R_pop_out'])
    sns.lineplot('N_cons_in', 'value', style='variable',
                 data=pd.DataFrame(melted_7[(melted_7.Algo == 'SparseP+1') & (melted_7.Prod_des == 5)]).reset_index(),
                 ci=None)
    plt.ylabel('Regret')
    plt.xlabel('Number of Consumers in Sample')
    plt.title('Generalization - 5 products desired (Sparse P+1, Population Regret)')
    plt.savefig('results/generalization 5 sparse pop.png')
