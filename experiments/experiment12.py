from experiment_utils import *
def exp_12():
    print('Running Experiment 12')
    ## Experiment 12 - final generalization

    random.seed(seed)
    results_12 = {'Prods': [], 'Prod_des': [], 'R_group_in': [], 'R_pop_in': [], 'R_group_out': [], 'R_pop_out': [],
                 'Algo': [], 'Instance': [], 'N_cons_in': []}
    n_consumer_choices = np.linspace(25, 600, 24)
    #n_consumer_choices = np.linspace(500, 800,16)

    max_instances = 100
    prod = 5
    test_size = 5000
    T = 500
    stepsize_mult = 1000
    out_groups, out_risks, out_returns = get_grr_vary_mean_multigroup(test_size, 0.02, 0.01, 0.002, 0.001,
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


            in_groups, in_risks, in_returns = get_grr_vary_mean_multigroup(int(n_cons), 0.02, 0.01, 0.002, 0.001,
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

    df_so_far = pd.DataFrame(results_12)
    dfsp_4 = df_so_far[df_so_far.Algo=='SparseP+4']
    means_only = dfsp_4[[col for col in dfsp_4.columns if col != 'Instance']].groupby(['N_cons_in']).mean()
    dfsp_4.to_csv('results/sparse_gen_data_aof_'+str(j) +'_instances.csv')
    means_only.to_csv('results/sparse_gen_data_aof_'+str(j) +'_instances_means_only.csv')
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
    plt.savefig('results/generalization_sp4_group_{}.png'.format(j))

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
    plt.savefig('results/generalization_sp4_pop.png')

    plt.close('all')
    end_time = time.time()
    print(end_time-start_time)
