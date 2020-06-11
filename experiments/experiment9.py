from experiment_utils import *
def exp_9():
    print('Running Experiment 9')

    ## Experiment 9 - performance

    random.seed(seed)
    results_9 = {'R_pop':[],'R_group':[],'R_dif':[],'subinst':[],'Products Used':[],'Algo':[],'Time (seconds)':[]}
    n_instances = 100
    T= 200
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
        results_ILP, prods_ILP = do_ILP(returns,groups,ngroups,products, groupsizes,use_avg_regret=True)
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
    df_9[df_9.Algo == 'No Regret']['Products Used'].describe()
    df_9.rename({'R_pop':'Population Regret', 'R_group':'Group Regret'}, axis=1,inplace=True)
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    ax = sns.barplot(x="Algo", y="Population Regret", data=df_9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.savefig('results/performance_population.png')
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    ax = sns.barplot(x="Algo", y="Group Regret", data=df_9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.savefig('results/performance_group.png')
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    ax = sns.barplot(x="Algo", y="Population Regret", data=df_9[df_9.Algo.isin(['Sparse+2','DP','ILP','Greedy'])])
    plt.savefig('results/performance_population_4.png')
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    ax = sns.barplot(x="Algo", y="Group Regret", data=df_9[df_9.Algo.isin(['Sparse+2','DP','ILP','Greedy'])])
    plt.savefig('results/performance_group_4.png')
    plt.clf()
    plt.clf()
    fig = plt.figure(figsize=(8,6))
    ax = sns.barplot(x="Algo", y="Time (seconds)", data=df_9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.savefig('results/time_taken.png')
    fig = plt.figure(figsize=(8,6))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.clf()
    plt.clf()
    ax = sns.barplot(x="Algo", y="Time (seconds)", data=df_9[df_9.Algo.isin(['Sparse+2','DP','ILP','Greedy'])])
    plt.savefig('results/time_taken_4.png')
    plt.clf()
    df_9.groupby(['Algo'])['Time (seconds)'].mean().to_csv('results/exp 9 times.csv')
