from experiment_utils import *
def exp_10():
    print('Running Experiment 10')
    ## Experiment 10 - Product Dependence

    random.seed(seed)
    results_10 = {'R_pop':[],'R_group':[],'R_dif':[],'subinst':[],'Products Desired':[],'Products Used':[],'Algo':[],'Time (seconds)':[]}
    n_instances = 10
    T= 200
    stepsize_mult=1000
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
            results_ILP, prods_ILP = do_ILP(returns,groups,ngroups,products,groupsizes,use_avg_regret=True)
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
    plt.savefig('results/pop_regret_by_prod_population.png')
    plt.clf()

    ax = sns.lineplot(x="Products Used", hue='Algo', y="Group Regret", data=df_10[df_10.Algo.isin(['DP','ILP','Greedy'])])
    df_10_sp1 = df_10[df_10.Algo=='Sparse+1'].groupby('Products Used')['Group Regret'].mean().reset_index()
    sns.scatterplot(x="Products Used", color='black', y="Group Regret", data=df_10_sp1)
    plt.title('Population Regret vs Products Used')
    plt.savefig('results/group_regret_by_prod_population.png')
    plt.clf()

    plt.clf()
    ax = sns.lineplot(x='Products Used', hue='Algo', y='Time (seconds)', data = df_10[df_10.Algo.isin(['DP','ILP','Greedy'])])
    df_10_time  = df_10[df_10.Algo=='Sparse+1'].groupby('Products Used')['Time (seconds)'].mean().reset_index()
    sns.scatterplot(x="Products Used", color='black', y="Group Regret", data=df_10_sp1)
    plt.savefig('results/time by products.png')
