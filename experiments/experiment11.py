from experiment_utils import *
def exp_11():
    print('Running Experiment 11')
    ## Experiment 11 - Consumer Dependence

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
            results_ILP, prods_ILP = do_ILP(returns,groups,ngroups,products,groupsizes,use_avg_regret=True)
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
    df_11 = pd.DataFrame(results_11)
    ax = sns.lineplot(x='N_cons', hue='Algo', y='Time (seconds)', data = df_11[df_11.Algo.isin(['ILP','Greedy','No Regret'])])
    plt.title('Time vs Number of consumers')
    plt.savefig('results/time vs consumers.png')



    df_11.rename({'R_pop':'Population Regret', 'R_group':'Group Regret'}, axis=1,inplace=True)
    plt.clf()
    ax = sns.lineplot(x="N_cons", hue='Algo', y="Population Regret", data=df_11[df_11.Algo.isin(['DP','ILP','Greedy'])])
    df_11_sp1 = df_11[df_11.Algo=='Sparse+1'].groupby('Products Used')['Population Regret'].mean().reset_index()
    sns.scatterplot(x="Products Used", color='black', y="Population Regret", data=df_11_sp1)
    plt.title('Population Regret vs Products Used')
    plt.savefig('results/pop_regret_by_prod_population.png')
    plt.clf()

    ax = sns.lineplot(x="N_cons", hue='Algo', y="Group Regret", data=df_11[df_11.Algo.isin(['DP','ILP','Greedy'])])
    df_11_sp1 = df_11[df_11.Algo=='Sparse+1'].groupby('Products Used')['Group Regret'].mean().reset_index()
    sns.scatterplot(x="Products Used", color='black', y="Group Regret", data=df_11_sp1)
    plt.title('Population Regret vs Products Used')
    plt.savefig('results/group_regret_by_cons_population.png')
    plt.clf()

    plt.clf()
    ax = sns.lineplot(x='N_cons', hue='Algo', y='Time (seconds)', data = df_11[df_11.Algo.isin(['ILP','Greedy','No Regret'])])
    df_11_time  = df_11[df_11.Algo=='Sparse+1'].groupby('Products Used')['Time (seconds)'].mean().reset_index()
    sns.scatterplot(x="N_cons", color='black', y="Time (seconds)", data=df_11[df_11.Algo=='No Regret'],ax=ax)
    plt.title('Time vs Number of consumers')
    plt.savefig('results/time vs consumers.png')
