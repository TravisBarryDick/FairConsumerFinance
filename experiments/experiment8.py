from experiment_utils import *
def exp_8():
    print('Running Experiment 8')

    ###experiment 8: performance of sparsify
    
    random.seed(seed)
    results_8 = {'R_pop':[],'R_group':[],'R_dif':[],'subinst':[],'Additional Products':[]}
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
    init_products = 5
    max_ad_products = 10
    for inst in range(n_instances):
        groups,risks,returns = get_grr_vary_mean_multigroup(n_cons,mean_lower, mean_dif,std_lower,std_dif,ps,ngroups)
        groups,risks,returns = sort_data(groups,risks,returns)
        np.unique(groups,return_counts=True)
        ngroups=len(np.unique(groups))
        val_counts = dict(zip(np.unique(groups, return_counts=True)[0], np.unique(groups, return_counts=True)[1]))
        groupsizes = [val_counts[g] if g in val_counts.keys() else 0 for g in range(ngroups)]
        stepsize = calc_stepsize(stepsize_mult, returns, groups, ngroups, T)
        results_game, prods_game = do_game(returns,groups,ngroups,groupsizes,init_products,stepsize,T)
        results_8['R_pop'].append(results_game['R_pop'])
        results_8['R_group'].append(results_game['R_max'])
        results_8['R_dif'].append(results_game['R_dif'])
        results_8['subinst'].append(inst)
        results_8['Additional Products'].append(-1)

        for k in range(0,max_ad_products+1):
            #as an aside, check to make sure do_sparsify is workign correctly. returns same results if slack==len(prods_game)-init_prods.
            results_sparsify, prods_sparsify = do_sparsify(returns,groups,ngroups,groupsizes,init_products,k,prods_game)
            results_8['R_pop'].append(results_sparsify['R_pop'])
            results_8['R_group'].append(results_sparsify['R_max'])
            results_8['R_dif'].append(results_sparsify['R_dif'])
            results_8['subinst'].append(inst)
            results_8['Additional Products'].append(k)

    df_8 = pd.DataFrame(results_8)
    df_8.rename({'R_group': 'Max Group Avg Regret','R_pop': 'Population Avg Regret'},axis=1,inplace=True)
    unsparse_group = [df_8[df_8['Additional Products'] == -1]['Max Group Avg Regret'].mean() for i in range(11)]
    unsparse_pop = [df_8[df_8['Additional Products'] == -1]['Population Avg Regret'].mean() for i in range(11)]
    plt.clf()
    sns.lineplot('Additional Products', 'Max Group Avg Regret',data=df_8[df_8['Additional Products']>-1])
    plt.plot([i for i in range(11)],unsparse_group ,'b--', label='Max Group Regret Unsparsified')
    plt.legend()
    plt.title('Group Regret as Sparsifier allowed more products')
    plt.savefig('results/Sparsification Group Regret.png')
    plt.clf()
    melted_8 = pd.melt(df_8, id_vars=['Additional Products'], value_vars=['Max Group Avg Regret', 'Population Avg Regret'])
    sns.lineplot('Additional Products','value',hue='variable',data=melted_8[melted_8['Additional Products']>-1])

    plt.plot([i for i in range(11)],unsparse_group ,'b--', label='Max Group Regret Unsparsified')
    plt.plot([i for i in range(11)],unsparse_pop ,'r--', label='Pop Regret Unsparsified')
    plt.legend()
    plt.title('Regret as Sparsifier allowed more products')
    plt.savefig('results/Sparisfication group pop regret.png')
    plt.clf()
