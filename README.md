# Code and experiments for *Algorithms and Learning for Fair Portfolio Design*

Suggested citation: \<redacted\>


Usage: `python experiments_paper.py e1 e2 e3...` where the  `e`'s are the experiments to run.  

E.g. `python experiments_paper.py 1 4 11 2` will run Experiments 1, 2, 4, and 11.


#### List of experiments and corresponding figures in the paper (if included)

##### Experiment 1: No regret convergence time as a function of the step size. 
##### Experiment 2: Comparison of ILP, sparsify, greedy, greedy MM
##### Experiment 3: Groups are drawn from gaussians with the same mean, different variance
##### Experiment 4: Groups are drawn from gaussians with different means, same variance
##### Experiment 5: Groups are drawn from gaussians, the second's mean and variance are scaled multiples of the first
##### Experiment 6: One group is relatively small compared to the other
##### Experiment 7: Generalization
##### Experiment 8: Examining sparsification 
##### Experiment 9: Comparison of algorithms (Figure 2)
##### Experiment 10: Product dependence (+timing) 
##### Experiment 11: Consumer dependence (+timing) 
##### Experiment 12: Generalization (Figure 3)

Edit `configs.py` to change important global variables, including the solver used by `CVXPY`.  Currently supported: Gurobi and Mosek; other values will result in the default solvers being called.

Required packages:
- `bs4, cvxpy, numpy, numba, pandas, seaborn`
