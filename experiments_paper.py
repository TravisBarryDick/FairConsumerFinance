"""Experiments for Paper.

# Experiment 1: No regret Convergence time vs Stepsize. 
# Experiment 2: Performance by ILP, sparsify, greedy, greedy MM 
# Experiment 3: varying mean difference of normals
# Experiment 4: varying std difference of normals
# Experiment 5: varying mean and std of normal via multiplier
# Experiment 6: Rare group
# Experiment 7: Generalization
# Experiment 8: performance of sparsify 
# Experiment 9: Performance of Algos (Figure 2)
# Experiment 10: Product Dependence (+timing) 
# Experiment 11: Consumer Dependence (+timing)
# Experiment 12: Generalization (Figure 3)
"""
import sys
sys.path.append("experiments")
sys.path.append("src")


from experiment1 import exp_1
from experiment2 import exp_2
from experiment3 import exp_3
from experiment4 import exp_4
from experiment5 import exp_5
from experiment6 import exp_6
from experiment7 import exp_7
from experiment8 import exp_8
from experiment9 import exp_9
from experiment10 import exp_10
from experiment11 import exp_11
from experiment12 import exp_12



def run_experiments(experiments):
    print(experiments)

    if '1' in experiments:
        exp_1()

    if '2' in experiments:
        exp_2()

    if '3' in experiments:
        exp_3()

    if '4' in experiments:
        exp_4()

    if '5' in experiments:
        exp_5()

    if '6' in experiments:
        exp_6()

    if '7' in experiments:
        exp_7()

    if '8' in experiments:
        exp_8()

    if '9' in experiments:
        exp_9()

    if '10' in experiments:
        exp_10()

    if '11' in experiments:
        exp_11()

    if '12' in experiments:
        exp_12()


if __name__ == '__main__':
    print(sys.argv[1])
    run_experiments(sys.argv[1:])
