import numpy as np
import matplotlib.pyplot as plt

def optimised_benchmark_toytracks(hit_coords, optimised_labels, algorithm_type : np.ndarray[str]):
    fig, ax = plt.subplots(1,len(algorithm_type), figsize=(13,8))
    
    for i, algorithm in enumerate(algorithm_type):
        ax[i].scatter(hit_coords[:, 0], hit_coords[:, 1], c=optimised_labels[i], cmap='bwr')
        ax[i].set_title(f'{algorithm}')
        
    fig.suptitle(f'Optimised Clusterings for Different Benchmark Algorithms: N = {len(hit_coords)}')
    fig.supylabel('y')
    fig.supxlabel('x')
    #plt.savefig(f'plots/ClassicalPlots/optimised_benchmark_clusters_{len(hit_coords)}_hits.png')
    plt.show()
    
    

def conv_traces(N: int, steps : np.ndarray, energy_histories : np.ndarray):
    reps = len(energy_histories)
    fig, ax = plt.subplots(reps, 1, figsize=(12,12))
    if reps > 1:
        for i in range(reps):
            ax[i].plot(np.arange(steps[i]), energy_histories[i], color='red')
            
    else:
        ax.plot(np.arange(steps[0]), energy_histories[0], color='red')
            
    fig.suptitle('Convergence Traces with Different Random Starting Points in SA')
    fig.supylabel('Energy')
    fig.supxlabel('Step')
    #plt.savefig(f'plots/ClassicalPlots/{reps}_Convergence_Traces_{N}_hits.png')
    plt.show()