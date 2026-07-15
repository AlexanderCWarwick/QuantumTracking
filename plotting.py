import numpy as np
import matplotlib.pyplot as plt


def energy_landscape(lambda_bal, KNN_energies, RBF_energies): 
    '''
    ENergy landscape plots for both the KNN and the RBF. 
    Each similarity matrix gets two plots:
    1. 2^N states and their energies (all ordered by energy).
    2. 10 lowest energy states to view groundstate degeneracy.
    '''
    
    fig, ax = plt.subplots(2, 2, figsize=(10,6))  
    ax[0,0].plot(np.sort(KNN_energies), color='orange')
    ax[0,0].set_title('KNN all states')
    
    ax[0,1].plot(np.sort(RBF_energies), color='red')
    ax[0,1].set_title('RBF all states')
    
    ax[1,0].plot(np.sort(KNN_energies)[:10], color='orange')
    ax[1,0].set_title('KNN lowest 10 energy states')
    ax[1,0].set_xlabel('Rank')
    
    ax[1,1].plot(np.sort(RBF_energies)[:10], color='red')
    ax[1,1].set_title('RBF lowest 10 energy states')
    ax[1,1].set_xlabel('Rank')
    
    fig.suptitle(f'Bruteforce Ising Energy landscapes for lambda={lambda_bal}')
    fig.supylabel('Energy')
    plt.tight_layout()
    plt.show()
    

##############################################################################################################################

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

    
    
def print_benchmark_table(hits, classical_results : dict[dict]):
    
    header = (f'{'Hits':<8}'
        f'{'Algorithm':<25}'
        f'{'ARI':<12}'
        f'{'Full Time (s)':<20}'
        f'{'Relative Energy Error':<25}'
        f'{'Convergence Fraction':<22}')
    
    print(header)
    print('-' * len(header))
    
    
    for alg, metrics in classical_results.items():
        success = classical_results[alg]['conv_frac']
        
        if success is None:
            success = "-"
        else:
            success = f"{success:.2f}"
            
        print(f"{2*hits:<8}"
            f"{alg:<25}"
            f"{metrics['ari']:<12.4f}"
            f"{metrics['runtime']:<20.4f}"
            f"{metrics['rel_error']:<25.4f}"
            f"{success:<22}")
    
    print('\n')
        
        
def print_quantum_table(hits : int, quantum_results : dict):
    def format_metric(metric):
        return f'{metric['mean']:.4f} \u00b1 {metric['error']:.4f}'
    
    header = (f'{'Hits':<8}'
        f'{'QAOA Optimiser':<20}'
        f'{'ARI':<20}'
        f'{'Time (s)':<20}'
        f'{'Relative Energy Error':<30}'
        f'{'GS Prob':<20}')
    
    print(header)
    print('-' * len(header))
    
    for optimiser_name, metrics in quantum_results.items():
        print(f'{2*hits:<8}'
            f'{optimiser_name:<20}'
            f'{format_metric(metrics['ari']):<20}'
            f'{format_metric(metrics['runtime']):<20}'
            f'{format_metric(metrics['rel_error']):<30}'
            f'{format_metric(metrics['gsp']):<20}')

    
def scaling_scan(track_hits : np.ndarray[int],  raw_results : dict,  raw_errors : dict,
                                                rel_results : dict, rel_errors : dict):
    
    fig, ax = plt.subplots(2, figsize=(7,7))
    
    for name in raw_results.keys():
        ax[0].errorbar(track_hits, raw_results[name], yerr=raw_errors[name], fmt='-o', capsize=3, label=name)
        ax[1].errorbar(track_hits, rel_results[name], yerr=rel_errors[name], fmt='-o', capsize=3, label=name)
        
    ax[0].set_xticks(track_hits)
    ax[1].set_xticks(track_hits)
    
    ax[0].set_title('Raw GSP')
    ax[1].set_title('Relative GSP')
    
    fig.supxlabel('Number of hits, N')
    fig.supylabel('Groundstate Probability')
    fig.suptitle('GS Probability Dependency on N')
    ax[0].legend()
    ax[1].legend()
    plt.show()
    
    
###############################################################################################################################

def plot_energy_hist(energies, true_groundstate_energy):
    plt.figure()
    plt.hist(energies, bins=30)
    plt.axvline(true_groundstate_energy, color='red', linestyle='--', label='Exact ground state')
    plt.xlabel('Ising energy')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    
    
    
def depth_scan_metric_scatter(layers, metric_means : np.ndarray[float, float], metric_errors : np.ndarray[float, float]):
    fig, ax = plt.subplots(2,2,figsize=(8,5))
    ax = ax.flatten()
    metrics = ['rel_error', 'ari', 'runtime', 'gsp']
    
    for idx, metric in enumerate(metrics):
        for optimiser_name in metric_means[layers[0]].keys():
            means = [metric_means[p][optimiser_name][metric] for p in layers]
        
            errors = [metric_errors[p][optimiser_name][metric] for p in layers]
            
            ax[idx].errorbar(layers, means, yerr=errors, fmt='o-', capsize=3, alpha=0.7, label=optimiser_name)
            
        ax[idx].set_title(f'{metric}')
        ax[idx].set_xticks(layers)
        ax[idx].grid(True)
    
    fig.supxlabel("Number of QAOA layers")
    fig.supylabel("Metric value")
    fig.suptitle("QAOA Depth Scan")

    plt.tight_layout()
    plt.show()
    