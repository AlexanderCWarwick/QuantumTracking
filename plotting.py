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

##############################################################################################################################
def format_metric(metric):
    return f'{metric['mean']:.4f} \u00b1 {metric['error']:.4f}'
    
def print_benchmark_table(hits, classical_results : dict[dict]):
    header = (f'{'Hits':<8}'
        f'{'Classical Algorithm':<30}'
        f'{'Relative Energy Error':<30}'
        f'{'ARI':<20}'
        f'{'Time (s)':<20}'
        f'{'Convergence Fraction':<30}')
    
    print(header)
    print('-' * len(header))
    
    for optimiser_name, metrics in classical_results.items():
        if optimiser_name == 'Simulated Annealing':
            print(f'{2*hits:<8}'
                f'{optimiser_name:<30}'
                f'{format_metric(metrics['rel_error']):<30}'
                f'{format_metric(metrics['ari']):<20}'
                f'{format_metric(metrics['runtime']):<20}'
                f'{format_metric(metrics['conv_frac']):<30}')
        else:
            print(f'{2*hits:<8}'
                f'{optimiser_name:<30}'
                f'{format_metric(metrics['rel_error']):<30}'
                f'{format_metric(metrics['ari']):<20}'
                f'{format_metric(metrics['runtime']):<20}'
                '-')
            
        
        
def print_quantum_table(hits : int, quantum_results : dict):
    header = (f'{'Hits':<8}'
        f'{'QAOA Optimiser':<20}'
        f'{'Relative Energy Error':<30}'
        f'{'ARI':<20}'
        f'{'Time (s)':<20}'
        f'{'GS Prob':<20}')
    
    print(header)
    print('-' * len(header))
    
    for optimiser_name, metrics in quantum_results.items():
        print(f'{2*hits:<8}'
            f'{optimiser_name:<20}'
            f'{format_metric(metrics['rel_error']):<30}'
            f'{format_metric(metrics['ari']):<20}'
            f'{format_metric(metrics['runtime']):<20}'
            f'{format_metric(metrics['gsp']):<20}')

    
def scaling_scan_metric_scatter(track_hits : np.ndarray[int],  raw_results : dict,  raw_errors : dict,
                                                rel_results : dict, rel_errors : dict):
    fig, ax = plt.subplots(2, figsize=(7,7))
    
    for name in raw_results.keys():
        ax[0].errorbar(track_hits, raw_results[name], yerr=raw_errors[name], fmt='-o', capsize=3, label=name)
        ax[1].errorbar(track_hits, rel_results[name], yerr=rel_errors[name], fmt='-o', capsize=3, label=name)
        
    
    baseline = 2 / (2 ** (2 * track_hits))
    ax[0].scatter(track_hits, baseline, label='Random Probability', color='r')
    ax[1].axhline(y=1, label='Random Probability', color='r')
    
    ax[0].set_xticks(track_hits)
    ax[1].set_xticks(track_hits)
    
    ax[0].set_title('Raw GSP')
    ax[1].set_title('Relative GSP')
    
    fig.supxlabel('Number of hits, N')
    fig.supylabel('Groundstate Probability')
    fig.suptitle('GS Probability Dependency on N')
    ax[0].legend()
    ax[1].legend()
    ax[0].yaxis.grid(True)
    ax[1].yaxis.grid(True)
    plt.show()
    
    
    
def depth_scan_metric_scatter(layers, metric_means : np.ndarray[float, float], metric_errors : np.ndarray[float, float], hits):
    fig, ax = plt.subplots(2,2,figsize=(9,7))
    ax = ax.flatten()
    metrics = {'rel_error': 'Relative Energy Error',
                'ari': 'Adjusted Rand Index (ARI)',
                'runtime': 'Runtime (s)',
                'gsp': 'Groundstate Probability'}
    optimisers = list(metric_means[layers[0]].keys())
    
    for idx, (metric, metric_name) in enumerate(metrics.items()):
        for optimiser_name in metric_means[layers[0]].keys():
            means = [metric_means[p][optimiser_name][metric] for p in layers]
            errors = [metric_errors[p][optimiser_name][metric] for p in layers]
            
            ax[idx].errorbar(layers, means, yerr=errors, fmt='o-', capsize=3, alpha=0.7, label=optimiser_name)
            
        if metric == 'gsp':
            baseline = 2 / (2**(2*hits))
            ax[idx].axhline(baseline, linestyle='--', label='Uniform baseline')
            ax[idx].legend()
            
        ax[idx].set_title(f'{metric_name}')
        ax[idx].set_xticks(layers)
        ax[idx].grid(True, alpha=0.2)
    
        
    handles, labels = ax[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper right', ncol=len(optimisers), bbox_to_anchor=(0.5, 0.98))
    
    fig.supxlabel('($p$) Layers')
    fig.supylabel('Metric value')
    fig.suptitle(f'QAOA Depth Scan Hits ($ N={hits} $)', x=0.8, fontsize=13)

    plt.tight_layout()
    plt.show()
    
##############################################################################################################################

    
def optimiser_energy_trace(restarts : int,  histories : np.ndarray[float],  best_history_idx : int, optimiser : str,
                           p: int, hits : int):
    fig, ax = plt.subplots(restarts, 1, figsize=(14,14))
    for j, history in enumerate(histories):
        ax[j].plot(history)
        ax[j].set_ylabel('Energy')
        
        if j == best_history_idx:
            best_energy = history[-1]

            ax[j].annotate('Best result',
                            xy=(len(history) - 1, best_energy),
                            xytext=(-60, 20),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='->'))
            
    fig.suptitle(f'{optimiser} Energy traces for ($p={p}$, $N={hits}$)')
    fig.supxlabel('COBYLA iteration')
    
    
    
def optimiser_result_energies(final_energies, optimiser, p, hits):
    plt.figure()
    x = np.arange(1,len(final_energies) + 1)
    plt.title(f'{optimiser} best energy evolution for ($p={p}$, $N={hits}$).')
    plt.xlabel('Restart iteration')
    plt.xticks(x)
    plt.ylabel('Energy')
    plt.plot(x, final_energies)
    
    
def plot_energy_hist(energies, true_groundstate_energy):
    plt.figure()
    plt.hist(energies, bins=30)
    plt.axvline(true_groundstate_energy, color='red', linestyle='--', label=f'Exact GS energy = {true_groundstate_energy:.2f}')
    plt.xlabel('Ising energy')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
        
            
def top_ten_states(counts):
    top_10 = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10])
    states = list(top_10.keys())
    frequencies = list(top_10.values())
    
    bar_colors = ["red" if state == "00001111" or state == "11110000" else "blue" for state in states]
        
    plt.bar(states, frequencies, color = bar_colors)
    plt.xlabel("Measured configuration")
    plt.ylabel("Counts")
    plt.title("10 Most Frequently Measured Configurations")
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
                
        
    
    