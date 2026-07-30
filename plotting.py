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
    
def print_benchmark_table(hits : int, similarity_type : str, lambda_bal : float, classical_results : dict[dict]):
    header = (f'{'Hits':<8}'
        f'{'Classical Algorithm':<30}'
        f'{'Relative Energy Error':<30}'
        f'{'ARI':<20}'
        f'{'Time (s)':<20}'
        f'{'Convergence Fraction':<30}')
    
    print(f'Parameters : sim_matrix={similarity_type}, λ={lambda_bal}')
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
            
        
        
def print_quantum_table(hits : int, similarity_type : str, lambda_bal : float,  p : int, noise_strengths : tuple[float], quantum_results : dict):
    header = (f'{'Hits':<8}'
        f'{'QAOA Optimiser':<20}'
        f'{'Relative Energy Error':<30}'
        f'{'ARI':<20}'
        f'{'Time (s)':<20}'
        f'{'GS Prob':<20}')
    
    print(f'Parameters : sim_matrix={similarity_type}, λ={lambda_bal}')
    print(f'Layers p = {p}, Depolarising Noise (1q, 2q) = {noise_strengths}')
    print(header)
    print('-' * len(header))
    
    for optimiser_name, metrics in quantum_results[p].items():
        print(f'{2*hits:<8}'
            f'{optimiser_name:<20}'
            f'{format_metric(metrics['rel_error']):<30}'
            f'{format_metric(metrics['ari']):<20}'
            f'{format_metric(metrics['runtime']):<20}'
            f'{format_metric(metrics['gsp']):<20}')
    print('\n')

    
def scaling_scan_metric_scatter(track_hits : np.ndarray[int], 
                                similarity_type : str, 
                                p : int,
                                lambda_bal : float,
                                metric_results : dict,
                                noise_strengths : tuple[float, float]):
    
    fig, ax = plt.subplots(2,2,figsize=(9,7))
    ax = ax.flatten()
    metrics = {'rel_error': 'Relative Energy Error',
                'ari': 'Adjusted Rand Index (ARI)',
                'runtime': 'Runtime (s)',
                'gsp': 'Groundstate Probability'}
    
    for idx, (metric, metric_name) in enumerate(metrics.items()):
        for optimiser_name in metric_results[track_hits[0]].keys():
            
            means = [metric_results[N][optimiser_name][metric]['mean'] for N in track_hits]
            errors = [metric_results[N][optimiser_name][metric]['error'] for N in track_hits]
            ax[idx].errorbar(2*track_hits, means, yerr=errors, fmt='o-', capsize=3, alpha=0.7)
        
        ax[idx].set_title(f'{metric_name}')
        ax[idx].set_xticks(2*track_hits)
        ax[idx].grid(True, alpha=0.2)
    
    
    info = (f'1-qubit gate noise: {float(noise_strengths[0])}\n'
            f'2-qubit gate noise: {float(noise_strengths[1])}\n'
            f'Sim Matrix: {similarity_type}\n'
            f'λ: {lambda_bal}')

    fig.text(0.9, 0.9,
            info,
            ha='center',
            va='top',
            bbox=dict(boxstyle='round',
                    facecolor='white',
                    edgecolor='black',
                    alpha=0.8))
    
    fig.supxlabel('($p$) Layers')
    fig.supylabel('Metric value')
    noises = (float(noise) for noise in noise_strengths)
    fig.suptitle(f'QAOA Scale Scan ($ p={p}, W={similarity_type}, λ={lambda_bal}$) ', x=0.4, fontsize=13)

    plt.tight_layout()
    plt.show()
    
    
    
def depth_scan_metric_scatter(layers, similarity_type, lambda_bal, noise_strengths, metric_results, hits, circuit_depth):
    fig, ax = plt.subplots(2,2,figsize=(9,7))
    ax = ax.flatten()
    metrics = {'rel_error': 'Relative Energy Error',
                'ari': 'Adjusted Rand Index (ARI)',
                'runtime': 'Runtime (s)',
                'gsp': 'Groundstate Probability'}
    
    for idx, (metric, metric_name) in enumerate(metrics.items()):
        for optimiser_name in metric_results[layers[0]].keys():
            
            means = [metric_results[p][optimiser_name][metric]['mean'] for p in layers]
            errors = [metric_results[p][optimiser_name][metric]['error'] for p in layers]
            ax[idx].errorbar(layers, means, yerr=errors, fmt='o-', capsize=3, alpha=0.7)
            
        if metric == 'gsp':
            baseline = 2 / (2**(2*hits))
            ax[idx].axhline(baseline, linestyle='--', label=f'Uniform baseline {baseline:.4f}')
            ax[idx].legend()
            
        ax[idx].set_title(f'{metric_name}')
        ax[idx].set_xticks(layers)
        ax[idx].grid(True, alpha=0.2)
    
    
    info = (f'1-qubit gate noise: {float(noise_strengths[0])}\n'
            f'2-qubit gate noise: {float(noise_strengths[1])}\n'
            f'Circuit depth: {circuit_depth}')

    fig.text(0.82, 0.85,
            info,
            ha='center',
            va='top',
            bbox=dict(boxstyle='round',
                    facecolor='white',
                    edgecolor='black',
                    alpha=0.8))
    
    fig.supxlabel('($p$) Layers')
    fig.supylabel('Metric value')
    noises = (float(noise) for noise in noise_strengths)
    fig.suptitle(f'QAOA Depth Scan Hits ($ N={2*hits}, W={similarity_type}, λ={lambda_bal}$) ', x=0.4, fontsize=13)

    plt.tight_layout()
    plt.show()
    
    
def depthscale_3d_scan_metric_scatter(similarity_type, 
                                      layers,
                                      seed_lim,
                                      no_of_shots,
                                      lambda_bal, 
                                      noise_strengths, 
                                      metric_results : dict,
                                      metric_to_plot,
                                      optimiser_name):
    
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    hits_array = np.array(list(metric_results.keys()))
    
    hits_cmap = plt.get_cmap('viridis', len(hits_array))
    hit_colours = {hits: hits_cmap(i) for i, hits in enumerate(hits_array)}
    
    p_cmap = plt.get_cmap('magma', len(layers))
    p_colours = {p: p_cmap(i) for i, p in enumerate(layers)}
    
    for hits, depth_results in metric_results.items():
        hit_colour = hit_colours[hits]
        hits = 2 * hits
        for p, optimiser_results in depth_results.items():
            p_colour = p_colours[p]
            mean = optimiser_results[optimiser_name][metric_to_plot]['mean']
            error = optimiser_results[optimiser_name][metric_to_plot]['error']

            ax.scatter(hits, p, mean, s=40, color=hit_colour,)
            
            ax.plot([hits, hits], 
                    [p, p], 
                    [mean - error, mean + error], 
                    linewidth=1.5,
                    color=p_colour)
            
            cap_width = 0.07
            ax.plot([hits - cap_width, hits + cap_width],
                    [p, p],
                    [mean - error, mean - error],
                    linewidth=1.5,
                    color=p_colour)

            ax.plot([hits - cap_width, hits + cap_width],
                    [p, p],
                    [mean + error, mean + error],
                    linewidth=1.5,
                    color = p_colour)
                
    ax.set_xlabel('Number of hits')
    ax.set_ylabel('QAOA depth $p$')
    ax.set_zlabel(metric_to_plot.replace("_", " ").title())
    
    ax.set_xticks(2*hits_array)
    ax.set_yticks(layers)
    
    
    info = (f'Optimiser: {optimiser_name}\n'
            f'Noise: {noise_strengths}\n'
            f'λ: {lambda_bal}\n'
            f'Sim Matrix: {similarity_type}\n'
            f'Seeds: {seed_lim}\n'
            f'#Shots: {no_of_shots}')
    
    ax.text2D(0.02,
            0.98,
            info,
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle='round',
                        facecolor='white',
                        edgecolor='black',
                        alpha=0.8))
    
    
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
        
            
def top_ten_states(counts : dict, true_gs : np.ndarray[int]):
    top_10 = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10])      #Gets top 10 states in {state : count} form.
    states = list(top_10.keys())                                                            #List conversion of states
    frequencies = list(top_10.values())                                                     #List conversion of counts
    
    inv_true_gs = ''.join((true_gs^1).astype(str))                      #Inverse true_gs as str
    true_gs = ''.join(true_gs.astype(str))                              #true_gs as str
    
    bar_colors = ["red" if state == true_gs or state == inv_true_gs else "blue" for state in states]
        
    plt.bar(states, frequencies, color = bar_colors)
    plt.xlabel("Measured configuration")
    plt.ylabel("Counts")
    plt.title("10 Most Frequently Measured Configurations")
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
                
        
    
    