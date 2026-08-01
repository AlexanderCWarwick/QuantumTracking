import numpy as np
import matplotlib.pyplot as plt
    
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
        
        if metric_name == 'rel_error':
            ax[idx].set_title(f'{metric_name} / 100')
        else:
            ax[idx].set_title(f'{metric_name}')
        ax[idx].set_xticks(2*track_hits)
        ax[idx].grid(True, alpha=0.2)
    
    
    info = (f'1-qubit gate noise: {float(noise_strengths[0])}\n'
            f'2-qubit gate noise: {float(noise_strengths[1])}\n')

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
            
        if metric_name == 'rel_error':
            ax[idx].set_title(f'{metric_name} / 100')
        else:
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
    
    
def depthscale_2d_metric_scatter(similarity_type, 
                                      layers,
                                      seed_lim,
                                      no_of_shots,
                                      lambda_bal, 
                                      noise_strengths, 
                                      metric_results : dict,
                                      metric_to_plot,
                                      optimiser_name,
                                      circuit_depth):
    
    hits = 2*np.array(list(metric_results.keys()))
    
    for N in hits:
        depth_metric_results = {metric_results[N] for N in hits}
        depth_scan_metric_scatter(layers, similarity_type, lambda_bal, noise_strengths, depth_metric_results, N, circuit_depth)
        
    for p in layers:
        scale_metric_results = {hits: metric_results[hits][p] for N in hits}
        scaling_scan_metric_scatter(hits, similarity_type, p, lambda_bal, scale_metric_results, noise_strengths)
    
    