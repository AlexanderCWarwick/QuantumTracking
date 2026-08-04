import numpy as np
import matplotlib.pyplot as plt
    
def scaling_scan_metric_scatter(similarity_type : str, 
                                lambda_bal : float,
                                track_hits : np.ndarray[int],
                                p : int, 
                                metric_results : dict,
                                noise_strengths : tuple[float, float], 
                                readout_prob : float):
    
    total_hits = 2 * track_hits
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
            ax[idx].errorbar(total_hits, means, yerr=errors, fmt='o-', capsize=3, alpha=0.7)
        
        if metric_name == 'rel_error':
            ax[idx].set_title(f'{metric_name} / 100')
        else:
            ax[idx].set_title(f'{metric_name}')
        ax[idx].set_xticks(total_hits)
        ax[idx].grid(True, alpha=0.2)
    
    
    info = (f'1-qubit gate noise: {float(noise_strengths[0])}\n'
            f'2-qubit gate noise: {float(noise_strengths[1])}\n'
            f'Readout error: {readout_prob}')

    fig.text(0.9, 0.9,
            info,
            ha='center',
            va='top',
            bbox=dict(boxstyle='round',
                    facecolor='white',
                    edgecolor='black',
                    alpha=0.8))
    
    fig.supxlabel('($N$) Hits')
    fig.supylabel('Metric value')
    fig.suptitle(f'QAOA Scale Scan ($ p={p}, W={similarity_type}, λ={lambda_bal}$) ', x=0.4, fontsize=13)

    plt.tight_layout()
    plt.savefig(f'assets/plots/ScalingScan_p{p}')
    
    
    
def depth_scan_metric_scatter(similarity_type : str, 
                              lambda_bal : float,
                              hits : int, 
                              layers : np.ndarray[int], 
                              metric_results : dict,
                              noise_strengths : float, 
                              readout_prob : float):
    
    total_hits = 2 * hits
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
            baseline = 2 / (2**(total_hits))
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
            f'Readout error: {readout_prob}')

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
    fig.suptitle(f'QAOA Depth Scan Hits ($ N={2*hits}, W={similarity_type}, λ={lambda_bal}$) ', x=0.4, fontsize=13)

    plt.tight_layout()
    plt.savefig(f'assets/plots/DepthScan_N{2*hits}')