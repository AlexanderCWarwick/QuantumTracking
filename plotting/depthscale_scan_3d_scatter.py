import numpy as np
import matplotlib.pyplot as plt

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
    if metric_to_plot == 'rel_error':
        ax.set_zlabel(f'{metric_to_plot.replace('-',' ').title()} / 100')
    elif metric_to_plot == 'runtime':
        ax.set_zlabel(f'{metric_to_plot.replace('-',' ').title()} (secs)')
    else:
        ax.set_zlabel(metric_to_plot.replace('-',' ').title())
    
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
