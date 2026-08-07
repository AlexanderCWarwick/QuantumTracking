import numpy as np
import matplotlib.pyplot as plt

def threeway_data_analysis(threeway_comp):
    summary = {}
    metrics = ['rel_error', 'ari', 'runtime', 'gsp']

    for app_name, app_data in threeway_comp.items():

        results = app_data['results']
        summary[app_name] = {}

        for metric in metrics:
            values = [data[metric] for data in results.values()]

            summary[app_name][metric] = {'mean': np.mean(values),
                                        'error': np.std(values)}
            
    '''
    summary dictionary structure:
    summary = {'clean' : {'rel_error' : {'mean' : 0, 'error' : 0}, 'ari' : {'mean' : 0, 'error' : 0}, ...,
                'noisy' : {'rel_error' : {'mean' : 0, 'error' : 0}, 'ari' : {'mean' : 0, 'error' : 0}, ...,
                'real' : {'rel_error' : {'mean' : 0, 'error' : 0}, 'ari' : {'mean' : 0, 'error' : 0}, ...}}'''
                
    return summary


def plot_threeway_metrics(sweet_spot : tuple[int, int],
                          similarity_type : str, 
                          lambda_bal : float, 
                          threeway_no_of_shots : int, 
                          readout_error_probability : float,
                            dep_noise_strengths : float,
                          summary_statistics : dict):
    colours = {'clean': 'tab:blue',
            'noisy': 'tab:orange',
            'real': 'tab:red'}
    bar_colours = [colours[name] for name in colours.keys()]
    
    baseline_gsp = 2 / 2 ** (2 * sweet_spot[0])
    metrics = ['rel_error', 'ari', 'gsp']
    fig, ax = plt.subplots(len(metrics), figsize=(11,11))
    for i, metric in enumerate(metrics):
        names = []
        means = []
        errors = []

        for app_name, app_data in summary_statistics.items():

            names.append(app_name)
            means.append(app_data[metric]['mean'])
            errors.append(app_data[metric]['error'])
            
        means = np.array(means)
        errors = np.array(errors)
        x = np.arange(len(metrics))
        
        ax[i].bar(x,
                2 * errors,
                bottom=means - errors,
                color = bar_colours,
                width=0.5,
                alpha=0.3)
        
        ax[i].bar(x, 
                  means,
                  color = bar_colours,
                  width=0.35)
        
        
        ax[i].set_ylabel(metric)
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(names)
        ax[i].set_title(f'{metric} Comparison')
        

        if metric == 'gsp':
            ax[i].set_ylim(0, 1.2)
            ax[i].axhline(y=baseline_gsp, 
                          color='black',
                        linestyle='--',
                        linewidth=1)
            
        elif metric == 'ari':
            ax[i].set_ylim(-0.5, 1.2)
            ax[i].axhline(y=0, 
                          color='grey',
                          linestyle='--',
                          linewidth=1)
    
    info = (f'(N,p) = {sweet_spot}\n'
            f'Similarity Type: {similarity_type}\n'
            f'Lambda = {lambda_bal}\n'
            f'#Shots = {threeway_no_of_shots}\n'
            f'ReadoutProb = {readout_error_probability}\n'
            f'Depolar error (1q,2q) = {dep_noise_strengths}')
    
    fig.suptitle('Metric Comparison')
    fig.suptitle('Metric Comparison')
    fig.subplots_adjust(hspace=0.3)

    fig.text(0.78, 0.85,
            info,
            ha='left',
            va='top',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                edgecolor='black',
                alpha=0.8))

    plt.tight_layout(rect=[0, 0, 0.75, 0.95])
    plt.show()
  


        
        