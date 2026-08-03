import numpy as np

def metric_stats(metrics_dict : dict) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    '''
    Input: metrics_dict contains lists of seed_lim values for each metric.
    Compute means and errors for each different metric given
    Output: Two seperate arrays for means and errors for each metric. The ordering is kept the same as the dictionary.
    relative energy -> ari -> runtime -> gsp
    '''
    
    metrics = metrics_dict.values()
    means = [np.mean(metric_values) for metric_values in metrics]
    std_metrics = [np.std(metric_values) for metric_values in metrics]
    
    return np.array(means), np.array(std_metrics)