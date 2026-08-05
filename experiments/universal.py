import numpy as np
from qaoa.qaoa import qaoa_results

def quantum_scan(params : tuple[np.ndarray[np.ndarray[float]], np.ndarray[int], float], 
                lambda_bal : float,
                N : int,
                p : int,
                no_of_shots : int,
                seed_lim : int,
                restarts : int,
                noise_strengths : tuple[np.float64, np.float64],
                readout_prob : float,
                qaoa_optimisers : dict,
                qpu_name : str,
                service) -> dict:
    '''
    Quantum scan over all listed optimiser in dictionary qaoa_optimiser.
    params ordering:
    1. similarity matrix (KNN OR RBF)
    2. the true gs, [0...01...1]
    3. the true gs energy.
    '''
    
    quantum_metrics = {N : {p : {name: {'rel_error': None,
                            'ari': None,
                            'runtime': None,
                            'gsp': None}
                            for name in qaoa_optimisers.keys()}}}
    
    warm_restarts = None
    
    for optimiser_name, optimiser in qaoa_optimisers.items():
        means, errors, _, _, best_counts, _ = qaoa_results(*params, 
                                           lambda_bal, 
                                           no_of_shots, 
                                           p, 
                                           seed_lim,
                                           optimiser,
                                           warm_restarts, 
                                           noise_strengths,
                                           readout_prob,
                                           restarts,
                                           qpu_name,
                                           service)
        
        for metric, mean, error in zip(quantum_metrics[N][p][optimiser_name].keys(), means, errors):
            quantum_metrics[N][p][optimiser_name][metric] = {'mean': mean,
                                                        'error': error}
    return quantum_metrics, best_counts



    
def classical_scan(classical_algs : dict, 
                   params : tuple[np.ndarray[np.ndarray[float]], np.ndarray[int], float],
                   lambda_bal : float) -> dict:
    '''
    Classical scan over all listed algorithms in dictionary classical_algs. Symmetric with quantum_scan.
    '''
    classical_metrics = {alg_name : {'rel_error' :[],
                         'ari' : [],
                         'runtime' : [],
                         'conv_frac' : []} for alg_name in classical_algs}
    
    classical_alg_loops = 10         #How many times to run each classical algorithm to obtain metric statistics.
    
    for alg_name, alg in classical_algs.items():
        means, errors = alg(*params, lambda_bal, classical_alg_loops)
        for metric, mean, std in zip(classical_metrics[alg_name].keys(), means, errors):
            classical_metrics[alg_name][metric] = {'mean': mean, 'error': std}
            
    return classical_metrics
    
    
    