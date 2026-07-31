import numpy as np
from qaoa.simple_qaoa import qaoa_results
from plotting.plotting import print_benchmark_table, print_quantum_table

def quantum_scan(qaoa_optimisers : dict,  
                 params : tuple[np.ndarray[np.ndarray[float]], np.ndarray[int], float], 
                 lambda_bal : float,
                 no_of_shots : int, 
                 p : int, 
                 seed_lim : int, 
                 noise_strengths : tuple[np.float64, np.float64],
                 readout_prob : float) -> dict:
    '''
    Quantum scan over all listed optimiser in dictionary qaoa_optimiser.
    params ordering:
    1. similarity matrix (KNN OR RBF)
    2. the true gs, [0...01...1]
    3. the true gs energy.
    '''
    
    quantum_metrics = {p : {name: {'rel_error': None,
                            'ari': None,
                            'runtime': None,
                            'gsp': None}
                            for name in qaoa_optimisers.keys()}}
    
    warm_restarts = None
    
    for optimiser_name, optimiser in qaoa_optimisers.items():
        means, errors, _, _ = qaoa_results(*params, 
                                           lambda_bal, 
                                           no_of_shots, 
                                           p, 
                                           seed_lim,
                                           optimiser,
                                           warm_restarts, 
                                           noise_strengths,
                                           readout_prob)
        
        for metric, mean, error in zip(quantum_metrics[p][optimiser_name].keys(), means, errors):
            quantum_metrics[p][optimiser_name][metric] = {'mean': mean,
                                                        'error': error}
    return quantum_metrics



    
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
    
    
    