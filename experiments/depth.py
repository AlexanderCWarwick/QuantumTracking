import numpy as np
from classical.generatesystem import generate_toyproblem_params
from qaoa.simple_qaoa import qaoa_results

from plotting.scan_plots import depth_scan_metric_scatter
from plotting.table_print import print_quantum_table

def depth_scan(params : tuple[np.ndarray[np.ndarray[float]], np.ndarray[int], float],
               similarity_type : str,
               fixed_hits : np.ndarray[int], 
               layers : np.ndarray[int], 
               lambda_bal : float, 
               qaoa_optimisers : dict, 
               no_of_shots : int, 
               seed_lim : int,
               noise_strengths : tuple[np.float64, np.float64],
               readout_prob : float):
    '''
    DEPTH SCAN -> VARY p
    
    In this experiment, we plot how our four success metrics change with p. 
    '''
    
    metrics = {p  : {name : {'rel_error' :[],
                                'ari' : [],
                                'runtime' : [],
                                'gsp' : []} for name in qaoa_optimisers.keys()} for p in layers}
       
    '''
    Above dictionaries are arranged like this:
    p -> optimiser name -> metric -> mean and error
    '''
    
    
    '''
    previous_params stores the best params obtained from the (p-1)th search. 
    So if p=2, then the first restart used in the cobyla function will use the p=1 best params.
    The remaining restarts are random.
    '''    
        
    previous_params = {name: None for name in qaoa_optimisers.keys()}
    params = generate_toyproblem_params(fixed_hits, lambda_bal, similarity_type)
        
    for p in layers:
        for optimiser_name, optimiser in qaoa_optimisers.items():
            if optimiser_name == 'COBYLA':
                warm_restart = previous_params[optimiser_name]
            else:
                warm_restart = None
                        
            means, errors, best_gammas, best_betas, circuit_depth = qaoa_results(*params, 
                                                                        lambda_bal, 
                                                                        no_of_shots, 
                                                                        p, 
                                                                        seed_lim, 
                                                                        optimiser, 
                                                                        warm_restart, 
                                                                        noise_strengths,
                                                                        readout_prob)
                    
                
            if optimiser_name != 'Grid':
                previous_params[optimiser_name] = np.concatenate([best_gammas, best_betas])
                    
            for i, metric in enumerate(metrics[p][optimiser_name].keys()):
                metrics[p][optimiser_name][metric] = {'mean' : means[i], 'error' : errors[i]}
                    
        print_quantum_table(fixed_hits, similarity_type, lambda_bal, p, noise_strengths, metrics)
    depth_scan_metric_scatter(layers, similarity_type, lambda_bal, noise_strengths, metrics, fixed_hits, circuit_depth)
