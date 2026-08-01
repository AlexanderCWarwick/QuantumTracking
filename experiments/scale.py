import numpy as np
from classical.generatesystem import generate_toyproblem_params
from qaoa.simple_qaoa import qaoa_results

from plotting.scan_plots import scaling_scan_metric_scatter
from plotting.table_print import print_quantum_table

def scale_scan(track_hits : np.ndarray[int], 
               similarity_type : str,
               qaoa_optimisers : dict, 
               no_of_shots : int, 
               fixed_layers : int, 
               seed_lim : int, 
               lambda_bal : float,
               noise_strengths : tuple[np.float64, np.float64],
               readout_prob : float):
    
    '''
    SCALE SCAN -> VARY N
    
    In this experiment we plot how the groundstate probability changes with N for a given number of QAOA layers p.
    Since p isn't varied, the scale scan does not use warm restarts. warm_restarts therefore is Nonetype
    '''
    
    
    metric_results = {N  : {optimiser_name : {'rel_error' :[],
                                            'ari' : [],
                                            'runtime' : [],
                                            'gsp' : []} 
                            for optimiser_name in qaoa_optimisers.keys()} 
                      for N in track_hits}
                
    warm_restart = None    
    
    for hits in track_hits:
        params = generate_toyproblem_params(hits, lambda_bal, similarity_type)
         
        for optimiser_name, optimiser in qaoa_optimisers.items():
            means, errors, best_gammas, best_betas, _ = qaoa_results(*params, 
                                                                    lambda_bal, 
                                                                    no_of_shots, 
                                                                    fixed_layers, 
                                                                    seed_lim, 
                                                                    optimiser, 
                                                                    warm_restart, 
                                                                    noise_strengths,
                                                                    readout_prob)
            
            
            for i, metric in enumerate(metric_results[hits][optimiser_name].keys()):
                metric_results[hits][optimiser_name][metric] = {'mean' : means[i], 'error' : errors[i]}
               
        print_quantum_table(hits, 
                            similarity_type, 
                            lambda_bal, 
                            fixed_layers, 
                            noise_strengths, 
                            metric_results, 
                            'scale')
    scaling_scan_metric_scatter(track_hits, 
                                     similarity_type,
                                     fixed_layers,
                                     lambda_bal,
                                     metric_results,
                                     noise_strengths)
    
    return best_gammas, best_betas 
