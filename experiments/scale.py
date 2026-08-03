import numpy as np
from problem.generatesystem import generate_toyproblem_params
from qaoa.qaoa import qaoa_results

from plotting.scan_plots import scaling_scan_metric_scatter
from plotting.table_print import print_quantum_table
    
def scale_scan(similarity_type : str,
                   lambda_bal : float,
                   hits_array : np.ndarray[int],
                   fixed_p : int,
                   no_of_shots : int,
                   seed_lim : int,
                   restarts : int,
                   noise_strengths : tuple[np.float64, np.float64],
                   readout_prob : float,
                   qaoa_optimisers : dict):
    
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
                      for N in hits_array}
                
    warm_restart = None    
    
    for hits in hits_array:
        params = generate_toyproblem_params(hits, lambda_bal, similarity_type)
         
        for optimiser_name, optimiser in qaoa_optimisers.items():
            means, errors, best_gammas, best_betas = qaoa_results(*params, 
                                                                    lambda_bal, 
                                                                    no_of_shots, 
                                                                    fixed_p, 
                                                                    seed_lim, 
                                                                    optimiser, 
                                                                    warm_restart, 
                                                                    noise_strengths,
                                                                    readout_prob,
                                                                    restarts)
            
            
            for i, metric in enumerate(metric_results[hits][optimiser_name].keys()):
                metric_results[hits][optimiser_name][metric] = {'mean' : means[i], 'error' : errors[i]}
               
        print_quantum_table(similarity_type, 
                                lambda_bal, 
                                hits, 
                                fixed_p, 
                                metric_results, 
                                noise_strengths, 
                                readout_prob, 
                                'scale')
        
    scaling_scan_metric_scatter(similarity_type,
                                     lambda_bal,
                                     hits_array,
                                     fixed_p,
                                     metric_results,
                                     noise_strengths,
                                     readout_prob)
    
    return best_gammas, best_betas 
