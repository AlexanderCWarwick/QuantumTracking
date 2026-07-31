import numpy as np
from classical.generatesystem import generate_toyproblem_params
from qaoa.simple_qaoa import qaoa_results
from plotting.plotting import depthscale_3d_scan_metric_scatter, print_quantum_table

def depthscale_scan(similarity_type : str,
                    hits_array : np.ndarray[int], 
                    layers : np.ndarray[int], 
                    lambda_bal : float, 
                    qaoa_optimisers : dict, 
                    no_of_shots : int, 
                    seed_lim : int,
                    noise_strengths : tuple[np.float64, np.float64],
                    readout_prob : float,
                    sus_sweet_spot : tuple[int, int]):
    '''
    DEPTH+SCALE SCAN -> VARY p and N
    
    In this experiment, we plot any of the four success metrics change with p and N in a 3D scatter plot.
    '''
    
    metrics = {hits : {p  : {name : {'rel_error' :[],
                                'ari' : [],
                                'runtime' : [],
                                'gsp' : []} for name in qaoa_optimisers.keys()} for p in layers} for hits in hits_array}
       
    '''
    Above dictionaries are arranged like this:
    p -> optimiser name -> metric -> mean and error
    '''
    
    
    '''
    previous_params stores the best params obtained from the (p-1)th search. 
    So if p=2, then the first restart used in the cobyla function will use the p=1 best params.
    The remaining restarts are random.
    '''
    
    for hits in hits_array:    
        
        previous_params = {name: None for name in qaoa_optimisers.keys()}
        params = generate_toyproblem_params(hits, lambda_bal, similarity_type)
        
        for p in layers:
            for optimiser_name, optimiser in qaoa_optimisers.items():
                if optimiser_name == 'COBYLA':
                    warm_restart = previous_params[optimiser_name]
                else:
                    warm_restart = None
                        
                means, errors, best_gammas, best_betas, _ = qaoa_results(*params, 
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
                    
                for i, metric in enumerate(metrics[hits][p][optimiser_name].keys()):
                    metrics[hits][p][optimiser_name][metric] = {'mean' : means[i], 'error' : errors[i]}
                    
                if (hits, p) == sus_sweet_spot:
                    sweet_spot_gammas = best_gammas                    
                    sweet_spot_betas = best_betas
                    
            print_quantum_table(hits, similarity_type, lambda_bal, p, noise_strengths, metrics[hits])
        
    metric_name_list = ['rel_error', 'ari', 'runtime', 'gsp']
    for metric_name in metric_name_list:
        depthscale_3d_scan_metric_scatter(similarity_type,
                                                layers, 
                                                seed_lim,
                                                no_of_shots,
                                                lambda_bal,
                                                noise_strengths,
                                                metrics,
                                                metric_name,
                                                'COBYLA')
        
    return np.array([sweet_spot_gammas, sweet_spot_betas])

    
