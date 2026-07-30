import numpy as np
import plotting as plot
import classical_benchmarks as cb
from simple_qaoa import grid, cobyla, cobyqa, qaoa_results
from generatesystem import generate_toyproblem_params
    
'''       
def scale_scan(track_hits : np.ndarray[int], 
               similarity_type : str,
               qaoa_optimisers : dict, 
               no_of_shots : int, 
               fixed_layers : int, 
               seed_lim : int, 
               lambda_bal : float,
               noise_strengths : tuple[np.float64, np.float64],):
    

    SCALE SCAN -> VARY N
    
    In this experiment we plot how the groundstate probability changes with N for a given number of QAOA layers p.
    
    
    raw_results = {name : [] for name in qaoa_optimisers.keys()}
    raw_errors = {name : [] for name in qaoa_optimisers.keys()}
    
    rel_results = {name : [] for name in qaoa_optimisers.keys()}
    rel_errors = {name : [] for name in qaoa_optimisers.keys()}
    
    warm_restart = None     #Scale scan does not use warm restarts.
    
    for hits in track_hits:
        params = generate_toyproblem_params(hits, lambda_bal, similarity_type)
         
        for optimiser_name, optimiser in qaoa_optimisers.items():
            means, stds, _, _ = qaoa_results(*params, 
                                             lambda_bal, 
                                             no_of_shots, 
                                             fixed_layers, 
                                             seed_lim, 
                                             optimiser, 
                                             warm_restart, 
                                             noise_strengths)
            
            baseline = 2 / (2**(2*hits))
            #The groundstate probability is the final entry of the means/std output arrays of the qaoa_results
            relative_gs_prob = means[-1] / baseline
            relative_error = stds[-1] / baseline
            
            raw_results[optimiser_name].append(means[-1])
            raw_errors[optimiser_name].append(stds[-1])
            
            rel_results[optimiser_name].append(relative_gs_prob)
            rel_errors[optimiser_name].append(relative_error)
            
    
    plot.scaling_scan_metric_scatter(track_hits, 
                                     similarity_type,
                                     raw_results, raw_errors,
                                     rel_results, rel_errors)
'''    

def depth_scan(params : tuple[np.ndarray[np.ndarray[float]], np.ndarray[int], float], 
               similarity_type : str,
               fixed_hits : np.ndarray[int], 
               layers : np.ndarray[int], 
               lambda_bal : float, 
               qaoa_optimisers : dict, 
               no_of_shots : int, 
               seed_lim : int,
               noise_strengths : tuple[np.float64, np.float64]):
    '''
    DEPTH SCAN -> VARY p
    
    In this experiment, we plot how our four success metrics change with p. 
    '''
    
    metrics = {fixed_hits : {p  : {name : {'rel_error' :[],
                                'ari' : [],
                                'runtime' : [],
                                'gsp' : []} for name in qaoa_optimisers.keys()} for p in layers}}
       
    '''
    Above dictionaries are arranged like this:
    p -> optimiser name -> metric -> mean and error
    '''
    
    previous_params = {name: None for name in qaoa_optimisers.keys()}
    
    '''
    previous_params stores the best params obtained from the (p-1)th search. 
    So if p=2, then the first restart used in the cobyla function will use the p=1 best params.
    The remaining restarts are random.
    '''
        
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
                                                                    noise_strengths)
                
            
            if optimiser_name != 'Grid':
                previous_params[optimiser_name] = np.concatenate([best_gammas, best_betas])
                
            for i, metric in enumerate(metrics[fixed_hits][p][optimiser_name].keys()):
                metrics[fixed_hits][p][optimiser_name][metric] = {'mean' : means[i], 'error' : errors[i]}
                
            
        plot.print_quantum_table(fixed_hits, similarity_type, lambda_bal, p, noise_strengths, metrics[fixed_hits])
    plot.depth_scan_metric_scatter(layers, similarity_type, lambda_bal, noise_strengths, metrics[fixed_hits], fixed_hits, circuit_depth)

    
    
def quantum_scan(qaoa_optimisers : dict,  
                 params : tuple[np.ndarray[np.ndarray[float]], np.ndarray[int], float], 
                 lambda_bal : float,
                 no_of_shots : int, 
                 p : int, 
                 seed_lim : int, 
                 noise_strengths : tuple[np.float64, np.float64]) -> dict:
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
        means, errors, _, _ = qaoa_results(*params, lambda_bal, no_of_shots, p, seed_lim, optimiser, warm_restarts, noise_strengths)
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
    
    
    
def main():
    option = 'depth-scale'                  #This is the identifier for which 'task' we want to do.
    no_of_shots =  20               #Number of measurements the quantum simulator will make of the circuit (all independent).
    seed_lim = 3                    #Number of runs of the QAOA to calculate means and errors.
    lambda_bal = 0.25                  #Lambda_balance parameter values to be used in the Hamiltonian. Modelled as a constant.
    similarity_type = 'KNN'
    
    
    single_qubit_noises = np.array([0, 1e-3])       #Single qubit gate depolarisation errors
    double_qubit_noises = 5 * single_qubit_noises                           #Two qubit gate depolarisation errors
    
    #Names of the classical algorithms used.
    classical_algs = {'Greedy' : cb.greedy_results, 
                      'Spectral Clustering': cb.spectral_results, 
                      'Simulated Annealing' : cb.sim_annealing_results}
    
    #All the different optimisers used in the qaoa. simple_qaoa contains three to use: grid, cobyla and cobyqa.
    qaoa_optimisers = {'COBYLA' : cobyla}
    
    '''
    IMPORTANT NAMING CONVENTION:
    hits, hits_array are given as the number of hits in a single (true) track (hits per particle).
    Plotting should use 2*hits for the total number of hits detected.
    
    Metrics in this code appear always in the following order: 
    1. relative energy error
    2. ari
    3. full runtime 
    4. gsp
    '''
    
    if option == 'depth-scale':
        '''
        Combined depth + scale scan to be used to find the sweet spot under noise-depth balance.
        '''
        
        hits_array = np.array([4,5])
        for noise_strengths in zip(single_qubit_noises, double_qubit_noises):
            for fixed_hits in hits_array:
                #Depth Scan fixes N varies p.
                #fixed_hits = 4                   #Hits per track -> 2*fixed_hits = Total hits detected.
                layers = np.arange(1, 3)
                    
                #params = (similarity_matrix, true_groundstate, true_groundstate_energy)
                #Can include before the depth_scan call since params doesn't change with p.
                params = generate_toyproblem_params(fixed_hits, lambda_bal, similarity_type)
                    
                depth_scan(params, 
                            similarity_type,
                            fixed_hits, 
                            layers, 
                            lambda_bal, 
                            qaoa_optimisers, 
                            no_of_shots, 
                            seed_lim, 
                            noise_strengths)        
                
    elif option == 'class':
        '''
        Class option is a universal comparison of all considered approaches, both classial and quantum.
        For fixed N and p
        '''
        
        #For a fixed N and p, compare all algorithms in one table.
        hits = 3
        layers = 2
        params = generate_toyproblem_params(hits, lambda_bal, similarity_type)
                
        classical_results = classical_scan(classical_algs, params, lambda_bal)
        plot.print_benchmark_table(hits, similarity_type, classical_results)
                
        for noise_strengths in zip(single_qubit_noises, double_qubit_noises):
            quantum_results = quantum_scan(qaoa_optimisers, params, lambda_bal, no_of_shots, layers, seed_lim, noise_strengths)
            plot.print_quantum_table(hits, similarity_type, layers, noise_strengths, quantum_results)
                        
                    
    '''            
    elif option == 'scale':
        for noise_strengths in zip(single_qubit_noises, double_qubit_noises):
            #Scaling Scan fixes p varies N.
            hits_array = np.array([3,4,5,6])
            fixed_layers = 2
            
            scale_scan(hits_array, similarity_type, qaoa_optimisers, no_of_shots, fixed_layers, seed_lim, lambda_bal, noise_strengths)
    '''       
            
 
        
if __name__ == "__main__":
    main()