import numpy as np
from plotting import table_print as tp

from classical_algs import classical_benchmarks as cb
from problem.generatesystem import generate_toyproblem_params
from qaoa.optimisers import cobyla

from experiments.universal import quantum_scan, classical_scan
from experiments.scale import scale_scan
from experiments.depth import depth_scan

from global_params import (similarity_type,
                           graph_switch,
                           lambda_bal,
                           no_of_shots,
                           seed_lim,
                           restarts,
                           dep_noise_strengths,
                           readout_error_probability)
    
    
def run_experiment(mode, option, backend, sweet_spot, params):
    #Names of the classical algorithms used.
    classical_algs = {'Greedy' : cb.greedy_results, 
                      'Spectral Clustering': cb.spectral_results, 
                      'Simulated Annealing' : cb.sim_annealing_results}
    
    qaoa_optimisers = {'COBYLA' : cobyla}
    
    '''
    IMPORTANT NAMING CONVENTION:
    hits, hits_array are given as the number of hits in a single (true) track (hits per particle).
    Plotting should use 2*hits for the total number of hits detected.
    
    ALL ALGORITHMS, SYSTEM GENERATION, OPTIMISATION etc WORK WITH THE HALF #HITS.
    PLOTTING IS DONE WITH THE TOTAL #HITS
    
    Metrics in this code appear always in the following order: 
    1. relative energy error
    2. ari
    3. full runtime 
    4. gsp
    '''
    
    if option == 'depth':
        #Depth Scan fixes N varies p. 
        
        if mode == 1:
            '''
            If user just wishes to optimise and see how metrics vary in a depth scan.
            Here the sweet_spot is irrelevant since real hardware is not needed in mode=1. 
            The variables fixed_hits and layers can be chosen here.
            '''
            from global_params import depth_hits, depth_layers
            
            params = generate_toyproblem_params(depth_hits, lambda_bal, similarity_type, graph_switch)
            depth_scan(params,
                        similarity_type,
                        lambda_bal,
                        depth_hits,
                        np.array(depth_layers),
                        no_of_shots,
                        seed_lim,
                        restarts,
                        dep_noise_strengths,
                        readout_error_probability,
                        qaoa_optimisers,
                        sweet_spot,
                        mode,
                        backend) 
        
        elif mode == 2:
            '''
            If user asks to submit an actual job (mode='OPTIMISE-HARDWARE') then a depth scan is used to extract the optimal {γ, β}
            parameters. Depth scan is used because it uses a warm restart unlike the scale scan.
            
            In this case we then need to fixed the number of hits to that in the sweet spot, sweet_spot[0].
            To ensure a warm restart is used (p>1) then layers is set so the maximum is sweet_spot[1].
            '''
            fixed_hits = sweet_spot[0]
            layers = np.arange(1, sweet_spot[1]+1)      #Layers range is always pushed up 1 to use warm restarts.
            from global_params import threeway_no_of_shots      #Optimisation and job submission use the same number of shots.
            
            gamma, beta, ss_gammas, ss_betas, ata_circuit = depth_scan(params,                      
                                                                        similarity_type,
                                                                        lambda_bal,
                                                                        fixed_hits,
                                                                        layers,
                                                                        threeway_no_of_shots,
                                                                        seed_lim,
                                                                        restarts,
                                                                        dep_noise_strengths,
                                                                        readout_error_probability,
                                                                        qaoa_optimisers,
                                                                        sweet_spot,
                                                                        mode,
                                                                        backend)
                            
            return gamma, beta, ss_gammas, ss_betas, ata_circuit
    
    elif option == 'scale':
        #Scaling Scan fixes p varies N.
        from global_params import scale_hits, scale_layers
        
        scale_scan(similarity_type,
                   graph_switch,
                            lambda_bal,
                            np.array(scale_hits),
                            scale_layers,
                            no_of_shots,
                            seed_lim,
                            restarts,
                            dep_noise_strengths,
                            readout_error_probability,
                            qaoa_optimisers,
                            backend) 
        return None
                            
                            
    elif option == 'class':
        '''
        Class option is a universal comparison of all considered approaches, both classial and quantum.
        For fixed N and p.
        
        Result is a table for all classical methods, and another for all qaoa optimisers.
        '''
        
        from global_params import uni_hits, uni_layers, classical_loops
        
        params = generate_toyproblem_params(uni_hits, lambda_bal, similarity_type, graph_switch)
                
        classical_results = classical_scan(classical_algs, 
                                           params, 
                                           lambda_bal,
                                           classical_loops)
        tp.print_benchmark_table(uni_hits, 
                                 similarity_type, 
                                 lambda_bal, 
                                 classical_results)
                
                
        quantum_results, counts = quantum_scan(params,
                                        lambda_bal,
                                        uni_hits,
                                        uni_layers,
                                        no_of_shots,
                                        seed_lim,
                                        restarts,
                                        dep_noise_strengths,
                                        readout_error_probability,
                                        qaoa_optimisers,
                                        backend)
        
        tp.print_quantum_table(similarity_type,
                                 lambda_bal,
                                 uni_hits,
                                 uni_layers, 
                                 quantum_results,
                                 dep_noise_strengths,
                                 readout_error_probability, 
                                 option)
        
        return None
        

    
        
        