import numpy as np
from plotting import table_print as tp

from classical_algs import classical_benchmarks as cb
from problem.generatesystem import generate_toyproblem_params
from qaoa.optimisers import cobyla

from experiments.universal import quantum_scan, classical_scan
from experiments.scale import scale_scan
from experiments.depth import depth_scan
from experiments.depthscale import depthscale_scan

from global_params import (option, 
                           similarity_type,
                           lambda_bal,
                           no_of_shots,
                           seed_lim,
                           restarts,
                           noise_strengths,
                           readout_prob)
    
    
def main():
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
        fixed_hits = 4
        layers = np.arange(1, 4)
        
        #params = (similarity_matrix, true_groundstate, true_groundstate_energy)
        #Can include before the depth_scan call since params doesn't change with p.
        params = generate_toyproblem_params(fixed_hits, lambda_bal, similarity_type)
        
        depth_scan(params, 
                   similarity_type,
                   lambda_bal,
                   fixed_hits,
                   layers,
                   no_of_shots,
                   seed_lim,
                   restarts,
                   noise_strengths,
                   readout_prob,
                   qaoa_optimisers)  
        
    
    elif option == 'scale':
        #Scaling Scan fixes p varies N.
        hits_array = np.array([3,4,5])
        fixed_p = 2
        
        scale_scan(similarity_type,
                   lambda_bal,
                   hits_array,
                   fixed_p,
                   no_of_shots,
                   seed_lim,
                   restarts,
                   noise_strengths,
                   readout_prob,
                   qaoa_optimisers) 
                
                
    elif option == 'class':
        '''
        Class option is a universal comparison of all considered approaches, both classial and quantum.
        For fixed N and p.
        
        Result is a table for all classical methods, and another for all qaoa optimisers.
        '''
        
        #For a fixed N and p, compare all algorithms in one table.
        hits = 3
        layers = 1
        
        params = generate_toyproblem_params(hits, lambda_bal, similarity_type)
                
        classical_results = classical_scan(classical_algs, 
                                           params, 
                                           lambda_bal)
        tp.print_benchmark_table(hits, 
                                 similarity_type, 
                                 lambda_bal, 
                                 classical_results)
                
                
        quantum_results = quantum_scan(params,
                                        lambda_bal,
                                        hits,
                                        layers,
                                        no_of_shots,
                                        seed_lim,
                                        restarts,
                                        noise_strengths,
                                        readout_prob,
                                        qaoa_optimisers)
        tp.print_quantum_table(similarity_type,
                                 lambda_bal,
                                 hits,
                                 layers, 
                                 quantum_results,
                                 noise_strengths,
                                 readout_prob, 
                                 option)
        
    elif option == 'depth-scale':
        hits_array = np.array([3,4,5])
        layers = np.arange(1,4)
        sweet_spot = (4,1)
        
        sweet_spot_gammas, sweet_spot_betas = depthscale_scan(similarity_type,
                                                                hits_array,
                                                                layers,
                                                                lambda_bal,
                                                                qaoa_optimisers,
                                                                no_of_shots,
                                                                seed_lim,
                                                                restarts,
                                                                noise_strengths,
                                                                readout_prob,
                                                                sweet_spot)
        
        print(sweet_spot_gammas, sweet_spot_betas)
        
        
if __name__ == "__main__":
    main()