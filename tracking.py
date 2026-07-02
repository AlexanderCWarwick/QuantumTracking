import numpy as np
from track_generation import construct_toytracks
import plotting as plot
from similarity import get_KNN_matrix, get_RBF_matrix
import classical_benchmarks as cb
import simple_qaoa as s_qaoa

    
#######################################################     Main workflow     #######################################################

def track_analysis(track_hits : int, algorithm_types : np.ndarray[str]): 
    '''
    Problem: Two particles pass through a detector each leaving N 'hits'. Using these 'hits' as coordinate
    locations of the particles, can we resolve the two tracks from each other and hence determine each particle's
    trajectory?
        
    Method:
    Two toy tracks (with noise) are randomly generated in 2D plane. 
    From the tracks obtain two types of similarity matrices:
    - RBF (radial basis function)
    - KNN (k nearest neighbours)
    Matrix elements measure the likelyhood that one hit is compatible to another, i.e. if they are the same particle.
    Obtain plots of these matrices as heatmaps.
    Obtain Graph representations of these similarity matrices.
    '''
    
    x = np.linspace(0,1,track_hits)         #Positions of detectors
    intersection_allowed = False            #Boolean to control whether particles intersect
    sigma_noise = 1e-2                      #External noise 
    nearneighb_n = 3                       #Number of nearest neighbours to consider in the KNN matrix
    
    lambda_bal = 1.0                 #Lambda_balance parameter values to be used in the Hamiltonian. Modelled as a constant.
    
    track0, track0_truthlabels, track1, track1_truthlabels = construct_toytracks(x, track_hits, sigma_noise, intersection_allowed)
    plot.plot_true_toytracks(x, track0, track1, intersection_allowed)
    
    hit_coords = np.column_stack([np.concatenate([x, x]),np.concatenate([track0, track1])])         #2D array of hit coordinates.   
    number_of_hits = len(hit_coords)                                                 
    hit_coords_dict = {i: tuple(hit_coords[i]) for i in range(number_of_hits)}          #Hit coordinates needed for plotting graph representations.
    
    
    KNN_matrix, nbrs = get_KNN_matrix(hit_coords, nearneighb_n)                      #nbrs only needed for graph visualisation.
    RBF_matrix = get_RBF_matrix(hit_coords)
    
    knn_G = plot.construct_KNN_graphrep(x, number_of_hits, hit_coords, hit_coords_dict, nbrs)
    rbf_G = plot.construct_RBF_graphrep(x, number_of_hits, hit_coords_dict, RBF_matrix)
    
#############################################################################################################################

    '''
    Turn problem into an optimisation problem. The optimal track is the one that minimises the energy objective, the ground state. Ideally these are
    000000111111 and 111111000000, the 'true ground states'.
    Here the energy objective is modelled as an Ising type function.
        
    The configuration space is the set of all possible hit labellings. Since each label is 0 or 1, this equates to assigning every configuration a binary string
    from 000000000000 up to 111111111111. This forms the binary configuration space with 2^12 = 4096 possibilities. 
        
    The objective function depends on a balance parameter, λ, which serves to discourage extreme configurations. 
    This simple optimisation technique will be tested for different λ values along with a ARI function to act as a numerical check on the randomness 
    of the calculated ground state. 
        
    The ARI check is symmetric to bit flops hence only need to use one of the two true groundstate tracks, true_groundstates[0] is used
    here but true_groundstates[1].
    
    ###############################################################################
    
    Week 2 Ising optimisation code. Do not run if testing higher values of track_hits, brute force technique will crash computer due to exponential order.
    
    KNN_energies, KNN_groundstate_energy, KNN_groundstate_binary_configs, RBF_energies, RBF_groundstate_energy, RBF_groundstate_binary_configs = ising_optimisation(number_of_hits, lambda_bal, KNN_matrix, RBF_matrix)
    plot.plot_energy_landscape(lambda_bal, KNN_energies, RBF_energies)
    
    KNN_aris = ARI_check(true_groundstate, KNN_groundstate_binary_configs)
    RBF_aris = ARI_check(true_groundstate, RBF_groundstate_binary_configs)
    
    '''
    true_groundstate = np.array(np.concatenate([track0_truthlabels, track1_truthlabels]))
    #The reference groundstate we use is 000000111111, but could just as well be the inverted state.
    
###############################################################################################################################
    
    true_groundstate_energy = cb.get_groundstate(RBF_matrix, true_groundstate, lambda_bal)
    rbf_similarity_params = (RBF_matrix, true_groundstate, true_groundstate_energy, lambda_bal)
    
    optimised_configs = []
    relative_energies = []
    aris = []
    times = []
    
    
    for algorithm in algorithm_types:
        if algorithm != 'QAOA':
            config, rel_energy, ari, time_elapsed, convergence_fraction = cb.run_classical_algorithm(algorithm, rbf_similarity_params)
            relative_energies.append(rel_energy)
            aris.append(ari)
            times.append(time_elapsed)
            optimised_configs.append(config)
        else:
            q_config, q_rel_energy, q_ari, _, _, q_time_elapsed = s_qaoa.qaoa_results(hit_coords, *rbf_similarity_params, rbf_G)
            relative_energies.append(q_rel_energy)
            aris.append(q_ari)
            times.append(q_time_elapsed)
            optimised_configs.append(q_config)
            
        
    plot.plot_optimised_benchmark_toytracks(hit_coords, optimised_configs, algorithm_types)
    
    
##################################################################################################################################
    

    print(aris)
    return np.array(relative_energies), np.array(aris), np.array(times), convergence_fraction



def main():
    algorithm_types = ['Greedy', 'Spectral Clustering', 'Simulated Annealing', 'QAOA']
    benchmark_times = []
    benchmark_aris = []
    relative_benchmark_energies = []
    conv_fractions = []
    
    hits_array = np.array([6])
    
    for hits in hits_array:
        np.random.seed(41)                  #Fixed random seed. Same for every number of track hits
        
        #Classical algorithm results
        rel_energies, aris, times, conv_frac = track_analysis(hits, algorithm_types)
        relative_benchmark_energies.append(rel_energies)
        benchmark_aris.append(aris)
        benchmark_times.append(times)
        conv_fractions.append(conv_frac)
        
        
    plot.print_benchmark_table(hits_array, algorithm_types, benchmark_aris, benchmark_times, relative_benchmark_energies, conv_fractions)
        
    
    
if __name__ == "__main__":
    main()