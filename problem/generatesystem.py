import numpy as np
from global_params import intersection_allowed, track_noise
from plotting.plotting_baseprob import energy_landscape
from problem.similarity import get_KNN_matrix, get_RBF_matrix
from problem.track_generation import construct_toytracks
from classical_algs.brute_force import ising_optimisation
from classical_algs.ari import ari_check
import plotting.plotting_baseprob as plot_base


def toy_track_generation(track_hits : int, 
                         x : np.ndarray, 
                         sigma_noise : float,
                         intersection_allowed : bool) -> tuple[list[float], list[int], list[float], list[int]]:
    '''
    Make toy tracks. All construction takes place in the track_generation.py file. 
    First source of noise is introduced : hit signal noise.
    
    Intersecting particle tracks are far more complicated to deal with. Toggled by intersection_allowed boolean.
    '''
    
    track0, track0_truthlabels, track1, track1_truthlabels = construct_toytracks(x, track_hits, sigma_noise, intersection_allowed)
    from plotting.plotting_baseprob import true_toytracks
    #true_toytracks(x, track0, track1, intersection_allowed)
    return track0, track0_truthlabels, track1, track1_truthlabels
    
    
    
def sim_matrices_calculation(x, track0, track1, similarity_type : str, graph_plot_switch : bool):
    '''
    Similarity matrices to be calculated are
    - KNN. Here nearneighb_n is k.
    - RBF. Continuous measure.
    
    Plotting functionality of the similarity matrices as heat maps and graph networks is included.
    KNN nearest neighbours is by default 3.
    '''
    
    from global_params import nearneighb_n, rbf_sigma
    hit_coords = np.column_stack([np.concatenate([x, x]),np.concatenate([track0, track1])])         #2D array of hit coordinates.   
                                                    
    if similarity_type == 'KNN':
        knn_matrix, nbrs = get_KNN_matrix(hit_coords, nearneighb_n)             #nbrs only needed for graph visualisation.
        
        if graph_plot_switch:
            plot_knn_matrix(x, hit_coords, knn_matrix, nbrs)

        return knn_matrix
    elif similarity_type == 'RBF':
        rbf_matrix = get_RBF_matrix(hit_coords, rbf_sigma)
        
        if graph_plot_switch:
            plot_rbf_matrix(x, hit_coords, rbf_matrix)
            
        return rbf_matrix
    
    
def plot_knn_matrix(x, hit_coords, knn_matrix, nbrs):
    
    number_of_hits = len(knn_matrix)
    hit_coords_dict = {i: tuple(hit_coords[i]) for i in range(number_of_hits)}          #Hit coordinates needed for plotting graph representations.
    knn_G, knn_edges = plot_base.construct_KNN_graphrep(number_of_hits, hit_coords, nbrs)
    plot_base.graphrep(knn_G, x, hit_coords_dict, knn_edges, None, 'KNN')
    
    
def plot_rbf_matrix(x, hit_coords, rbf_matrix):
        
    number_of_hits = len(rbf_matrix)
    hit_coords_dict = {i: tuple(hit_coords[i]) for i in range(number_of_hits)}     
    rbf_G, rbf_edges, edge_contrasts = plot_base.construct_RBF_graphrep(number_of_hits, rbf_matrix)
    plot_base.graphrep(rbf_G, x, hit_coords_dict, rbf_edges, edge_contrasts, 'RBF')
    

def exhaustive_method_ari_check(gs_configs, true_gs):
    '''
    Checks that the predicted groundstate from the brute force method are the true track GS labels.
    Effectively checks that λ is a good value to use for the analysis.
    If there are ground state tracks that are not 0...01...1 or 1...10...0 then the values of λ is bad.
    '''
    for gs_config in gs_configs:
        ari = ari_check(true_gs, np.array([gs_config]))[0]
        print(f'ARI check from exhaustive GS search: {gs_config} -> {ari:.4f}')

def exhaustive_ising_method(true_gs : float, 
                            lambda_bal : float,
                            similarity_matrix : np.ndarray[float],
                            similarity_type : str) -> tuple[float, float]:
    '''
    Brute force ising landscape method.
    
    Computes gs energy for every possible configuration (way of clustering hits). Then finds minimum which encodes the configuration
    we want for other methods we want to use later. 
    This configuration is for example N=3 (6 hits in total) = 000111 or 111000.
    '''
    
    energies, gs_energy, gs_configs = ising_optimisation(len(similarity_matrix), lambda_bal, similarity_matrix)
    #energy_landscape(len(similarity_matrix), lambda_bal, energies, similarity_type)
    
    exhaustive_method_ari_check(gs_configs, true_gs)
    
    return gs_energy



def generate_toyproblem_params(hits : int, 
                               lambda_bal : float, 
                               similarity_type : str,
                               graph_switch : bool):
    '''
    Generates:
    - the tracks and the truth labels.
    - the system simlarity matrices. 
    
    Executes the brute force ising energy search for the true groundstate.
    Shouldn't be run for large N.
    '''
    from global_params import track_seed
    np.random.seed(track_seed)                  #Fixed random seed. Same for every number of track hits
    x = np.linspace(0,1,hits)         #Positions of detectors
                
    track0, track0_truthlabels, track1, track1_truthlabels = toy_track_generation(hits, x, track_noise, intersection_allowed)
    true_gs = np.array(np.concatenate([track0_truthlabels, track1_truthlabels]))
    
    similarity_matrix = sim_matrices_calculation(x, track0, track1, similarity_type, graph_switch)
    true_gs_energy = exhaustive_ising_method(true_gs, lambda_bal, similarity_matrix, similarity_type)
    return similarity_matrix, true_gs, true_gs_energy
    
        