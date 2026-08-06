import numpy as np
from plotting.plotting_baseprob import energy_landscape
from problem.similarity import get_KNN_matrix, get_RBF_matrix
from problem.track_generation import construct_toytracks
from classical_algs.brute_force import ising_optimisation
from classical_algs.ari import ari_check


def toy_track_generation(track_hits : int, x : np.ndarray) -> tuple[list[float], list[int], list[float], list[int]]:
    '''
    Make toy tracks. All construction takes place in the track_generation.py file. 
    First source of noise is introduced : hit signal noise.
    
    Intersecting particle tracks are far more complicated to deal with. Toggled by intersection_allowed boolean.
    '''
    
    intersection_allowed = False            #Control whether particles tracks intersect
    sigma_noise = 1e-2                      #External noise 
    
    track0, track0_truthlabels, track1, track1_truthlabels = construct_toytracks(x, track_hits, sigma_noise, intersection_allowed)
    #plot_base.true_toytracks(x, track0, track1, intersection_allowed)
    
    return track0, track0_truthlabels, track1, track1_truthlabels
    
    
    
def sim_matrices_calculation(x, track0, track1, similarity_type : str):
    '''
    Similarity matrices to be calculated are
    - KNN. Here nearneighb_n is k.
    - RBF. Continuous measure.
    
    Plotting functionality of the similarity matrices as heat maps and graph networks is included.
    '''
    
    nearneighb_n = 3                       #Number of nearest neighbours to consider in the KNN matrix
    hit_coords = np.column_stack([np.concatenate([x, x]),np.concatenate([track0, track1])])         #2D array of hit coordinates.                                                   
    if similarity_type == 'KNN':
        KNN_matrix, nbrs = get_KNN_matrix(hit_coords, nearneighb_n)             #nbrs only needed for graph visualisation.
        return KNN_matrix
    elif similarity_type == 'RBF':
        return get_RBF_matrix(hit_coords)
    
    #number_of_hits = len(RBF_matrix)
    #hit_coords_dict = {i: tuple(hit_coords[i]) for i in range(number_of_hits)}          #Hit coordinates needed for plotting graph representations.
    #knn_G, knn_edges = plot_base.construct_KNN_graphrep(number_of_hits, hit_coords, nbrs)
    #rbf_G, rbf_edges, edge_contrasts = plot_base.construct_RBF_graphrep(number_of_hits, RBF_matrix)
    #plot_base.graphrep(knn_G, x, hit_coords_dict, knn_edges, None, 'KNN')
    #plot_base.graphrep(rbf_G, x, hit_coords_dict, rbf_edges, edge_contrasts, 'RBF')
    

def exhaustive_method_ari_check(gs_configs, true_gs):
    for gs_config in gs_configs:
        ari = ari_check(true_gs, np.array([gs_config]))[0]
        print(f'ARI check from exhaustive KNN GS search: {gs_config} -> {ari:.4f}')
            

def exhaustive_ising_method(true_gs : float, 
                            lambda_bal : float,
                            similarity_matrix : np.ndarray[float]) -> tuple[float, float]:
    '''
    Brute force ising landscape method.
    
    Computes gs energy for every possible configuration (way of clustering hits). Then finds minimum which encodes the configuration
    we want for other methods we want to use later. 
    This configuration is for N=3 (6 hits in total) = 000111 or 111000.
    '''
    
    energies, gs_energy, gs_configs = ising_optimisation(len(similarity_matrix), lambda_bal, similarity_matrix)
    energy_landscape(lambda_bal, energies) 
    
    exhaustive_method_ari_check(gs_configs, true_gs)
    
    return gs_energy



def generate_toyproblem_params(hits : int, 
                               lambda_bal : float, 
                               similarity_type : str):
    '''
    Generates:
    - the tracks and the truth labels.
    - the system simlarity matrices. 
    
    Executes the brute force ising energy search for the true groundstate.
    Shouldn't be run for large N.
    '''
    
    np.random.seed(45)                  #Fixed random seed. Same for every number of track hits
    x = np.linspace(0,1,hits)         #Positions of detectors
                
    track0, track0_truthlabels, track1, track1_truthlabels = toy_track_generation(hits, x)
    true_gs = np.array(np.concatenate([track0_truthlabels, track1_truthlabels]))
    
    similarity_matrix = sim_matrices_calculation(x, track0, track1, similarity_type)
    true_gs_energy = exhaustive_ising_method(true_gs, lambda_bal, similarity_matrix)
    return similarity_matrix, true_gs, true_gs_energy
    
        