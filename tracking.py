import numpy as np
from track_generation import construct_toytracks
from plotting import plot_toytracks, plot_energy_landscape
from similarity import get_KNN_matrix, get_RBF_matrix
from ising import ising_optimisation, ARI_check

np.random.seed(41)

#######################################################     Main workflow     #######################################################

def main(): 
    '''
    Problem: Two particles pass through a detector each leaving N 'hits'. Using these 'hits' as coordinate
    locations of the particles, can we resolve the two tracks from each other and hence determine each particle's
    trajectory?
        
    Method:
    Two toy tracks (with noise) are randomly generated in 2D plane. 
    From the tracks obtain two types of similarity matrices:
    - RBF (radial basis function)
    - KNN (k nearest neighbours)
    - wKNN (weighted k nearest neighbours)
    Matrix elements measure the likelyhood that one hit is compatible to another, i.e. if they are the same particle.
    Obtain plots of these matrices as heatmaps.
    Obtain Graph representations of these similarity matrices.
    '''
    track_hits = 6                          #Number of detectors
    x = np.linspace(0,1,track_hits)         #Positions of detectors
    intersection_allowed = False            #Boolean to control whether particles intersect
    sigma_noise = 1e-2                      #External noise 
    nearneighb_n = 3                       #Number of nearest neighbours to consider in the KNN matrix
    
    track0, track0_truthlabels, track1, track1_truthlabels = construct_toytracks(x, track_hits, sigma_noise, intersection_allowed)
    plot_toytracks(x, track0, track1, sigma_noise, intersection_allowed)
    
    hit_coords = np.column_stack([np.concatenate([x, x]),np.concatenate([track0, track1])])         #2D array of hit coordinates.
    number_of_hits = len(hit_coords)
    hit_coords_dict = {i: tuple(hit_coords[i]) for i in range(number_of_hits)}
        
    KNN_matrix, nbrs = get_KNN_matrix(x, hit_coords, nearneighb_n)
    RBF_matrix = get_RBF_matrix(x, hit_coords)
    
#########################################################################################################################################################


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
    '''
    lambda_bal = 1.0                 #Lambda_balance parameter values to be used in the Hamiltonian. Modelled as a constant.
    
    true_groundstates = np.array([np.concatenate([track0_truthlabels, track1_truthlabels]), np.concatenate([track1_truthlabels, track0_truthlabels])])
    
    KNN_energies, _, KNN_groundstate_binary_configs, RBF_energies, _, RBF_groundstate_binary_configs = ising_optimisation(number_of_hits, lambda_bal, KNN_matrix, RBF_matrix)
    plot_energy_landscape(lambda_bal, KNN_energies, RBF_energies)
    ARI_check(true_groundstates[0], KNN_groundstate_binary_configs, RBF_groundstate_binary_configs)

if __name__ == "__main__":
    main()