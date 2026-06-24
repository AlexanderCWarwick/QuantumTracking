import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from itertools import product

#######################     Brute force Ising minimisation        ####################### 

def ising_energy(bitstring : np.ndarray[int], W : np.ndarray[np.float64], lambda_bal : float):
    '''
    Ising objective function. The function rewards like spins and discourages extreme configurations with a penalty term e.g. 111111111111.
    Note that the approach here is to use brute force since there is only 12 hits. If n = 20 then brute force would be VERY inefficient.
    '''
    isingstring = (2 * bitstring) - 1                   #Convert the bitstring (configuration) into a ising spin configuration.
    n = len(W)
    H = 0

    for i in range(n):
        for j in range(i+1, n):                                         #j > i in Hamiltonian. Avoids double counting the interacting spins.
            H -= W[i][j] * isingstring[i] * isingstring[j]              #Rewarding term for like spins
            
    H += ising_penalty_term(lambda_bal, isingstring)                    #Penalty term penalising big clustering.
    
    return H


def ising_penalty_term(lambda_bal, isingstring):
    '''
    Type of penalty term to be used in the Ising energy. Should be non-negative.
    Types: 
    1. lambda_bal * (np.sum(isingstring))**2  
    
    2. lambda_bal * (mod(np.sum(isingstring)))
    3. lambda_bal * (np.sum(isingstring))**4
    4. lambda_bal * (sum_(i<j)(isingstring))**2
    '''
    
    return lambda_bal * (np.sum(isingstring))**2



def get_ising_energies(W, lambda_bal, config_space):
    '''
    Returns the 'energy landscape' of the the chosen hamiltonian. 
    '''
    energies = []
    for bitstring in config_space:
        energy = ising_energy(bitstring, W, lambda_bal)
        energies.append(energy)

    return np.array(energies)



def get_groundstate(energies, groundstate_energy, config_space):
    '''
    Returns where the ground state configuration is (the indices) using the energies array.
    Exchange degeneracy means energy landscape is symmetric. Hence there are at least two ground states.
    '''
    groundstates_indices = np.where(np.isclose(energies,groundstate_energy))[0]
    groundstate_configs = np.array([config_space[i] for i in groundstates_indices])

    return groundstate_configs



def KNN_RBF_opt(sim_matrix, lambda_bal, config_space):
    '''
    Primary function of the Ising optimisation block. Returns:
    1. The energy landscape 
    2. The calculated ground state energy
    3. The ground state configurations
    '''
    
    energies = get_ising_energies(sim_matrix, lambda_bal, config_space)
    groundstate_energy = min(energies)  
    
    return energies, groundstate_energy, get_groundstate(energies, groundstate_energy, config_space)     


            
def ising_optimisation(number_of_hits, lambda_bal, KNN_matrix, RBF_matrix):
    
    binary_config_space = np.array(list(product([0,1], repeat=number_of_hits)))             #List of all 2^12 possible BINARY label configurations. 
        
    KNN_energies, KNN_groundstate_energy, KNN_groundstate_binary_configs = KNN_RBF_opt(KNN_matrix, lambda_bal, binary_config_space)
    RBF_energies, RBF_groundstate_energy, RBF_groundstate_binary_configs = KNN_RBF_opt(RBF_matrix, lambda_bal, binary_config_space)
            
        
    return KNN_energies, KNN_groundstate_energy, KNN_groundstate_binary_configs, RBF_energies, RBF_groundstate_energy, RBF_groundstate_binary_configs
        

#######################     ARI calculation and check       ####################### 

'''
def ARI_check_ising(true_groundstate, KNN_groundstate_binary_configs, RBF_groundstate_binary_configs):
    Adjusted random score measures randomness of the cluster labels. It compares the computed groundstate and the true answer
    and returns: 
    ARI = 1 - Perfect clustering (what we are aiming for)
    0 < ARI < 1 - Random clustering
    ARI < 0 - Something has gone wrong
    
    Test the optimised groundstates against ONLY ONE of the true groundstate tracks, here called true_track.
    Hence double loop (ising_simmatrix_optimisation())
    KNN_aris = []
    RBF_aris = []
        
    for KNN_state, RBF_state in zip(RBF_groundstate_binary_configs, RBF_groundstate_binary_configs):
        KNN_aris.append(adjusted_rand_score(true_groundstate, KNN_state))
        RBF_aris.append(adjusted_rand_score(true_groundstate, RBF_state))
        
    KNN_ind = np.argmax(np.array(KNN_aris))
    RBF_ind = np.argmax(np.array(RBF_aris))
        
    print(f'Best KNN clustering is: {KNN_groundstate_binary_configs[KNN_ind]} with ARI = {KNN_aris[KNN_ind]}')
    print(f'Best RBF clustering is: {RBF_groundstate_binary_configs[RBF_ind]} with ARI = {RBF_aris[RBF_ind]}')
'''    

def ARI_check(true_groundstate, optimised_tracks, optimisation_method : str):
    aris = []
    
    for track in optimised_tracks:
        aris.append(adjusted_rand_score(true_groundstate, track))
        
    ind = np.argmax(np.array(aris))
    print(f'Best clustering using {optimisation_method} is: {optimised_tracks[ind]} with ARI = {aris[ind]}')
    
    return np.array(aris)
