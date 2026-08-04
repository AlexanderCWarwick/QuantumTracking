import numpy as np
from itertools import product
from classical_algs.ising import ising_energy

def get_ising_energies(W, lambda_bal, config_space):
    '''
    Returns the 'energy landscape' of the the chosen hamiltonian. 
    '''
    energies = []
    for bitstring in config_space:
        energy = ising_energy(W, bitstring, lambda_bal)
        energies.append(energy)

    return np.array(energies)



def get_groundstates(energies : np.ndarray,  groundstate_energy : np.ndarray,  config_space):
    '''
    Returns where the ground state configuration is (the indices) using the energies array.
    Exchange degeneracy means energy landscape is symmetric. Hence there are at least two ground states.
    '''
    groundstates_indices = np.where(np.isclose(energies, groundstate_energy))[0]
    groundstate_configs = np.array([config_space[i] for i in groundstates_indices])
    return groundstate_configs



def KNN_RBF_opt(sim_matrix, lambda_bal,  config_space : np.ndarray):
    '''
    Primary function of the Ising optimisation block. Returns:
    1. The energy landscape 
    2. The calculated ground state energy
    3. The ground state configurations
    '''
    
    energies = get_ising_energies(sim_matrix, lambda_bal, config_space)
    groundstate_energy = min(energies)  
    
    return energies, groundstate_energy, get_groundstates(energies, groundstate_energy, config_space)     


            
def ising_optimisation(number_of_hits : int,  lambda_bal : float,  KNN_matrix : np.ndarray,  RBF_matrix : np.ndarray):
    
    binary_config_space = np.array(list(product([0,1], repeat=number_of_hits)))             #List of all 2^(N) possible BINARY label configurations. 
        
    KNN_energies, KNN_groundstate_energy, KNN_groundstate_binary_configs = KNN_RBF_opt(KNN_matrix, lambda_bal, binary_config_space)
    RBF_energies, RBF_groundstate_energy, RBF_groundstate_binary_configs = KNN_RBF_opt(RBF_matrix, lambda_bal, binary_config_space)
            
        
    return KNN_energies, KNN_groundstate_energy, KNN_groundstate_binary_configs, RBF_energies, RBF_groundstate_energy, RBF_groundstate_binary_configs
