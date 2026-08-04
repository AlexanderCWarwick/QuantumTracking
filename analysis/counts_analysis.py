import numpy as np
from classical_algs.ari import ari_check

from classical_algs.ising import ising_energy
from plotting.qaoa_energy_plots import (plot_energy_hist) #optimiser_energy_trace, optimiser_result_energies, top_ten_states

def get_groundstate_prob(best_counts, true_groundstate, no_of_shots):
    '''
    Returns the sample probability of measuring the groundstate configuration.
    
    Using the endian corrected bitstring as a key, search through the best counts (the collection of no_of_shots) samples for the best
    beta and gamma parameters.
    '''
    
    gs1_counts = best_counts.get(''.join(true_groundstate[::-1].astype(str)), 0)            
    gs2_counts = best_counts.get(''.join((true_groundstate^1)[::-1].astype(str)), 0)
    return (gs1_counts + gs2_counts) / no_of_shots

        
def get_counts_data(best_counts, 
                    W, 
                    true_groundstate, 
                    true_groundstate_energy, 
                    lambda_bal, 
                    no_of_shots):
    '''
    Input: A sample of the pdf from a circuit, best_counts. In total no_of_shots independent measurements made.
    Output: ari, rel_energy and gsp. best_config is found from which configuration was the most sampled (highest frequency)
    '''
    
    best_config = max(best_counts, key=best_counts.get)             #This is the configuration with the highest measurement frequency.
    best_config = best_config[::-1]                                 #Qiskit endian correction. Reverses configuration order (not inverting)
    best_config = np.array(list(best_config), dtype=int)            #Convert string to numpy array of integers.
    groundstate_prob = get_groundstate_prob(best_counts, true_groundstate, no_of_shots)
    
    best_config_energy = ising_energy(W, best_config, lambda_bal)
    
    best_rel_energy = abs((best_config_energy - true_groundstate_energy) / true_groundstate_energy)
    best_ari =  ari_check(true_groundstate, np.array([best_config]))        
        
    return best_config, best_rel_energy, best_ari, groundstate_prob


def energy_data_plot(best_counts, W, lambda_bal, true_groundstate_energy):
    '''
    Instead of plotting a histogram of how many times each configuration was measured, e.g. '0010' : 102, '1001' : 20 etc, 
    we instead convert this into an energy distribution. 
    '''
    energies = []
    
    for config, count in best_counts.items():
        config = config[::-1]
        config = np.array(list(config), dtype=int)
        config_energy = ising_energy(W, config, lambda_bal)
        energies.extend([config_energy] * count)
        
    plot_energy_hist(energies, true_groundstate_energy)
    
    