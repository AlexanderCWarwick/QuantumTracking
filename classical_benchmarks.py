import numpy as np

import time
from sklearn.cluster import SpectralClustering
from plotting import plot_optimised_toytracks
from ising import ising_energy, ARI_check

##################################################      GREEDY ALGORITHM      ################################################## 

def greedy_alg(W : np.ndarray[float]):
    '''
    Greedy algorithm optimisation approach. Greedy makes local (short-sighted) decisions. Given a Yes/No question, go with which every gives the most benefit 
    at time when choosing.
    Algorithm:
    1. Find the two most dissimilar hits according to similarity matrix and split them into clusters, 0 and 1.
    2. Randomly pick a hit not yet assigned a cluster.
    3. Compute avg of all compatible hits.
    4. If mean_0 > mean_1 then cluster 0 is favourable and the hit is assigned to cluster 0, and vice versa.
    5. Repeat for all hits.
    
    Sets are used for functionality over np arrays. 
    '''
    
    n = len(W)
    hits_to_assign = set(range(n))                        #Hits that we have yet to assign. 
    
    W_no_diag = W.copy()                                 
    np.fill_diagonal(W_no_diag, np.inf)                         #We don't want to include the diagonals so we force them, in this copy, to inf.

    greedy_start_time = time.time()
    i, j = np.unravel_index(np.argmin(W_no_diag), W_no_diag.shape)          #np.argmin finds the indices of the minimum. Since W is symmetric we only need one position.

    hits_to_assign.remove(i)
    hits_to_assign.remove(j)
    
    cluster0 = {i}
    cluster1 = {j}
    
    for k in hits_to_assign:                     
        mean_0 = np.mean(np.array([W[k][x] for x in cluster0]))       #Calculate means of the similarity matrix row k (excluding hits that aren't yet selected in the cluster).
        mean_1 = np.mean(np.array([W[k][x] for x in cluster1]))
        
        if mean_0 > mean_1:
            cluster0.add(k)
        else:
            cluster1.add(k)
    
    greedy_end_time = time.time()
    return np.array(list(cluster0)), np.array(list(cluster1)), (greedy_end_time - greedy_start_time)


##################################################      SPECTRAL CLUSTERING ALGORITHM      ################################################## 


def spectral(W : np.ndarray[float]):

    spectral_start_time = time.time()

    clustering = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=41).fit_predict(W)

    spectral_end_time = time.time()   
    return clustering, (spectral_end_time - spectral_start_time)


##################################################      SIMULATED ANNEALING ALGORITHM      ################################################## 
    
def perturb_current_state(state):
    rand_ind = np.random.randint(len(state))
    new_state = state.copy()
        
    new_state[rand_ind] ^= 1                        #Bitwise XOR. The (rand_ind)th element is XORed with 1 which always flips the element (0 to 1 and vice versa)
    return new_state

        
def sim_annealing(init, W, lambda_bal):
    sim_anneal_start_time = time.time()
    current_state = init
    current_energy = ising_energy(current_state, W, lambda_bal)
    number_of_steps = 1
    T = 5.0
        
    best_state = current_state.copy()
    best_energy = current_energy
    
    energy_history = np.array([current_energy])
        
    while T > 0.001:
            
        candidate_state = perturb_current_state(current_state)
        candidate_energy = ising_energy(candidate_state, W, lambda_bal)
            
        energy_change = candidate_energy - current_energy
        random_num = np.random.random()
            
        if energy_change < 0:
            current_state = candidate_state
            current_energy = candidate_energy    
            
        elif random_num < np.exp(-(energy_change) / T):
            current_state = candidate_state
            current_energy = candidate_energy
            
                
        if current_energy < best_energy:
            best_energy = current_energy
            best_state = current_state.copy()
                
            
        energy_history = np.append(energy_history, current_energy)
            
        T *= 0.999
        number_of_steps += 1
    
    sim_anneal_end_time = time.time()
            
    return best_state, best_energy, energy_history, (sim_anneal_end_time - sim_anneal_start_time), number_of_steps


##################################################      Handling functions      ################################################## 

def print_results(hit_coords, optimised_config, optimised_energy, ari, time_elapsed, algorithm_type):
    
    print(f'{algorithm_type} algorithm returned optimised configuration = {optimised_config}')
    print(f'Ising energy of this state is {optimised_energy}')
    print(f'ARI check returns a value of {ari}')
    print(f'Time to run was {time_elapsed}')
    plot_optimised_toytracks(hit_coords, optimised_config, algorithm_type)
    print('\n')



def greedy_results(W, true_groundstate,lambda_bal):
    greedy_cluster0, greedy_cluster1, greedy_time_elapsed = greedy_alg(W)
    greedy_config = np.zeros_like(np.concatenate([greedy_cluster0, greedy_cluster1]))
    
    for node in greedy_cluster0:
        greedy_config[node] = 0
        
    for node in greedy_cluster1:
        greedy_config[node] = 1
        
    greedy_ari = ARI_check(true_groundstate, np.array([greedy_config]))
    greedy_energy = ising_energy(greedy_config, W, lambda_bal)
    
    return greedy_config, greedy_energy, greedy_ari, greedy_time_elapsed 



def spectral_clustering_results(W, true_groundstate, lambda_bal):
    spectral_config, spectral_time_elapsed = spectral(W)
    
    spectral_ari = ARI_check(true_groundstate, np.array([spectral_config]))
    spectral_energy = ising_energy(spectral_config, W, lambda_bal)
    
    return spectral_config, spectral_energy, spectral_ari, spectral_time_elapsed 



def sim_annealing_results(W, true_groundstate, lambda_bal, number_of_loops):

    energy_histories = []
    best_configs = []
    best_configs_energies = []
    aris = []
    times = []
    steps = []
    
    for _ in range(number_of_loops):
        init = np.random.randint(0,2, len(W))
        
        sa_config, sa_energy, energy_history, sa_time_elapsed, no_steps = sim_annealing(init, W, lambda_bal)
        sa_ari = ARI_check(true_groundstate, np.array([sa_config]))   
        
        best_configs.append(sa_config)
        energy_histories.append(energy_history)
        best_configs_energies.append(sa_energy)
        aris.append(sa_ari)
        times.append(sa_time_elapsed)
        steps.append(no_steps)
        
    return best_configs, best_configs_energies, aris, times, energy_histories, steps


def find_optimised_sa_data(best_sa_configs, best_sa_config_energies, sa_aris, sa_times_elapsed):
    best_ari = np.max(sa_aris)
    best_index = np.where(sa_aris == best_ari)[0][0]           #Gets first instance (or only instance) of the best result using the ARIs.
    
    return best_sa_configs[best_index], best_sa_config_energies[best_index], best_ari, sa_times_elapsed[best_index]
            