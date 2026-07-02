import numpy as np

import time
from sklearn.cluster import SpectralClustering
from plotting import conv_traces
from ising import ising_energy, ARI_check

##################################################      GREEDY ALGORITHM      ################################################## 
def get_mostdissimlar_hits(RBF_matrix):
    RBF_no_diag = RBF_matrix.copy()                                 
    np.fill_diagonal(RBF_no_diag, np.inf)                         #We don't want to include the diagonals so we force them, in this copy, to inf.

    i, j = np.unravel_index(np.argmin(RBF_no_diag), RBF_no_diag.shape)          #np.argmin finds the indices of the minimum. Since W is symmetric we only need one position.
    return i,j 


def greedy(W : np.ndarray[float], i, j):
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
    hits_to_assign = list(range(n))                        #Hits that we have yet to assign. 
    greedy_config = np.zeros_like(hits_to_assign)
    
    hits_to_assign.remove(i)
    hits_to_assign.remove(j)
    
    greedy_start_time = time.time()
    
    cluster0 = [i]
    cluster1 = [j]
    
    hits_to_assign.sort(key=lambda k: max(W[k, i], W[k, j]), reverse=True)
    
    for k in hits_to_assign:                    
        mean_0 = np.mean(np.array([W[k][x] for x in cluster0]))       #Calculate means of the similarity matrix row k (excluding hits that aren't yet selected in the cluster).
        mean_1 = np.mean(np.array([W[k][x] for x in cluster1]))
        if mean_0 > mean_1:
            cluster0.append(k)
        else:
            cluster1.append(k)
    
    greedy_end_time = time.time()
    
    for node in cluster0:
        greedy_config[node] = 0
        
    for node in cluster1:
        greedy_config[node] = 1
   
    return np.array(greedy_config), (greedy_end_time - greedy_start_time)


##################################################      SPECTRAL CLUSTERING ALGORITHM      ################################################## 


def spectral(W : np.ndarray[float]):
    '''
    Makes globally informed choices using graph Laplacian followed by eigen analysis.
    '''
    spectral_start_time = time.time()
    clustering = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=41).fit_predict(W)
    spectral_end_time = time.time()   
    return clustering, (spectral_end_time - spectral_start_time)


##################################################      SIMULATED ANNEALING ALGORITHM      ################################################## 
    
def perturb_current_state(state):
    '''
    Use XOR to choose a new state roughly close to the current state. 
    The energy change depends on this operation.
    '''
    rand_ind = np.random.randint(len(state))
    new_state = state.copy()
        
    new_state[rand_ind] ^= 1                        #Bitwise XOR. The (rand_ind)th element is XORed with 1 which always flips the element (0 to 1 and vice versa)
    return new_state

        
def sim_annealing(W, init, lambda_bal):
    '''
    Stochastic algorithm. Uses a cooling scheme to search through state space. Probability of accepting a new state decreases with T.
    Ideally the algorithm converges to the global minimum (true groundstate) but can converge to a local minimum and get stuck.
    '''
    sim_anneal_start_time = time.time()
    current_state = init
    current_energy = ising_energy(W, current_state, lambda_bal)
    number_of_steps = 1
    T = 5.0
        
    best_state = current_state.copy()
    best_energy = current_energy
    
    energy_history = np.array([current_energy])
        
    while T > 0.001:
            
        candidate_state = perturb_current_state(current_state)
        candidate_energy = ising_energy(W, candidate_state, lambda_bal)
            
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

def run_classical_algorithm(algorithm : str, similarity_params : tuple, i, j):
    if algorithm == 'Greedy':
        return greedy_results(*similarity_params, i, j)
    
    elif algorithm == 'Spectral Clustering':
        return spectral_results(*similarity_params)
    
    else:
        number_of_loops = 2
        best_sa_configs, best_sa_config_energies, best_sa_aris, sa_times_elapsed, sa_energy_histories, sa_number_of_steps, sa_conv_count = sim_annealing_results(*similarity_params, number_of_loops)
        sa_config, sa_rel_energy, sa_ari, sa_time_elapsed = find_optimised_sa_data(best_sa_configs, best_sa_config_energies, similarity_params[2], best_sa_aris, sa_times_elapsed)
       
        conv_traces(len(sa_config), sa_number_of_steps, sa_energy_histories)
        return sa_config, sa_rel_energy, sa_ari, sa_time_elapsed, (sa_conv_count / number_of_loops) 
    

def get_groundstate_energy(W, true_groundstate, lambda_bal):
    return ising_energy(W, true_groundstate, lambda_bal)



def greedy_results(W, true_groundstate, true_groundstate_energy, lambda_bal, i, j):
    optimised_config, time_elapsed = greedy(W, i, j)

    ari = ARI_check(true_groundstate, np.array([optimised_config]))
    energy = ising_energy(W, optimised_config, lambda_bal)
    rel_energy = abs((true_groundstate_energy - energy) / true_groundstate_energy)
    
    return optimised_config, rel_energy, ari, time_elapsed, None


def spectral_results(W, true_groundstate, true_groundstate_energy, lambda_bal):
    optimised_config, time_elapsed = spectral(W)
    
    ari = ARI_check(true_groundstate, np.array([optimised_config]))
    energy = ising_energy(W, optimised_config, lambda_bal)
    rel_energy = abs((true_groundstate_energy - energy) / true_groundstate_energy)
    
    return optimised_config, rel_energy, ari, time_elapsed, None



def sim_annealing_results(W, true_groundstate, true_groundstate_energy, lambda_bal, number_of_loops: int):

    energy_histories = []
    best_configs = []
    best_config_energies = []
    aris = []
    times = []
    steps = []
    convergence_counter = 0
    
    for _ in range(number_of_loops):
        init = np.random.randint(0,2, len(W))
        
        sa_config, sa_energy, energy_history, sa_time_elapsed, no_steps = sim_annealing(W, init, lambda_bal)
        sa_ari = ARI_check(true_groundstate, np.array([sa_config]))   
        
        best_configs.append(sa_config)
        energy_histories.append(energy_history)
        best_config_energies.append(sa_energy)
        aris.append(sa_ari)
        times.append(sa_time_elapsed)
        steps.append(no_steps)
        
        if np.isclose(sa_energy, true_groundstate_energy):
            convergence_counter += 1
        
    return best_configs, best_config_energies, aris, times, energy_histories, steps, convergence_counter


def find_optimised_sa_data(best_sa_configs, best_sa_config_energies, true_groundstate_energy, sa_aris, sa_times_elapsed):
    '''
    Using all the data we found from performing the sim_annealing algorithm number_of_loops times, we:
    1. Find the best configuration based on which has the minimum energy.
    2. Compute the relative error in this energy with the true_groundstate_energy
    
    If multiple loops do actually find the true groundstate, we pick the one which took the least time.
    3. Return all data: best configuration, best relative error, the configuration ari, the smallest time-to-run.'''
    
    optimum_energy = np.min(best_sa_config_energies)
    optimum_rel_energy = abs((true_groundstate_energy - optimum_energy) / true_groundstate_energy)
    
    optimum_indices = np.where(np.isclose(best_sa_config_energies, optimum_energy))[0]
    times = [sa_times_elapsed[i] for i in optimum_indices]
    
    optimum_index = np.where(np.isclose(sa_times_elapsed, min(times)))[0][0]
    
    return best_sa_configs[optimum_index], optimum_rel_energy, sa_aris[optimum_index], sa_times_elapsed[optimum_index]
            