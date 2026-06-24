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
    
    while len(hits_to_assign) != 0:                     
        k = np.random.choice(list(hits_to_assign))                  #Randomly select a node from the hits we have yet to assign.
        
        mean_0 = np.mean(np.array([W[k][x] for x in cluster0]))       #Calculate means of the similarity matrix row k (excluding hits that aren't yet selected in the cluster).
        mean_1 = np.mean(np.array([W[k][x] for x in cluster1]))
        
        if mean_0 > mean_1:
            cluster0.add(k)
        else:
            cluster1.add(k)
        hits_to_assign.remove(k)
    
    greedy_end_time = time.time()
    return np.array(list(cluster0)), np.array(list(cluster1)), (greedy_end_time - greedy_start_time)


##################################################      SPECTRAL CLUSTERING ALGORITHM      ################################################## 


def spectral(W : np.ndarray[float]):

    spectral_start_time = time.time()

    clustering = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=1).fit_predict(W)

    spectral_end_time = time.time()   
    return clustering, (spectral_end_time - spectral_start_time)


##################################################      SIMULATED ANNEALING ALGORITHM      ################################################## 
    
def perturb_current_state(state):
    rand_ind = np.random.randint(len(state))
    new_state = state.copy()
        
    new_state[rand_ind] ^= 1
    return new_state
        
def sim_annealing(init, W, lambda_bal):
    sim_anneal_start_time = time.time()
    current_state = init
    current_energy = ising_energy(current_state, W, lambda_bal)
    T = 2.0
        
    best_state = current_state.copy()
    best_energy = current_energy
    
    state_history = np.array([current_state])
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
                
            
        state_history = np.vstack([state_history, current_state])
        energy_history = np.append(energy_history, current_energy)
            
        T *= 0.99
    
    sim_anneal_end_time = time.time()
            
    return best_state, best_energy, state_history, energy_history, (sim_anneal_end_time - sim_anneal_start_time)

##################################################      SA CONVERGENCE TRACE      ################################################## 

def plot_conv_trace():
    pass

##################################################      Handling functions      ################################################## 

def print_results(hit_coords, optimised_config, ari, time_elapsed, algorithm_type):
    
    print(f'{algorithm_type} algorithm returned configuration = {optimised_config}')
    print(f'ARI check returns a value of {ari}')
    print(f'Time to run was {time_elapsed}')
    plot_optimised_toytracks(hit_coords, optimised_config, algorithm_type)
    print('\n')



def greedy_results(W, true_groundstate,lambda_bal):
    greedy_cluster0, greedy_cluster1, greedy_time_elapsed = greedy_alg(W)
    greedy_config = np.zeros_like(np.concatenate([greedy_cluster0, greedy_cluster1]))
    
    for node0, node1 in zip(greedy_cluster0, greedy_cluster1):
        greedy_config[node0] = 0
        greedy_config[node1] = 1
        
    greedy_ari = ARI_check(true_groundstate, np.array([greedy_config]))
    greedy_energy = ising_energy(greedy_config, W, lambda_bal)
    
    return greedy_config, greedy_energy, greedy_ari, greedy_time_elapsed 



def spectral_clustering_results(W, true_groundstate, lambda_bal):
    spectral_config, spectral_time_elapsed = spectral(W)
    
    spectral_ari = ARI_check(true_groundstate, np.array([spectral_config]))
    spectral_energy = ising_energy(spectral_config, W, lambda_bal)
    
    return spectral_config, spectral_energy, spectral_ari, spectral_time_elapsed, 
    
    

def sim_annealing_results(W, true_groundstate, lambda_bal):
    init = np.random.randint(0,2, len(W))
    
    sa_track, sa_track_energy, state_history, energy_history, sa_time_elapsed = sim_annealing(init, W, lambda_bal)

    sa_ari = ARI_check(true_groundstate, np.array([sa_track]))   
    return sa_track, sa_track_energy, sa_ari, sa_time_elapsed, state_history, energy_history
