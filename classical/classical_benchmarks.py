import numpy as np

import time
from sklearn.cluster import SpectralClustering
from plotting.plotting import conv_traces
from classical.ising import ising_energy, ARI_check

##################################################      GREEDY ALGORITHM      ################################################## 
def get_mostdissimlar_hits(RBF_matrix):
    RBF_no_diag = RBF_matrix.copy()                                 
    np.fill_diagonal(RBF_no_diag, np.inf)                         #We don't want to include the diagonals so we force them, in this copy, to inf.

    i, j = np.unravel_index(np.argmin(RBF_no_diag), RBF_no_diag.shape)          #np.argmin finds the indices of the minimum. Since W is symmetric we only need one position.
    return i,j 


def greedy(W : np.ndarray[float]):
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
    
    i = np.random.randint(0, n/2)
    j = np.random.randint(n/2, n)
    
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
    Spectral Clustering makes globally informed choices using graph Laplacian followed by eigen analysis.
    '''
    #First clustering is a 'warm-up' call. Otherwie full runtime is order 2 seconds (longer than sim ann) which shouln't be the case.
    clustering = SpectralClustering(n_clusters=2, affinity='precomputed').fit_predict(W)
    
    spectral_start_time = time.time()
    clustering = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=3).fit_predict(W)
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

def metric_stats(metrics_dict : dict) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    '''
    Input: metrics_dict contains lists of (respective algorithm)_loop values for each metric.
    Compute means and errors for each different metric given
    Output: Two seperate arrays for means and errors for each metric. The ordering is kept the same as the dictionary.
    relative energy -> ari -> runtime -> convergence fraction
    '''
    
    metrics = metrics_dict.values()
    means = [np.mean(metric_values) for metric_values in metrics]
    std_metrics = [np.std(metric_values) for metric_values in metrics]
    
    return np.array(means), np.array(std_metrics)
    

def greedy_results(W, true_groundstate, true_groundstate_energy, lambda_bal,  loops : int) -> tuple[np.ndarray[np.float64, np.float64]]:
    metrics = {'rel_error' : [],
               'ari' : [],
               'runtime' : [],
               'conv_frac' : [0 for _ in range(loops)]}
    
    for _ in range(loops):
        optimised_config, runtime = greedy(W)
        energy = ising_energy(W, optimised_config, lambda_bal)
        rel_energy = abs((true_groundstate_energy - energy) / true_groundstate_energy)
        
        metrics['ari'].append(ARI_check(true_groundstate, np.array([optimised_config])))
        metrics['rel_error'].append(rel_energy)
        metrics['runtime'].append(runtime)
            
    return metric_stats(metrics)



def spectral_results(W, true_groundstate, true_groundstate_energy, lambda_bal, loops : int) -> tuple[np.ndarray[np.float64, np.float64]]:
    metrics = {'rel_error' : [],
               'ari' : [],
               'runtime' : [],
               'conv_frac' : [0 for _ in range(loops)]}
    
    for _ in range(loops):
        optimised_config, runtime = spectral(W)
        energy = ising_energy(W, optimised_config, lambda_bal)
        rel_energy = abs((true_groundstate_energy - energy) / true_groundstate_energy)
        
        metrics['ari'].append(ARI_check(true_groundstate, np.array([optimised_config])))
        metrics['rel_error'].append(rel_energy)
        metrics['runtime'].append(runtime)
            
    return metric_stats(metrics)



def sim_annealing_results(W, true_gs, true_gs_energy, lambda_bal, loops : int) -> tuple[np.ndarray[np.float64, np.float64]]:
    convergence_counter = 0
    energy_histories = []
    steps = []
    
    metrics = {'rel_error' : [],
               'ari' : [],
               'runtime' : [],
               'conv_frac' :[]}
    
    for _ in range(loops):
        init = np.random.randint(0,2, len(W))
        
        sa_config, sa_energy, energy_history, sa_time_elapsed, no_steps = sim_annealing(W, init, lambda_bal)
          
        energy_histories.append(energy_history)
        steps.append(no_steps)
        
        if np.isclose(sa_energy, true_gs_energy):
            convergence_counter += 1
            
        metrics['rel_error'].append(abs(sa_energy - true_gs_energy) / true_gs_energy)
        metrics['ari'].append(ARI_check(true_gs, np.array([sa_config])))
        metrics['runtime'].append(sa_time_elapsed)
    metrics['conv_frac'].append(convergence_counter / loops)
    
    #conv_traces(len(sa_config), steps, energy_histories)
    
    return metric_stats(metrics)
            