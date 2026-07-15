import numpy as np
from track_generation import construct_toytracks
import plotting as plot
from similarity import get_KNN_matrix, get_RBF_matrix
import classical_benchmarks as cb
from simple_qaoa import grid, cobyla, qaoa_results
from ising import ising_optimisation

    

def toy_track_generation(track_hits : int, x : np.ndarray) -> tuple[list[float], list[int], list[float], list[int]]:
    
    intersection_allowed = False            #Boolean to control whether particles intersect
    sigma_noise = 1e-2                      #External noise 
    
    track0, track0_truthlabels, track1, track1_truthlabels = construct_toytracks(x, track_hits, sigma_noise, intersection_allowed)
    #plot.plot_true_toytracks(x, track0, track1, intersection_allowed)
    
    return track0, track0_truthlabels, track1, track1_truthlabels
    
    
    
def sim_matrices_calculation(x, track0, track1):
    nearneighb_n = 3                       #Number of nearest neighbours to consider in the KNN matrix
    
    hit_coords = np.column_stack([np.concatenate([x, x]),np.concatenate([track0, track1])])         #2D array of hit coordinates.                                                   
    
    
    KNN_matrix, nbrs = get_KNN_matrix(hit_coords, nearneighb_n)                      #nbrs only needed for graph visualisation.
    RBF_matrix = get_RBF_matrix(hit_coords)
    
    #hit_coords_dict = {i: tuple(hit_coords[i]) for i in range(number_of_hits)}          #Hit coordinates needed for plotting graph representations.
    #knn_G, knn_edges = plot.construct_KNN_graphrep(number_of_hits, hit_coords, nbrs)
    #rbf_G, rbf_edges, edge_contrasts = plot.construct_RBF_graphrep(number_of_hits, RBF_matrix)
    #plot.graphrep(knn_G, x, hit_coords_dict, knn_edges, None, 'KNN')
    #plot.graphrep(rbf_G, x, hit_coords_dict, rbf_edges, edge_contrasts, 'RBF')
    
    return KNN_matrix, RBF_matrix



def exhaustive_ising_method(RBF_matrix, KNN_matrix, lambda_bal):
    KNN_energies, KNN_groundstate_energy, _, RBF_energies, RBF_groundstate_energy, _ = ising_optimisation(len(RBF_matrix), lambda_bal, KNN_matrix, RBF_matrix)
    #plot.energy_landscape(lambda_bal, KNN_energies, RBF_energies)
    
    return KNN_groundstate_energy, RBF_groundstate_energy


    
def classical_scan(classical_algs, lambda_bal, hits):
    params = generate_toyproblem(hits, lambda_bal)
    i, j = cb.get_mostdissimlar_hits(params[0])        #Gets the most dissimilar hits for the greedy algorithm. Uses RBF matrix for both RBF and KNN options.
    classical_metrics = {alg : {'rel_error' :[],
                         'ari' : [],
                         'runtime' : [],
                         'conv_frac' : []} for alg in classical_algs}
    
    

    for alg in classical_algs:
        
        _, rel_energy_error, ari, runtime, convergence_fraction = cb.run_classical_algorithm(alg, params, lambda_bal, i, j)
        classical_metrics[alg]['rel_error'] = rel_energy_error
        classical_metrics[alg]['ari'] = ari[0]
        classical_metrics[alg]['runtime'] = runtime
        classical_metrics[alg]['conv_frac'] = convergence_fraction
        
    
    return classical_metrics


def depth_scan(fixed_hits, layers, lambda_bal, qaoa_optimisers, no_of_shots, seed_lim):
    
    params = generate_toyproblem(fixed_hits, lambda_bal)
    
    metric_means = {p  : {name : {'rel_error' :[],
                                'ari' : [],
                                'runtime' : [],
                                'gsp' : []} for name in qaoa_optimisers.keys()} for p in layers}
    
    
    metric_errors = {p  : {name : {'rel_error' :[],
                                'ari' : [],
                                'runtime' : [],
                                'gsp' : []} for name in qaoa_optimisers.keys()} for p in layers}
    
    
    '''
    Above dictionaries are arranged like this:
    Metric : [[Grid search p=1, Grid search p=2, ...], [COBYLA p=1, COBYLA p=2, ...]]
    Where for each p in the COBYLA list the mean (or std) is given for seed_lim samples. 
    '''
    
    for optimiser_name, optimiser in qaoa_optimisers.items():
        for p in layers:
            means, errors = qaoa_results(*params, lambda_bal, no_of_shots, p, seed_lim, optimiser)

            metric_means[p][optimiser_name]['rel_error'] = means[0]
            metric_means[p][optimiser_name]['ari'] = means[1]
            metric_means[p][optimiser_name]['runtime'] = means[2]
            metric_means[p][optimiser_name]['gsp'] = means[3]
            
            
            metric_errors[p][optimiser_name]['rel_error'] = errors[0]
            metric_errors[p][optimiser_name]['ari'] = errors[1]
            metric_errors[p][optimiser_name]['runtime'] = errors[2]
            metric_errors[p][optimiser_name]['gsp'] = errors[3]
            
    
    plot.depth_scan_metric_scatter(layers, metric_means, metric_errors)
        
        
        
def scale_scan(track_hits : np.ndarray[int], qaoa_optimisers : dict, no_of_shots : int, 
               fixed_layers : int, seed_lim : int, lambda_bal : float):
    raw_results = {name : [] for name in qaoa_optimisers.keys()}
    raw_errors = {name : [] for name in qaoa_optimisers.keys()}
    
    rel_results = {name : [] for name in qaoa_optimisers.keys()}
    rel_errors = {name : [] for name in qaoa_optimisers.keys()}
    
    for hits in track_hits:
         
        params = generate_toyproblem(hits, lambda_bal)
         
        for optimiser_name, optimiser in qaoa_optimisers.items():
            means, stds = qaoa_results(*params, lambda_bal, no_of_shots, fixed_layers, seed_lim, optimiser)
            
            baseline = 2 / (2**(2*hits))
            #The groundstate probability is the final entry of the means/std output arrays of the qaoa_results
            relative_gs_prob = means[-1] / baseline
            relative_error = stds[-1] / baseline
            
            raw_results[optimiser_name].append(means[-1])
            raw_errors[optimiser_name].append(stds[-1])
            
            rel_results[optimiser_name].append(relative_gs_prob)
            rel_errors[optimiser_name].append(relative_error)
            
    
    plot.scaling_scan(track_hits, raw_results, raw_errors,
                                  rel_results, rel_errors)
    
    
    
def quantum_scan(qaoa_optimisers : dict,  lambda_bal : float,  hits : int, no_of_shots : int, p : int, seed_lim : int):
    params = generate_toyproblem(hits, lambda_bal)
    
    quantum_metrics = {name: {'rel_error': None,
                            'ari': None,
                            'runtime': None,
                            'gsp': None}
                            for name in qaoa_optimisers.keys()}
    
    for optimiser_name, optimiser in qaoa_optimisers.items():
        means, errors = qaoa_results(*params, lambda_bal, no_of_shots, p, seed_lim, optimiser)
        for metric, mean, std in zip(quantum_metrics[optimiser_name].keys(), means, errors):
            quantum_metrics[optimiser_name][metric] = {'mean': mean,
                                                        'error': std}
    return quantum_metrics
   
   
   
def generate_toyproblem(hits : int, lambda_bal : float):
    np.random.seed(45)                  #Fixed random seed. Same for every number of track hits
    x = np.linspace(0,1,hits)         #Positions of detectors
                
    track0, track0_truthlabels, track1, track1_truthlabels = toy_track_generation(hits, x)
    KNN_matrix, RBF_matrix = sim_matrices_calculation(x, track0, track1)
    #Choose to use the RBF matrix over KNN as similarity measure.
    _, RBF_true_gs_energy = exhaustive_ising_method(RBF_matrix, KNN_matrix, lambda_bal)
                
    true_groundstate = np.array(np.concatenate([track0_truthlabels, track1_truthlabels]))
    return RBF_matrix, true_groundstate, RBF_true_gs_energy
    
    
    
def main():
    option = 'scale'                  #This is the identifier for which 'task' we want to do.
    
    no_of_shots = 8192                 #Number of measurements the quantum simulator will make of the circuit (all independent).
    seed_lim = 8                  #Number of runs of the QAOA to calculate means and errors.
    lambda_bal = 0.75                 #Lambda_balance parameter values to be used in the Hamiltonian. Modelled as a constant.
    classical_algs = ['Greedy', 'Spectral Clustering', 'Simulated Annealing']
    qaoa_optimisers = {'Grid' : grid, 'COBYLA' : cobyla}


    
    if option == 'depth':
        #Depth Scan fixes N varies p.
        fixed_hits = 4
        layers = np.arange(1, 3)
        depth_scan(fixed_hits, layers, lambda_bal, qaoa_optimisers, no_of_shots, seed_lim)        
            
    elif option == 'scale':
        #Scaling Scan fixes p varies N.
        hits_array = np.array([3,4,5,6])
        fixed_layers = 1
        scale_scan(hits_array, qaoa_optimisers, no_of_shots, fixed_layers, seed_lim, lambda_bal)
        
    elif option == 'class':
        #For a fixed N and p, compare all algorithms in one table.
        hits = 6
        layers = 1
        
        classical_results = classical_scan(classical_algs, lambda_bal, hits)
        plot.print_benchmark_table(hits, classical_results)
        
        quantum_results = quantum_scan(qaoa_optimisers, lambda_bal, hits, no_of_shots, layers, seed_lim)
        
        plot.print_quantum_table(hits, quantum_results)
        
        
if __name__ == "__main__":
    main()