import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_true_toytracks(x, track0, track1, intersection_allowed):
  
    plt.scatter(x, track0, c='blue', s=40, marker='o')
    plt.scatter(x, track1, c='red', s=40, marker='o')
    plt.xlim(-0.1, 1.1)
    plt.title(f'Particle track plot with intersection = {intersection_allowed}')
    plt.grid(axis='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'plots/ClassicalPlots/TrueTracks_{len(np.concatenate([track0, track1]))}')
    plt.show()
    
##############################################################################################################################

def plot_similaritymatrix_heatmap(sim_matrix,  matrix_type : str):
    '''
    Plot heat map representation of any input similarity matrix. The more correlated hits i and j are, the brighter the (ij)th 
    coordinate in the heat map.
    '''
    plt.imshow(sim_matrix, cmap='viridis')
    plt.title(f'{matrix_type} Heat map')
    plt.colorbar()
    plt.xlabel('Hit index')
     
##############################################################################################################################  

def construct_RBF_graphrep(number_of_hits,  RBF_matrix : np.ndarray[np.float64]):
    '''
    Create nodes and edges of RBF Graph representation. Edges are weighted and this is shwon via contrast (labels 
    would be too cluttered). A higher contrast means a smaller similarity.
    '''
    H = nx.Graph()
    rbf_edge_weights = []
    
    for i in range(number_of_hits):                         
        for j in range(i+1, number_of_hits):                       #Similarity matrix must be symmetric hence loop over the upper triangle of the matrix.
            rbf_edge_weight = RBF_matrix[i][j]
            
            H.add_edge(i, j, weight=rbf_edge_weight)               #Add weighted edge to graph instance H.
            rbf_edge_weights.append(rbf_edge_weight)                
            
    rbf_edge_weights = np.asarray(rbf_edge_weights)                #rbf_edge_weights is the ordered list of RBF matrix weights.
    
    edge_contrasts = get_edge_contrasts(rbf_edge_weights)          #Convert the RBF weights into contrasts for graph edges. Only for visualisation.
    
    return H, H.edges(), edge_contrasts
    
    
def get_edge_contrasts(rbf_edge_weights : np.ndarray[np.float64])  ->  np.ndarray[np.float64]:
    '''
    Contrast parameter alpha must be between 0 and 1. Formula is a linear normalisation.
    '''
    return 0.05 + 0.95 * (rbf_edge_weights - rbf_edge_weights.min()) / (rbf_edge_weights.max() - rbf_edge_weights.min())



def construct_KNN_graphrep(number_of_hits, hit_coords,  nbrs):
    '''
    KNN similarity is discrete so no contrast. Function obtains 2D array with each hits k nearest neighbours. 
    keighbors command returns indices including the hit itself. Hence the first index (closest, being the hit itself)
    is cut off from the array.
    '''
    H = nx.Graph()
    
    _, indices = nbrs.kneighbors(hit_coords)                #Can ignore distances here since KNN matrix is a discrete metric.
    indices = indices[:,1:]                                 #Slicing first column to remove self-similarity nodes.
    
    for i in range(number_of_hits):
        for j in indices[i]:
            H.add_edge(i,j)
     
    return H, H.edges()
    
  
def graphrep(H, x, hit_coords_dict, edges, edge_contrasts, matrix_type):
    '''
    Given the hit_coords dictionary, the edges and edge contrasts (for RBF matrix), plot Graph.'''
    
    
    nx.draw_networkx_edges(H, hit_coords_dict, edgelist=edges, alpha=edge_contrasts, edge_color='black')        #Draw  edges
    nx.draw_networkx_nodes(H, hit_coords_dict, node_size=200)                                                   #Draw nodes
    nx.draw_networkx_labels(H, hit_coords_dict)                                                                 #Draw hit labels.
      
    for detector_x in x:                                        #Shows positions of the detectors.
        plt.axvline(detector_x, linestyle='--', alpha=0.12)
    
    plt.title(f'{matrix_type} Graph Representation')
    plt.show()
    
    


##############################################################################################################################


def energy_landscape(lambda_bal, KNN_energies, RBF_energies): 
    
    fig, ax = plt.subplots(2, 2, figsize=(10,6))  
    ax[0,0].plot(np.sort(KNN_energies), color='orange')
    ax[0,0].set_title('KNN all states')
    
    ax[0,1].plot(np.sort(RBF_energies), color='red')
    ax[0,1].set_title('RBF all states')
    
    ax[1,0].plot(np.sort(KNN_energies)[:10], color='orange')
    ax[1,0].set_title('KNN lowest 10 energy states')
    ax[1,0].set_xlabel('Rank')
    
    ax[1,1].plot(np.sort(RBF_energies)[:10], color='red')
    ax[1,1].set_title('RBF lowest 10 energy states')
    ax[1,1].set_xlabel('Rank')
    
    fig.suptitle(f'Bruteforce Ising Energy landscapes for lambda={lambda_bal}')
    fig.supylabel('Energy')
    plt.tight_layout()
    plt.show()
    

##############################################################################################################################
def optimised_benchmark_toytracks(hit_coords, optimised_labels, algorithm_type : np.ndarray[str]):
    fig, ax = plt.subplots(1,len(algorithm_type), figsize=(13,8))
    
    for i, algorithm in enumerate(algorithm_type):
        ax[i].scatter(hit_coords[:, 0], hit_coords[:, 1], c=optimised_labels[i], cmap='bwr')
        ax[i].set_title(f'{algorithm}')
        
    fig.suptitle(f'Optimised Clusterings for Different Benchmark Algorithms: N = {len(hit_coords)}')
    fig.supylabel('y')
    fig.supxlabel('x')
    #plt.savefig(f'plots/ClassicalPlots/optimised_benchmark_clusters_{len(hit_coords)}_hits.png')
    plt.show()
    
    

def conv_traces(N: int, steps : np.ndarray, energy_histories : np.ndarray):
    reps = len(energy_histories)
    fig, ax = plt.subplots(reps, 1, figsize=(12,12))
    if reps > 1:
        for i in range(reps):
            ax[i].plot(np.arange(steps[i]), energy_histories[i], color='red')
            
    else:
        ax.plot(np.arange(steps[0]), energy_histories[0], color='red')
            
    fig.suptitle('Convergence Traces with Different Random Starting Points in SA')
    fig.supylabel('Energy')
    fig.supxlabel('Step')
    #plt.savefig(f'plots/ClassicalPlots/{reps}_Convergence_Traces_{N}_hits.png')
    plt.show()

    
    
def print_benchmark_table(hits, classical_results : dict[dict]):
    
    header = (f'{'Hits':<8}'
        f'{'Algorithm':<25}'
        f'{'ARI':<12}'
        f'{'Full Time (s)':<20}'
        f'{'Relative Energy Error':<25}'
        f'{'Convergence Fraction':<22}')
    
    print(header)
    print('-' * len(header))
    
    
    for alg, metrics in classical_results.items():
        success = classical_results[alg]['conv_frac']
        
        if success is None:
            success = "-"
        else:
            success = f"{success:.2f}"
            
        print(f"{2*hits:<8}"
            f"{alg:<25}"
            f"{metrics['ari']:<12.4f}"
            f"{metrics['runtime']:<20.4f}"
            f"{metrics['rel_error']:<25.4f}"
            f"{success:<22}")
    
    print('\n')
        
        
def print_quantum_table(hits : int, quantum_results : dict):
    def format_metric(metric):
        return f'{metric['mean']:.4f} \u00b1 {metric['error']:.4f}'
    
    header = (f'{'Hits':<8}'
        f'{'QAOA Optimiser':<20}'
        f'{'ARI':<20}'
        f'{'Time (s)':<20}'
        f'{'Relative Energy Error':<30}'
        f'{'GS Prob':<20}')
    
    print(header)
    print('-' * len(header))
    
    for optimiser_name, metrics in quantum_results.items():
        print(f'{2*hits:<8}'
            f'{optimiser_name:<20}'
            f'{format_metric(metrics['ari']):<20}'
            f'{format_metric(metrics['runtime']):<20}'
            f'{format_metric(metrics['rel_error']):<30}'
            f'{format_metric(metrics['gsp']):<20}')

    
def scaling_scan(track_hits : np.ndarray[int],  raw_results : dict,  raw_errors : dict,
                                                rel_results : dict, rel_errors : dict):
    
    fig, ax = plt.subplots(2, figsize=(7,7))
    
    for name in raw_results.keys():
        ax[0].errorbar(track_hits, raw_results[name], yerr=raw_errors[name], fmt='-o', capsize=3, label=name)
        ax[1].errorbar(track_hits, rel_results[name], yerr=rel_errors[name], fmt='-o', capsize=3, label=name)
        
    ax[0].set_xticks(track_hits)
    ax[1].set_xticks(track_hits)
    
    ax[0].set_title('Raw GSP')
    ax[1].set_title('Relative GSP')
    
    fig.supxlabel('Number of hits, N')
    fig.supylabel('Groundstate Probability')
    fig.suptitle('GS Probability Dependency on N')
    ax[0].legend()
    ax[1].legend()
    plt.show()
    
    
###############################################################################################################################

def plot_energy_hist(energies, true_groundstate_energy):
    plt.figure()
    plt.hist(energies, bins=30)
    plt.axvline(true_groundstate_energy, color='red', linestyle='--', label='Exact ground state')
    plt.xlabel('Ising energy')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    
    
    
def depth_scan_metric_scatter(layers, metric : np.ndarray[float, float], metric_std, metric_name):
    fig, ax = plt.subplots(2,1,figsize=(8,5))
    
    grid_metric = metric[0]
    grid_metric_std = metric_std[0]
    
    cobyla_metric = metric[1]
    cobyla_metric_std = metric_std[1]
    
    ax[0].errorbar(layers, grid_metric, yerr=grid_metric_std, fmt='o-', capsize=2,  alpha=0.7, elinewidth=1)
    ax[0].set_title(f'Grid optimiser for {metric_name}')
    ax[0].set_xticks(layers)
    
    ax[1].errorbar(layers, cobyla_metric, yerr=cobyla_metric_std, fmt='o-', capsize=2,  alpha=0.5, elinewidth=0.3)
    ax[1].set_title(f'COBYLA optimiser for {metric_name}')
    ax[1].set_xticks(layers)
    
    fig.suptitle('Side-by-side Comparison of Grid and COBYLA Performance')
    fig.supylabel(f'{metric_name}')
    fig.supxlabel('Number of layers')
    plt.show()