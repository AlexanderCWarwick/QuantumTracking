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
    
    fig.suptitle(f'Energy landscapes for lambda={lambda_bal}')
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
    plt.savefig(f'plots/ClassicalPlots/optimised_benchmark_clusters_{len(hit_coords)}_hits.png')
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
    plt.savefig(f'plots/ClassicalPlots/{reps}_Convergence_Traces_{N}_hits.png')
    plt.show()

    
    
def print_benchmark_table(track_hit_array, algorithm_types, benchmark_aris, benchmark_times, relative_benchmark_energies, conv_fractions):
    header = (f'{'Hits':<8}'
        f'{'Algorithm':<25}'
        f'{'ARI':<12}'
        f'{'Time (s)':<15}'
        f'{'Relative Energy Error':<25}'
        f'{'Convergence Fraction':<15}')

    print(header)

    for i, hits in enumerate(track_hit_array):
        print('-' * len(header))
        for j, algorithm in enumerate(algorithm_types):
            if algorithm == 'Simulated Annealing' or j == 2:
                print(f'{2*hits:<8}'
                    f'{algorithm:<25}'
                    f'{benchmark_aris[i][j][0]:<12.4f}'
                    f'{benchmark_times[i][j]:<15.4f}'
                    f'{relative_benchmark_energies[i][j]:<25.4f}'
                    f'{conv_fractions[i]:<15.2f}')
                
            else:
                print(f'{2*hits:<8}'
                    f'{algorithm:<25}'
                    f'{benchmark_aris[i][j][0]:<12.4f}'
                    f'{benchmark_times[i][j]:<15.4f}'
                    f'{relative_benchmark_energies[i][j]:<25.4f}') 
            

###############################################################################################################################

def plot_energy_hist(energies, true_groundstate_energy):
    plt.figure()
    plt.hist(energies, bins=30)
    plt.axvline(true_groundstate_energy, color='red', linestyle='--', label='Exact ground state')
    plt.xlabel('Ising energy')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    