import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def true_toytracks(x, track0, track1, intersection_allowed):
  
    plt.scatter(x, track0, c='blue', s=40, marker='o')
    plt.scatter(x, track1, c='blue', s=40, marker='o')
    plt.xlim(-0.1, 1.1)
    plt.xticks(x)
    plt.title(f'Particle track plot with intersection = {intersection_allowed}')
    plt.title(f'Hit Coordinates')
    plt.grid(axis='x')
    plt.xlabel('Detector positions')
    plt.ylabel('y')
    plt.savefig(f'assets/plots/ClassicalPlots/TrueTracks_{len(np.concatenate([track0, track1]))}')
    plt.show()
    
##############################################################################################################################

def similaritymatrix_heatmap(sim_matrix,  matrix_type : str):
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


def energy_landscape(hits, lambda_bal, energies, similarity_type): 
    '''
    Energy landscape plots for chosen similarity matrix. 
    Two plots:
    1. 2^N states and their energies (all ordered by energy).
    2. 10 lowest energy states to view groundstate degeneracy.
    '''
    
    fig, ax = plt.subplots(2, 1, figsize=(10,6))  
    ax[0].plot(np.sort(energies), color='orange')
    ax[0].set_title('All state energies')
    
    ax[1].plot(np.sort(energies)[:10], color='orange')
    ax[1].set_title('Lowest 10 energy states')
    
    fig.suptitle(f'Bruteforce Ising Energy landscapes: N={hits}, λ={lambda_bal}, W={similarity_type}')
    fig.supylabel('Energy')
    fig.supxlabel('Rank')
    plt.savefig(f'assets/plots/ClassicalPlots/Ising_energy_landscapes_N{hits}_W{similarity_type}')
    plt.tight_layout()
    plt.show()
    