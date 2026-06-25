import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_true_toytracks(x, track0, track1, sigma_noise, intersection_allowed):
  
    plt.scatter(x, track0, c='blue', s=40, marker='o')
    plt.scatter(x, track1, c='red', s=40, marker='o')
    plt.xlim(-0.1, 1.1)
    plt.title(f'Particle track plot with intersection = {intersection_allowed} and noise = {sigma_noise}')
    plt.grid(axis='x')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.savefig('plots/Toytracks.png')
    plt.show()
    
def plot_optimised_toytracks(hit_coords, optimised_labels, algorithm_type : str):
    
    plt.scatter(hit_coords[:, 0], hit_coords[:, 1], c=optimised_labels, cmap='bwr')
    plt.xlim(-0.1, 1.1)
    plt.title(f'Particle track optimisation using {algorithm_type}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(axis='x')
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
    #plt.savefig(f'plots/{matrix_type}_heat_map.png')
    plt.show()
     
##############################################################################################################################  

def construct_RBF_graphrep(x, number_of_hits,  hit_coords_dict : dict,  RBF_matrix : np.ndarray[np.float64]):
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
    plot_graphrep(H, x, hit_coords_dict, H.edges(), edge_contrasts, 'RBF')
    
    
    
def get_edge_contrasts(rbf_edge_weights : np.ndarray[np.float64])  ->  np.ndarray[np.float64]:
    '''
    Contrast parameter alpha must be between 0 and 1. Formula is a linear normalisation.
    '''
    return 0.05 + 0.95 * (rbf_edge_weights - rbf_edge_weights.min()) / (rbf_edge_weights.max() - rbf_edge_weights.min())



def construct_KNN_graphrep(x, number_of_hits, hit_coords, hit_coords_dict,  nbrs):
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
    
    plot_graphrep(H, x, hit_coords_dict, H.edges(), None, 'KNN')    #No edge_contrasts needed for KNN matrix.
  
  
def plot_graphrep(H, x, hit_coords_dict, edges, edge_contrasts, matrix_type):
    '''
    Given the hit_coords dictionary, the edges and edge contrasts (for RBF matrix), plot Graph.'''
    
    
    nx.draw_networkx_edges(H, hit_coords_dict, edgelist=edges, alpha=edge_contrasts, edge_color='black')        #Draw  edges
    nx.draw_networkx_nodes(H, hit_coords_dict, node_size=200)                                                   #Draw nodes
    nx.draw_networkx_labels(H, hit_coords_dict)                                                                 #Draw hit labels.
      
    for detector_x in x:                                        #Shows positions of the detectors.
        plt.axvline(detector_x, linestyle='--', alpha=0.12)
    
    plt.title(f'{matrix_type} Graph Representation')
    #plt.savefig(f'plots/{matrix_type}_Graph.png')
    plt.show()


##############################################################################################################################


def plot_energy_landscape(lambda_bal, KNN_energies, RBF_energies): 
    
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
    #plt.savefig(f'plots/energy_landscape_λ={lambda_bal}.png')
    plt.show()
    

##############################################################################################################################

def plot_conv_trace(number_of_steps : int, energy_history : np.ndarray):
    steps = np.arange(number_of_steps)
    
    plt.plot(steps, energy_history)
    plt.xlabel('Steps')
    plt.ylabel('Ising energy')
    plt.title('Convergence Trace of Simulated Annealing algorithm')
    plt.show()


def plot_conv_traces(steps : np.ndarray, energy_histories : np.ndarray):
    reps = len(energy_histories)
    fig, ax = plt.subplots(reps, 1, figsize=(11,11))
    
    for i in range(reps):
        s = np.arange(steps[i])
        ax[i].plot(s, energy_histories[i], color='red')
    
    fig.suptitle('Convergence trace plots for different starting points in SA')
    fig.supylabel('Energy')
    fig.supxlabel('Step')
    plt.show()
    