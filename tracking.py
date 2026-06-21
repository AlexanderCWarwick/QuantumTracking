import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.cluster import adjusted_rand_score
import networkx as nx
from itertools import product

np.random.seed(41)

#######################################################     Generate toy data set     #######################################################

def create_hits(N, x,  m : np.float64,  c : np.float64, sigma_noise,  truth_label : int)   ->  tuple[np.ndarray[np.float64], np.ndarray[int]]:
    '''
    Generate N evenly spaced hits on x. truth_colour designates which track hits belong to and is the truth label. 
    '''
    
    truth_labels = np.full(N, truth_label)
    hits = (m*x + c) + np.random.normal(loc=0, scale=sigma_noise, size=N)               #Track modelled as straight line: y = mx + c over domain [0,1].
    return hits, truth_labels



def generate_lineparams(m_low, m_high, c_low, c_high)  ->  tuple[int, int]:    
    '''
    Random generator for gradient and y-intercept of particle tracks.
    '''      
    return np.random.uniform(m_low, m_high), np.random.uniform(c_low, c_high)

    
    
def construct_toytracks(x : np.ndarray[np.float64],  N : int,  sigma_noise : np.float64,  allow_intersection : bool
                    
                    )  ->  tuple[np.ndarray[np.float64],  np.ndarray[int],  np.ndarray[np.float64],  np.ndarray[int]]:
    '''
    Function generates the y coords and associated truth labels of all hits. Bounds on m and c are arbitrary but
    m is small to reflect a forward-type experiment. 
    Distinction between whether the tracks are allowed to intercept or not is included for generality.
    '''
        
    track_parameter_limits = (-0.5, 0.5, 0, 1)                           
    m0, c0 = generate_lineparams(*track_parameter_limits)                          
    m1, c1 = 0, 0
    
    if allow_intersection:
        m1, c1 = generate_lineparams(*track_parameter_limits)
        
    else:
        while True:
            m1, c1 = generate_lineparams(*track_parameter_limits)
            
            while np.isclose(m0, m1, rtol=1e-8):                   
                m1, _ = generate_lineparams(*track_parameter_limits)
                
            x_int = (c0 - c1) / (m1 - m0)
            if x_int > 1 or x_int < 0:
                break 
            else:
                continue
        
    track0, track0_labels = create_hits(N, x, m0, c0, sigma_noise, 0)
    track1, track1_labels = create_hits(N, x, m1, c1, sigma_noise, 1)

    return track0, track0_labels, track1, track1_labels



def plot_toytracks(x, track1, track2, sigma_noise, allow_intsection):
    
    plt.scatter(x, track1, c='blue', s=40, marker='o')
    plt.scatter(x, track2, c='red', s=40, marker='o')
    plt.xlim(-0.1, 1.1)
    plt.title(f'Two track plot with intersection={allow_intsection}. Noise standev.={sigma_noise}')
    plt.grid(axis='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('plots/Toytracks.png')
    plt.show()
    
    
#######################################################     Construct similarity matrices     #######################################################
    
    
def get_distmatrix(hit_coords : np.array) -> np.array: 
    '''
    Scipy cdist return matrix of all pairwise Eucldiean norm distances between hits.
    '''
    return cdist(hit_coords, hit_coords)

    
    
def construct_RBFmatrix(hit_coords,  sigma_rbf : np.float64)  ->  np.ndarray[np.float64]:
    '''
    Radial Basis Function (RBF) similarity models the correlation between hits using the Euclidean distance
    between hits.  
    
    Similarity between two hits is modelled with a Gaussian. Very close hits have a RBF similaity close to 1, whilst 
    hits that are distant have close to 0 RBF similarity.
    '''                                                    
    return np.exp(-(get_distmatrix(hit_coords))**2 / (2*(sigma_rbf)**2))  



def construct_KNN_matrix(hit_coords, n : int):
    '''
    The KNN matrix is a discrete matrix of 0's and 1's. neighbour_matrix find the n nearest neighbours of every 
    hit. If hit j is a nearest neighbour of i, then the (ij)th entry (and (ji)th since it must be symmetric) is 1.
    Otherwise it is 0. 
    '''
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(hit_coords)       
    neighbour_matrix = nbrs.kneighbors_graph(hit_coords).toarray()              #Creates binary matrix detailing which hits are connected.

    knn_matrix = ((neighbour_matrix + neighbour_matrix.T)>0).astype(int)        #Symmetrise knn_matrix. as.type(int) converts binary boolean values back to integers 0 and 1.
    
    return knn_matrix, nbrs
    
    
def construct_weightedKNNmatrix(KNN_matrix, RBF_matrix):
    '''
    The weighted KNN matrix combines the KNN and RBF. The k nearest neighbours are weighted by their RBF matrix values. 
    '''
    return KNN_matrix * RBF_matrix          #Elementwise multiplication.
    
    
def plot_similaritymatrix_heatmap(sim_matrix,  matrix_type : str):
    '''
    Plot heat map representation of any input similarity matrix. The more correlated hits i and j are, the brighter the (ij)th 
    coordinate in the heat map.
    '''
    plt.imshow(sim_matrix, cmap='viridis')
    plt.title(f'{matrix_type} Heat map')
    plt.colorbar()
    plt.xlabel('Hit index')
    plt.savefig(f'plots/{matrix_type}_heat_map.png')
     

#######################################################    Graph Plots    #######################################################
    
    
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



def construct_KNN_graphrep(x, number_of_hits, hit_coords, hit_coords_dict,  nbrs : NearestNeighbors):
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
    plt.savefig(f'plots/{matrix_type}_Graph.png')
    


#######################################################     Ising Optimisation     #######################################################       

def ising_energy(bitstring : np.ndarray[int], W : np.ndarray[np.float64], lambda_bal : float):
    '''
    Ising objective function. The function rewards like spins and discourages extreme configurations with a penalty term e.g. 111111111111.
    '''
    isingstring = (2 * bitstring) - 1                   #Convert the bitstring (configuration) into a ising spin configuration.
    n = len(W)
    H = 0

    for i in range(n):
        for j in range(i+1, n):                                         #j > i in Hamiltonian. Avoids double counting the interacting spins.
            H -= W[i][j] * isingstring[i] * isingstring[j]              #Rewarding term for like spins
            
    H += ising_penalty_term(lambda_bal, isingstring)                    #Penalty term penalising big clustering.
    
    return H


def ising_penalty_term(lambda_bal, isingstring):
    '''
    Type of penalty term to be used in the Ising energy. Should be non-negative.
    Types: 
    1. lambda_bal * (np.sum(isingstring))**2  
    
    2. lambda_bal * (mod(np.sum(isingstring)))
    3. lambda_bal * (np.sum(isingstring))**4
    4. lambda_bal * (sum_(i<j)(isingstring))**2
    '''
    
    return lambda_bal * (np.sum(isingstring))**2



def get_ising_energies(W, lambda_bal, config_space):
    '''
    Returns the 'energy landscape' of the the chosen hamiltonian. 
    '''
    energies = []
    for bitstring in config_space:
        energy = ising_energy(bitstring, W, lambda_bal)
        energies.append(energy)

    return np.array(energies)



def get_groundstate(energies, groundstate_energy, config_space):
    '''
    Returns where the ground state configuration is (the indices) using the energies array.
    Exchange degeneracy means energy landscape is symmetric. Hence there are at least two ground states.
    '''
    groundstates_indices = np.where(energies == groundstate_energy)[0]
    groundstate_configs = np.array([config_space[i] for i in groundstates_indices])

    return groundstate_configs



def ising_optimise(sim_matrix, lambda_bal, config_space):
    '''
    Primary function of the Ising optimisation block. Returns:
    1. The energy landscape 
    2. The calculated ground state energy
    3. The ground state configurations
    '''
    
    energies = get_ising_energies(sim_matrix, lambda_bal, config_space)
    groundstate_energy = min(energies)  
    
    return energies, groundstate_energy, get_groundstate(energies, groundstate_energy, config_space)     
            
        
def plot_energy_landscape(decimal_config_space, lambda_bal,
                          KNN_energies, KNN_groundstate_energy,
                          RBF_energies, RBF_groundstate_energy): 
    
    fig, ax = plt.subplots(1, 2, figsize=(8,4))  
    
    ax[0].plot(np.sort(KNN_energies)[:10], color='orange')
    ax[0].set_title('KNN similarity')
    
    ax[1].plot(np.sort(RBF_energies)[:10], color='red')
    ax[1].set_title('RBF similarity')
    
    fig.suptitle(f'Energy landscape for lambda={lambda_bal}')
    fig.supxlabel('Rank')
    fig.supylabel('Energy')
    plt.tight_layout()
    
    plt.savefig(f'plots/energy_landscape_λ={lambda_bal}.png')
    
       

def ARI_check(true_track, groundstate_track):  
    '''
    Adjusted random score measures randomness of the cluster labels. It compares the computed groundstate and the true answer
    and returns: 
    ARI = 1 - Perfect match (what we are aiming for)
    0 < ARI < 1 - Random assignment (0 being worst).
    ARI < 0 - Something has gone wrong.
    '''
    
    return adjusted_rand_score(true_track , groundstate_track) 
    
    
#######################################################     Main workflow     #######################################################


def main():
    def toytrack_similarity_plot(track_hits, intersection_allowed):
        '''
        Problem: Two particles pass through a detector each leaving N 'hits'. Using these 'hits' as coordinate
        locations of the particles, can we resolve the two tracks from each other and hence determine each particle's
        trajectory?
        
        Method:
        Two toy tracks (with noise) are randomly generated in 2D plane. 
        From the tracks obtain two types of similarity matrices:
        - RBF (radial basis function)
        - KNN (k nearest neighbours)
        - wKNN (weighted k nearest neighbours)
        Matrix elements measure the likelyhood that one hit is compatible to another, i.e. if they are the same particle.
        Obtain plots of these matrices as heatmaps.
        Obtain Graph representations of these similarity matrices.
        '''
                             
        x = np.linspace(0,1,track_hits)         #Positions of detectors
        sigma_noise = 1e-2                      #External noise 
        sigma_rbf = 0.1                         #RBF standard dev parameter
        nearneighb_n = 3                       #Number of nearest neighbours to consider in the KNN matrix

        track0, track0_truthlabels, track1, track1_truthlabels = construct_toytracks(x, track_hits, sigma_noise, intersection_allowed)
        plot_toytracks(x, track0, track1, sigma_noise, intersection_allowed)
        #truth_track_labels = np.array([np.concatenate([track0_truthlabels, track1_truthlabels]), np.concatenate([track1_truthlabels, track0_truthlabels])])
        
        hit_coords = np.column_stack([np.concatenate([x, x]),np.concatenate([track0, track1])])         #2D array of hit coordinates.
        number_of_hits = len(hit_coords)
        
        hit_coords_dict = {i: tuple(hit_coords[i]) for i in range(number_of_hits)}
        
        KNN_matrix, nbrs = construct_KNN_matrix(hit_coords, nearneighb_n)                       #Calculate the KNN similarity matrix.
        plot_similaritymatrix_heatmap(KNN_matrix, f'{nearneighb_n}_NearestNeighbours')
        construct_KNN_graphrep(x, number_of_hits, hit_coords, hit_coords_dict, nbrs)
        
        RBF_matrix = construct_RBFmatrix(hit_coords, sigma_rbf)                                 #Calculate the RBF similarity matrix. 
        plot_similaritymatrix_heatmap(RBF_matrix, 'Radial Basis Function')
        construct_RBF_graphrep(x, number_of_hits, hit_coords_dict, RBF_matrix)
        
        #wKNN_matrix = construct_weightedKNNmatrix(KNN_matrix, RBF_matrix)                       #Calculate the weighted KNN matrix.
        #wKNN_matrix heatmap and graph representation are just  versions of RBF.
        
        
        return number_of_hits, track0_truthlabels, track1_truthlabels, KNN_matrix, RBF_matrix
        
    def ising_optimisation_of_similarity_matrices(number_of_hits, true_groundstate0, true_groundstate1, KNN_matrix, RBF_matrix):
        
        '''
        Turn problem into an optimisation problem. The optimal track is the one that minimises the energy objective, the ground state. Ideally these are
        000000111111 and 111111000000.
        Here the energy objective is modelled as an Ising type function.
        
        The configuration space is the set of all possible hit labellings. Since each label is 0 or 1, this equates to assigning every configuration a binary string
        from 000000000000 up to 111111111111. This forms the binary configuration space with 2^12 = 4096 possibilities. 
        
        The objective function depends on a balance parameter, λ, which serves to discourage extreme configurations. 
        This simple optimisation technique will be tested for different λ values along with a ARI function to act as a numerical check on the randomness 
        of the calculated ground state. 
        '''
        
        lambda_bal = 1.0                                                                        #Lambda_balance parameter values to be used in the Hamiltonian. Modelled as a constant.
        binary_config_space = np.array(list(product([0,1], repeat=number_of_hits)))             #List of all 2^12 possible BINARY label configurations.
        decimal_config_space = np.arange(0, 2**(number_of_hits))                                #List of corresponding DECIMAL configurations.  
        
        
        KNN_energies, KNN_groundstate_energy, KNN_groundstate_binary_configs = ising_optimise(KNN_matrix, lambda_bal, binary_config_space)
        RBF_energies, RBF_groundstate_energy, RBF_groundstate_binary_configs = ising_optimise(RBF_matrix, lambda_bal, binary_config_space)
            
        plot_energy_landscape(decimal_config_space, lambda_bal,
                            KNN_energies, KNN_groundstate_energy,
                            RBF_energies, RBF_groundstate_energy)
            
        KNN_aris = np.array([ARI_check(true_groundstate0, state) for state in KNN_groundstate_binary_configs])
        RBF_aris = np.array([ARI_check(true_groundstate0, state) for state in KNN_groundstate_binary_configs])
            
        plt.show()
        
    track_hits = 6                          #Number of detectors
    intersection_allowed = False            #Boolean to control whether particles intersect
    
    number_of_hits, track0_truthlabels, track1_truthlabels, KNN_matrix, RBF_matrix = toytrack_similarity_plot(track_hits, intersection_allowed)
    true_groundstate0 = np.concatenate([track0_truthlabels, track1_truthlabels])
    true_groundstate1 = np.concatenate([track1_truthlabels, track0_truthlabels])
    
    ising_optimisation_of_similarity_matrices(number_of_hits, true_groundstate0, true_groundstate1, KNN_matrix, RBF_matrix)
    
if __name__ == "__main__":
    main()