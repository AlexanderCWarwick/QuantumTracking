import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import networkx as nx

np.random.seed(41)

def create_hits(N, x, m, c, sigma, truth_colour):
    '''
    Generate N evenly spaced hits on x. truth_colour designates which track hits belong to and is the truth label. 
    '''
    
    truth_labels = np.full(N, truth_colour)
    hits = (m*x + c) + np.random.normal(loc=0,scale=sigma,size=N)
    return hits, truth_labels



def generate_lineparams(m_low, m_high, c_low, c_high):    
    '''
    Random generator for gradient and y-intercept of particle tracks.
    '''      
    return np.random.uniform(m_low, m_high), np.random.uniform(c_low, c_high)

    
    
def construct_toytracks(x, N, sigma_noise, allow_intersection):
    '''
    Function generates the y coords and associated truth labels of all hits. Bounds on m and c are arbitrary but
    m is small to reflect a forward-type experiment.
        
    Distinction between whether the tracks are allowed to intercept or not is included for generality.
    '''
        
    track_lims = (-0.5, 0.5, 0, 1)                           
    m1, c1 = generate_lineparams(*track_lims)                          
    m2, c2 = 0, 0
    
    if allow_intersection:
        m2, c2 = generate_lineparams(*track_lims)
        
    else:
        while True:
            m2, c2 = generate_lineparams(*track_lims)
            
            while np.isclose(m1, m2, rtol=1e-8):
                m2, _ = generate_lineparams(*track_lims)
                
            x_int = (c1 - c2) / (m2 - m1)
            if x_int > 1 or x_int < 0:
                break 
            else:
                continue
        
    track1, track1_labels = create_hits(N, x, m1, c1, sigma_noise, 'red')
    track2, track2_labels = create_hits(N, x, m2, c2, sigma_noise, 'blue')

    return track1, track1_labels, track2, track2_labels



def plot_toytracks(x, track1, track1_labels, track2, track2_labels, sigma_noise, intsec):
    plt.scatter(x, track1, c=track1_labels, s=40, marker='o')
    plt.scatter(x, track2, c=track2_labels, s=40, marker='o')
    plt.xlim(-0.1, 1.1)
    plt.title(f'Two track plot with intersection={intsec}. Noise standev.={sigma_noise}')
    plt.grid(axis='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('plots/Toytracks.png')
    plt.show()
    
    
################################################################################################
    
    
def get_distmatrix(hit_coords): 
    '''
    Scipy cdist return matrix of all pairwise Eucldiean norm distances between hits.
    '''
    return cdist(hit_coords, hit_coords)

    
    
def construct_RBFmatrix(hit_coords, sigma_rbf):
    '''
    Radial Basis Function (RBF) similarity models the correlation between hits using the Euclidean distance
    between hits.  
    
    Similarity between two hits is modelled with a Gaussian. Very close hits have a RBF similaity close to 1, whilst 
    hits that are distant have close to 0 RBF similarity.
    '''     
                                                     
    return np.exp(-(get_distmatrix(hit_coords))**2 / (2*(sigma_rbf)**2))  


def contruct_KNN_matrix(x, hit_coords, n):
    '''
    The KNN matrix is a discrete matrix of 0's and 1's. neighbour_matrix find the n nearest neighbours of every 
    hit. If hit j is a nearest neighbour of i, then the (ij)th entry (and (ji)th since it must be symmetric) is 1.
    Otherwise it is 0. 
    
    '''
    
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(hit_coords)
    neighbour_matrix = nbrs.kneighbors_graph(hit_coords).toarray()

    knn_matrix = ((neighbour_matrix + neighbour_matrix.T)>0).astype(int)
    
    return knn_matrix, nbrs
    
    
    
def plot_matrix_heat_map(sim_matrix, matrix_type):
    '''
    Plot heat map representation of the matrix. The more correlated hits i and j are, the brighter the (ij)th 
    coordinate in the heat map.
    '''
        
    plt.imshow(sim_matrix, cmap='viridis')
    plt.title(f'{matrix_type} Heat map')
    plt.colorbar()
    plt.xlabel('Hit index')
    plt.savefig(f'plots/{matrix_type}_heat_map.png')
    plt.show()
     

################################################################################################
    
    
def construct_RBF_graphrep(x, number_of_hits, hit_coords_dict, RBF_matrix):
    H = nx.Graph()
    rbf_edge_weights = []
    
    for i in range(number_of_hits):
        for j in range(i+1, number_of_hits):
            rbf_edge_weight = RBF_matrix[i][j]
            H.add_edge(i, j, weight=rbf_edge_weight)
            rbf_edge_weights.append(rbf_edge_weight)
            
            
    rbf_edge_weights = np.asarray(rbf_edge_weights)
    
    edge_contrasts = 0.05 + 0.95 * (rbf_edge_weights - rbf_edge_weights.min()) / (rbf_edge_weights.max() - rbf_edge_weights.min())
    
    plot_graphrep(H, x, hit_coords_dict, H.edges(), edge_contrasts, 'RBF')
    
    

def construct_KNN_graphrep(x, number_of_hits, hit_coords, hit_coords_dict, nbrs):
    H = nx.Graph()
    
    _, indices = nbrs.kneighbors(hit_coords)
    indices = indices[:,1:]
    
    for i in range(number_of_hits):
        for j in indices[i]:
            H.add_edge(i,j)
    
    plot_graphrep(H, x, hit_coords_dict, H.edges(), None, 'KNN')
  
  
  
def plot_graphrep(H, x, hit_coords_dict, edges, edge_contrasts, matrix_type):

    nx.draw_networkx_edges(H, hit_coords_dict, edgelist=edges, alpha=edge_contrasts, edge_color='black')
    nx.draw_networkx_nodes(H, hit_coords_dict, node_size=200)
    nx.draw_networkx_labels(H, hit_coords_dict)
      
    for detector_x in x:
        plt.axvline(detector_x, linestyle='--', alpha=0.12)
    
    plt.title(f'{matrix_type} Graph Representation')
    plt.savefig(f'plots/{matrix_type}_Graph.png')
    plt.show()


################################################################################################


def main():
    track_hits = 6
    x = np.linspace(0,1,track_hits)
    sigma_noise = 1e-2
    sigma_rbf = 0.3
    intsection_allow = False
    nearneighb_n = 3

    track1, track1_labels, track2, track2_labels = construct_toytracks(x, track_hits, sigma_noise, intsection_allow)
    plot_toytracks(x, track1, track1_labels, track2, track2_labels, sigma_noise, intsection_allow)

    hit_coords = np.column_stack([np.concatenate([x, x]),np.concatenate([track1, track2])])
    number_of_hits = len(hit_coords)
    
    hit_coords_dict = {i: tuple(hit_coords[i]) for i in range(number_of_hits)}
    
    RBF_matrix = construct_RBFmatrix(hit_coords, sigma_rbf)
    plot_matrix_heat_map(RBF_matrix, 'Radial Basis Function')
    construct_RBF_graphrep(x, number_of_hits, hit_coords_dict, RBF_matrix)

    KNN_matrix, nbrs = contruct_KNN_matrix(x, hit_coords, nearneighb_n)
    plot_matrix_heat_map(KNN_matrix, f'{nearneighb_n}_NearestNeighbours')
    construct_KNN_graphrep(x, number_of_hits, hit_coords, hit_coords_dict, nbrs)
    
    
if __name__ == "__main__":
    main()