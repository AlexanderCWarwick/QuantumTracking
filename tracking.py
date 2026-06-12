import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import networkx as nx

np.random.seed(41)

def create_hits(N, x, m, c, sigma, truth_color):
    truth_labels = np.full(N, truth_color)
    hits = (m*x + c) + np.random.normal(loc=0,scale=sigma,size=N)
    return hits, truth_labels



def generate_lineparams(m_low, m_high, c_low, c_high):          
    return np.random.uniform(m_low, m_high), np.random.uniform(c_low, c_high)

    
    
def construct_toytracks(x,N,sigma_noise,allow_intersection):
    track_lims = (-0.5, 0.5, 0, 1)                           #Gradient bounds small in forward-type experiment.
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



def plot_toytracks(x, track1, track1_labels, track2, track2_labels, sigma, intsec):
    plt.scatter(x, track1, c=track1_labels, s=40, marker='o')
    plt.scatter(x, track2, c=track2_labels, s=40, marker='o')
    plt.xlim(-0.1, 1.1)
    plt.title(f'Two track plot with intersection={intsec}. Noise standev.={sigma}')
    plt.grid(axis='x')
    plt.savefig('plots/Toytracks.png')
    plt.show()
    
    
    
def get_distmatrix(hit_coords): 
    return cdist(hit_coords, hit_coords)

    
    
def construct_RBFmatrix(hit_coords, sigma_rbf):                                                  
    return np.exp(-(get_distmatrix(hit_coords))**2 / (2*(sigma_rbf)**2))  


    
def plot_matrix_heat_map(sim_matrix, matrix_type):
    plt.imshow(sim_matrix, cmap='viridis')
    plt.title(f'{matrix_type} Heat map')
    plt.colorbar()
    plt.xlabel('Hit index')
    plt.savefig(f'plots/{matrix_type}_heat_map.png')
    plt.show()
     
     
    
def contruct_KNN_matrix(x, hit_coords, n, t1l, t2l, sigma_rbf):
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(hit_coords)
    neighbour_matrix = nbrs.kneighbors_graph(hit_coords).toarray()

    knn_matrix = ((neighbour_matrix + neighbour_matrix.T)>0).astype(int)
    return knn_matrix, nbrs

        
        
def plot_KNN_graphrep(x, number_of_hits, nbrs, hit_coords):
    H = nx.Graph()
    hit_coords_dict = {i: tuple(hit_coords[i]) for i in range(len(hit_coords))}
    
    _, indices = nbrs.kneighbors(hit_coords)
    indices = indices[:,1:]
    
    for i in range(len(hit_coords)):
        for j in indices[i]:
            H.add_edge(i,j)
           
    nx.draw_networkx_labels(H, hit_coords_dict)
    
    for detector_x in x:
        plt.axvline(detector_x,linestyle='--',alpha=0.08)
        
    nx.draw(H, hit_coords_dict)
    plt.title('KNN_graph')
    plt.savefig('plots/KNN_Graph.png')
    plt.show()
    
    
    
def plot_RBF_graphrep(x, number_of_hits, hit_coords, RBF_matrix):
    H = nx.Graph()
    hit_coords_dict = {i: tuple(hit_coords[i]) for i in range(number_of_hits)}
    rbf_edge_weights = []
    
    for i in range(number_of_hits):
        for j in range(i+1, number_of_hits):
            rbf_edge_weight = RBF_matrix[i][j]
            H.add_edge(i, j, weight=rbf_edge_weight)
            rbf_edge_weights.append(rbf_edge_weight)
            
            
    rbf_edge_weights = np.asarray(rbf_edge_weights)
    
    edge_contrasts = 0.05 + 0.95 * (rbf_edge_weights - rbf_edge_weights.min()) / (rbf_edge_weights.max() - rbf_edge_weights.min())

        
    nx.draw_networkx_edges(H, hit_coords_dict, edgelist=H.edges(), alpha=edge_contrasts, edge_color='black')
    nx.draw_networkx_nodes(H, hit_coords_dict, node_size=200)
    nx.draw_networkx_labels(H, hit_coords_dict)
       
    for detector_x in x:
        plt.axvline(detector_x, linestyle='--', alpha=0.08)
    
    plt.title('RBF Graph')
    plt.savefig('plots/RBF_Graph.png')
    plt.show()



def main():
    track_hits = 6
    x = np.linspace(0,1,track_hits)
    sigma_noise = 1e-2
    sigma_rbf = 0.2
    intsection_allow = False
    nearneighb_n = 3

    track1, track1_labels, track2, track2_labels = construct_toytracks(x, track_hits, sigma_noise, intsection_allow)
    plot_toytracks(x, track1, track1_labels, track2, track2_labels, sigma_noise, intsection_allow)

    hit_coords = np.column_stack([np.concatenate([x, x]),np.concatenate([track1, track2])])
    number_of_hits = len(hit_coords)
    
    RBF_matrix = construct_RBFmatrix(hit_coords, sigma_rbf)
    plot_matrix_heat_map(RBF_matrix, 'Radial Basis Function')
    plot_RBF_graphrep(x, number_of_hits, hit_coords, RBF_matrix)

    KNN_matrix, nbrs = contruct_KNN_matrix(x, hit_coords, nearneighb_n, track1_labels, track2_labels, sigma_rbf)
    plot_matrix_heat_map(KNN_matrix, f'{nearneighb_n}_NearestNeighbours')
    plot_KNN_graphrep(x, number_of_hits, nbrs, hit_coords)
    
    
main()