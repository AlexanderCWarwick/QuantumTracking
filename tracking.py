import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import networkx as nx

def create_hits(N, x, m, c, sigma, truth_color):
    truth_labels = np.full(N, truth_color)
    hits = (m*x + c) + np.random.normal(loc=0,scale=sigma,size=N)
    return hits, truth_labels

def generate_lineparams(m_low, m_high, c_low, c_high):          
    return np.random.uniform(m_low, m_high), np.random.uniform(c_low, c_high)
    
def construct_toytracks(x,N,sigma,intsec):
    lims = (-0.5,0.5,0,1)           #Gradient bounds small in forward-type experiment.
    m1, c1 = generate_lineparams(*lims)                          
    m2, c2 = 0, 0
    if not intsec:
        while True:
            m2, c2 = generate_lineparams(*lims)
            x_int = (c1 - c2) / (m2 - m1)
            if x_int > 1 or x_int < 0:
                break 
            else:
                continue
    if intsec:
        m2, c2 = generate_lineparams(*lims)
        
    track1, track1_labels = create_hits(N, x, m1, c1, sigma, 'red')
    track2, track2_labels = create_hits(N, x, m2, c2, sigma, 'blue')

    return track1, track1_labels, track2, track2_labels

def plot_toytracks(x, track1, track1_labels, track2, track2_labels, sigma, intsec):
    plt.scatter(x, track1, c=track1_labels, s=40, marker='o')
    plt.scatter(x, track2, c=track2_labels, s=40, marker='o')
    plt.xlim(-0.1, 1.1)
    plt.title(f'Two track plot with intersection={intsec}. Noise standev.={sigma}')
    plt.grid(axis='x')
    plt.show()
    
def get_distmatrix(hit_coords): 
    return cdist(hit_coords, hit_coords)

def construct_RBFmatrix(hit_coords, sigma):                                                #Radial Basis Function type similarity matrix. [D]_ij = d(i,j) for normal euclid norm    
    return np.exp(-(get_distmatrix(hit_coords))**2 / (2*(sigma)**2))  
    
def plot_matrix_heat_map(sim_matrix, matrix_type):
    plt.imshow(sim_matrix, cmap='viridis')
    plt.title(f'{matrix_type} Heat map')
    plt.colorbar()
    plt.show()
    
def KNN(x, hit_coords, n, t1l, t2l):
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(hit_coords)
    distances, indices = nbrs.kneighbors(hit_coords)
    distances = distances[:,1:]
    indices = indices[:,1:]

    G = nx.Graph()
    pos = {i: tuple(hit_coords[i]) for i in range(len(hit_coords))}

    edges = []

    for i in range(len(hit_coords)):
        for j, d in zip(indices[i], distances[i]):
            edges.append((i, j, d))
            
    G.add_weighted_edges_from(edges)  
    for detector_x in x:
        plt.axvline(detector_x,linestyle='--',alpha=0.2) 
    plt.title(f'{n} NN Undirected Graph')

    nx.draw(G,pos=pos, node_color=np.concatenate([t1l, t2l]))
    
Num_hits = 6
x = np.linspace(0,1,Num_hits)
sigma_noise = 0.01
intsec = False

track1, track1_labels, track2, track2_labels = construct_toytracks(x, Num_hits, sigma_noise, intsec)
plot_toytracks(x, track1, track1_labels, track2, track2_labels, sigma_noise, intsec)

hit_coords = np.column_stack([np.concatenate([x, x]),np.concatenate([track1, track2])])
RBF_matrix = construct_RBFmatrix(hit_coords, 0.5)

plot_matrix_heat_map(RBF_matrix, 'Radial Basis Function')
plot_matrix_heat_map(get_distmatrix(hit_coords), 'Distance matrix')
KNN(x, hit_coords, 3, track1_labels, track2_labels)

