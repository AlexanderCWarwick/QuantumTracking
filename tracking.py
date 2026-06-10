import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def create_hits(N, x, m, c, sigma, truth_color):
    truth_labels = np.full(N, truth_color)
    hits = (m*x + c) + np.random.normal(loc=0,scale=sigma,size=N)
    return hits, truth_labels

def generate_lineparams(c_lowlimit, c_highlimit, m_lowlimit, m_highlimit):
    m = np.random.uniform(m_lowlimit, m_highlimit)
    c = np.random.uniform(c_lowlimit, c_highlimit)  
    return m, c
    
def construct_toytracks(x,N,intsec):
    m1, c1 = generate_lineparams(0,1,-0.3,0.3)
    sigma = 0.005
    m2, c2 = 0, 0
    if intsec == False:
        while True:
            m2, c2 = generate_lineparams(0,1,-0.3,0.3)
            x_int = (c1 - c2) / (m2 - m1)
            if x_int > 1 or x_int < 0:
                break 
            else:
                continue
        
    track1, track1_labels = create_hits(N, x, m1, c1, sigma, 'red')
    track2, track2_labels = create_hits(N, x, m2, c2, sigma, 'blue')

    plt.scatter(x, track1, c=track1_labels, s=40, marker='o')
    plt.scatter(x, track2, c=track2_labels, s=40, marker='o')
    plt.xlim(-0.1, 1.1)
    plt.title(f'Two track plot with intersection={intsec}. Noise standev.={sigma}')
    plt.grid(axis='x')
    plt.show()
    
    return track1, track1_labels, track2, track2_labels
  
def construct_RBFmatrix(hit_coords, sigma):
    dist_matrix = cdist(hit_coords, hit_coords)
    RBF_matrix = np.exp(-(dist_matrix)**2 / 2*(sigma)**2)
    
    plt.imshow(RBF_matrix, cmap='viridis')
    plt.show()
    

Num_hits = 6
x = np.linspace(0,1,Num_hits)
track1, _, track2, _ = construct_toytracks(x, Num_hits, False)
hit_coords = np.column_stack([np.concatenate([x, x]),np.concatenate([track1, track2])])
construct_RBFmatrix(hit_coords, 0.01)