import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def create_hits(N, x, m, c, sigma, truth_color):
    truth_labels = np.full(N, truth_color)
    hits = (m*x + c) + np.random.normal(loc=0,scale=sigma,size=N)
    return hits, truth_labels

def generate_lineparams(m_lowlimit, m_highlimit, c_lowlimit, c_highlimit):          #Generate model track parameters from uniform dist. Bounds of gradient
    m = np.random.uniform(m_lowlimit, m_highlimit)                                  #is chosen small (-0.3,0.3) to small to model LHCb 'forward-type' interactions.
    c = np.random.uniform(c_lowlimit, c_highlimit)  
    return m, c
    
def construct_toytracks(x,N,sigma,intsec):                              #Differetiate between intersecting or not with bool intsec. Tracks are particles hitting N 
    m1, c1 = generate_lineparams(-0.3,0.3,0,1)                          #equally spaced detectors. Model noise as gaussian, sd = sigma.
    m2, c2 = 0, 0
    if not intsec:
        while True:
            m2, c2 = generate_lineparams(-0.3,0.3,0,1)
            x_int = (c1 - c2) / (m2 - m1)
            if x_int > 1 or x_int < 0:
                break 
            else:
                continue
    if intsec:
        m1, c1 = generate_lineparams(-0.3,0.3,0,1)
        
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

def construct_RBFmatrix(hit_coords, sigma):                                                #Radial Basis Function type similarity matrix. [dist_matrix]_ij = d(i,j) 
    dist_matrix = cdist(hit_coords, hit_coords)                                            #with 0-5 being red truth labels and 6-11 being blue.
    RBF_matrix = np.exp(-(dist_matrix)**2 / 2*(sigma)**2)
    
    plt.imshow(RBF_matrix, cmap='viridis')
    plt.show()
    
Num_hits = 6
x = np.linspace(0,1,Num_hits)
sigma_noise = 0.01
intsec = False

track1, track1_labels, track2, track2_labels = construct_toytracks(x, Num_hits, sigma_noise, intsec)
plot_toytracks(x, track1, track1_labels, track2, track2_labels, sigma_noise, intsec)

hit_coords = np.column_stack([np.concatenate([x, x]),np.concatenate([track1, track2])])
construct_RBFmatrix(hit_coords, 0.01)