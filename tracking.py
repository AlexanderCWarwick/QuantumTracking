import numpy as np
import matplotlib.pyplot as plt

def create_hits(N, x, m, c, sigma, truth_color):
    truth_labels = np.full(N, truth_color)
    hits = (m*x + c) + np.random.normal(loc=0,scale=sigma,size=N)
    return hits, truth_labels

def generate_lineparams(c_lowlimit, c_highlimit, m_lowlimit, m_highlimit):
    m = np.random.uniform(m_lowlimit, m_highlimit)
    c = np.random.uniform(c_lowlimit, c_highlimit)  
    return m, c
    
def plot_toytracks(N,intsec):
    x = np.linspace(0,1,N)
    m1, c1 = generate_lineparams(0,1,-0.3,0.3)
    sigma = 0.005
    m2 = 0
    c2 = 0
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
    
    plt.scatter(x, track1, c=track1_labels, s=30, marker='x')
    plt.scatter(x, track2, c=track2_labels, s=30, marker='x')
    plt.xlim(-0.1, 1.1)
    plt.title(f'Two track plot with intersection={intsec}. Noise standev.={sigma}')
    plt.grid(axis='x')
    plt.show()
  
plot_toytracks(6, False)