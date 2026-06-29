import numpy as np

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
