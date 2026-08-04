import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

def ari_check(true_groundstate : np.ndarray,  optimised_tracks : np.ndarray[np.ndarray[int]]) -> np.ndarray[float]:
    '''
    Adjusted random score measures randomness of the cluster labels. It compares the computed groundstate and the true answer
    and returns: 
    ARI = 1 - Perfect clustering (what we are aiming for)
    0 < ARI < 1 - Random clustering
    ARI < 0 - Something has gone wrong
    
    Test the optimised groundstates against ONLY ONE of the true groundstate tracks, here called true_track.
    Hence double loop (ising_optimisation())
    '''
    aris = []
    for track in optimised_tracks:
        aris.append(adjusted_rand_score(true_groundstate, track))
    
    return np.array(aris)
