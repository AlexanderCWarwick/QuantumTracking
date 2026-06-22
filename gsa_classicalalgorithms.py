import numpy as np

#####################################   Greedy  #####################################

def greedy_alg(W : np.ndarray[float]):
    '''
    Greedy algorithm optimisation approach. Greedy makes local (short-sighted) decisions. Given a Yes/No question, go with which every gives the most benefit 
    at time when choosing.
    Algorithm:
    1. Find the two most dissimilar hits according to similarity matrix and split them into clusters, 0 and 1.
    2. Randomly pick a hit not yet assigned a cluster.
    3. Compute avg of all compatible hits.
    4. If mean_0 > mean_1 then cluster 0 is favourable and the hit is assigned to cluster 0, and vice versa.
    5. Repeat for all hits.
    
    Sets are used for functionality over np arrays. 
    '''
    
    n = len(W)
    hits_to_assign = set(range(n))                        #Hits that we have yet to assign. 
    
    W_no_diag = W.copy()                                    
    np.fill_diagonal(W_no_diag, np.inf)                         #We don't want to include the diagonals so we force them, in this copy, to inf.

    i, j = np.unravel_index(np.argmin(W_no_diag), W_no_diag.shape)          #np.argmin finds the indices of the minimum. Since W is symmetric we only need one position.
    hits_to_assign.remove(i)
    hits_to_assign.remove(j)
    
    cluster0 = {i}
    cluster1 = {j}
    
    while len(hits_to_assign) != 0:                     
        k = np.random.choice(list(hits_to_assign))                  #Randomly select a node from the hits we have yet to assign.
        
        mean_0 = np.mean(np.array([W[k][x] for x in cluster0]))       #Calculate means of the similarity matrix row k (excluding hits that aren't yet selected in the cluster).
        mean_1 = np.mean(np.array([W[k][x] for x in cluster1]))
        
        if mean_0 > mean_1:
            cluster0.add(k)
        else:
            cluster1.add(k)
        hits_to_assign.remove(k)
        
    return np.array(list(cluster0)), np.array(list(cluster1))