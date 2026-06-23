import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


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

    
def get_KNN_matrix(x, hit_coords, nearneighb_n):   
    KNN_matrix, nbrs = construct_KNN_matrix(hit_coords, nearneighb_n)                       #Calculate the KNN similarity matrix.
        
    return KNN_matrix, nbrs
    
    
    
def get_RBF_matrix(x, hit_coords):
    sigma_rbf = 0.1
        
    RBF_matrix = construct_RBFmatrix(hit_coords, sigma_rbf)                                 #Calculate the RBF similarity matrix.     
    return RBF_matrix
