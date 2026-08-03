import numpy as np

def ising_energy(W : np.ndarray[np.float64], bitstring : np.ndarray[int],  lambda_bal : float):
    '''
    Ising objective function. The function rewards like spins and discourages extreme configurations with a penalty term e.g. 111111111111.
    Note that the approach here is to use brute force since there is only 12 hits. If n = 20 then brute force would be VERY inefficient.
    '''
    isingstring = (2 * bitstring) - 1                   #Convert the bitstring (configuration) into a ising spin configuration.
    n = len(W)
    H = 0
    for i in range(n):
        for j in range(i+1, n):                                         #j > i in Hamiltonian. Avoids double counting the interacting spins.
            H -= W[i][j] * isingstring[i] * isingstring[j]              #Rewarding term for like spins
    
    H += ising_penalty_term(lambda_bal, isingstring)                    #Penalty term penalising big clustering.
    
    return H


def ising_penalty_term(lambda_bal, isingstring : np.ndarray[float]) -> int:
    '''
    Type of penalty term to be used in the Ising energy. Should be non-negative.
    Types: 
    1. lambda_bal * (np.sum(isingstring))**2  
    
    2. lambda_bal * (mod(np.sum(isingstring)))
    3. lambda_bal * (np.sum(isingstring))**4
    4. lambda_bal * (sum_(i<j)(isingstring))**2
    '''
   
    return lambda_bal * (np.sum(isingstring))**2

