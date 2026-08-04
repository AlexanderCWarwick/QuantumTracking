import numpy as np
from classical_algs.ising import ising_energy
from circuits.qaoa_circuit import bind_params

def evaluate(W,  
             circuit,  
             backend,   
             gamma,  
             beta, 
             gamma_values, 
             beta_values, 
             lambda_bal,  
             no_of_shots, 
             p)  ->  float:
    '''
    Evaluation step. Binds parameter inputs to the general QAOA circuit.
    Returns the average energy of the such circuit after no_of_shots samples.
    '''
    
    paramed_circuit = bind_params(circuit, gamma, beta, gamma_values, beta_values, p)
    counts = run_qaoa(backend, 
                      paramed_circuit, 
                      no_of_shots)
            
    avg_energy = 0 
    for rev_config, count in counts.items():
        config = rev_config[::-1]                         #Corrects for qiskit endian convention (qubits are ordered in reverse)  
        config = np.array(list(config), dtype=int)        #counts is a dictionary of bitstrings and their corresponding frequencies. The bitstrings are given as strings so convert to a np array
        config_energy = ising_energy(W, config, lambda_bal)               
        avg_energy += config_energy * (count / no_of_shots)
        
    return avg_energy


def run_qaoa(backend, circuit, no_of_shots):
    '''
    Input: The simulator standing in for the quantum computer, the binded circuit and the number of shots.
    Output: The sampled probability distribution for that circuit with those specific parameter values.
    '''
    
    result = backend.run(circuit, 
                         shots=no_of_shots).result()
    counts = result.get_counts()
    #counts is the sampling histogram, e.g. '110101' was meausred 37 times etc.
    
    return counts
    