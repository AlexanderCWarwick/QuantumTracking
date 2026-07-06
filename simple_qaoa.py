import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import Aer
from ising import ising_energy, ARI_check
from plotting import plot_energy_hist
import time

def qaoa(W : np.ndarray,  lambda_bal : float, no_of_shots : int):
    N = len(W)
    qreg_q = QuantumRegister(N, 'q')
    creg_c = ClassicalRegister(N, 'c')
    grid_counts = 20
    backend = Aer.get_backend('aer_simulator')
    backend.set_options(seed_simulator=44)
    
    best_energy = np.inf
    best_gamma = 0
    best_beta = 0
    best_counts = 0
        
    start = time.time()
    
    for gamma in np.linspace(0, np.pi, grid_counts):              #Grid search optimisation for tuning parameters beta and gamma.
        for beta in np.linspace(0, np.pi/2, grid_counts):
            circuit = build_qaoa_circuit(W, lambda_bal, gamma, beta)           #Build a new circuit for eevry (gamma, beta) pair. (Hadamard layer redundant)
            
            if gamma == 0 and beta == 0:
                #Plotting selection
                circuit.draw('mpl', fold=-1)
                plt.show()
                
            counts = run_qaoa(backend, circuit, no_of_shots)
            
    avg_energy = 0 
    for rev_config, count in counts.items():
        config = rev_config[::-1]                         #Corrects for qiskit endian convention (qubits are ordered in reverse)  
        config = np.array(list(config), dtype=int)        #counts is a dictionary of bitstrings and their corresponding frequencies. The bitstrings are given as strings so convert to a np array
        config_energy = ising_energy(W, config, lambda_bal)               
        avg_energy += config_energy * (count / no_of_shots)      
                
    if avg_energy < best_energy:
        #If this 'shot' measures a better expectation value of the energy, it replaces the previous best shot.
        best_energy = avg_energy
        best_gamma = gamma
        best_beta = beta
        best_counts = counts
                
    end = time.time()
    
    return best_counts, best_gamma, best_beta, (end-start)



def build_qaoa_circuit(W, lambda_bal, gamma : float, beta : float):
    '''
    p=1 QAOA circuit. Superposition --> Cost Layer --> Mixer layer --> Measurement
    '''
    
    N = len(W)
    qreg_q = QuantumRegister(N, 'q')
    creg_c = ClassicalRegister(N, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)
    
    circuit.h(qreg_q)                       #Superposition layer
    circuit.barrier()
            
    for i in range(N):
        for j in range(i+1, N):             #Start at i+1 since we don't want to double count the similarity measures.

            J = 2*lambda_bal - W[i][j]                                  #Effective coupling matrix. Equivalent to classical Ising energy.
            circuit.rzz(2*gamma*J, qreg_q[i], qreg_q[j])                #Cost layer. Applies RZZ gates to all connected vertices. Factor of 2 cancels the qiskit convention of a gamma/2.
    circuit.barrier() 
                   
    circuit.rx(2 * beta, qreg_q)            #Mixer layer. Applies RX gates to every qubit. Allows for interference between qubit phases.
    circuit.barrier()            
    circuit.measure(qreg_q, creg_c)
    
    return circuit
    
    
    
def run_qaoa(backend, circuit, no_of_shots):
    result = backend.run(circuit, shots=no_of_shots).result()
    counts = result.get_counts()
    #counts is the sampling histogram, e.g. '110101' was meausred 37 times etc.
    
    return counts



def get_groundstate_prob(best_counts, true_groundstate, no_of_shots):
    '''
    Returns the sample probability of measuring the groundstate configuration.
    
    Using the endian corrected bitstring as a key, search through the best counts (the collection of no_of_shots) samples for the best
    beta and gamma parameters.
    '''
    
    gs1_counts = best_counts.get(''.join(true_groundstate[::-1].astype(str)), 0)            
    gs2_counts = best_counts.get(''.join((true_groundstate^1)[::-1].astype(str)), 0)
    return (gs1_counts + gs2_counts) / no_of_shots


def energy_data(best_counts, W, lambda_bal, true_groundstate_energy):
    '''
    Instead of plotting a histogram of how many times each configuration was measured, e.g. '0010' : 102, '1001' : 20 etc, 
    we instead convert this into an energy distribution. 
    '''
    energies = []
    
    for config, count in best_counts.items():
        config = config[::-1]
        config = np.array(list(config), dtype=int)
        config_energy = ising_energy(W, config, lambda_bal)
        energies.extend([config_energy] * count)
        
    plot_energy_hist(energies, true_groundstate_energy)
    


def roundtrip_test(W, true_groundstate, true_groundstate_energy, lambda_bal):
    '''
    Round trip test checks the decoding-to-energy process using the true groundstate. 
    '''
    
    true_groundstate = true_groundstate[::-1]
    rt_energy = ising_energy(W, true_groundstate, lambda_bal)

    assert np.isclose(rt_energy, true_groundstate_energy)
    
    
        
def qaoa_results(W, true_groundstate, true_groundstate_energy, lambda_bal):
    no_of_shots = 8192
    best_counts, best_gamma, best_beta, best_qaoa_time = qaoa(W, lambda_bal, no_of_shots)
    best_config = max(best_counts, key=best_counts.get)             #This is the configuration with the highest measurement frequency.
    
    best_config = best_config[::-1]                                 #Qiskit endian correction. Reverses configuration order (not inverting)
    best_config = np.array(list(best_config), dtype=int)            #Convert string to numpy array of integers.
    
    
    roundtrip_test(W, true_groundstate, true_groundstate_energy, lambda_bal)
    
    energy_data(best_counts, W, lambda_bal, true_groundstate_energy)
    groundstate_prob = get_groundstate_prob(best_counts, true_groundstate, no_of_shots)
    
    config_energy = ising_energy(W, best_config, lambda_bal)
    rel_energy = abs((config_energy - true_groundstate_energy) / true_groundstate_energy)
    ari = ARI_check(true_groundstate, np.array([best_config]))
    
    print(f' Groundstate probability = {groundstate_prob}')
    return best_config, rel_energy, ari, best_gamma, best_beta, best_qaoa_time


    