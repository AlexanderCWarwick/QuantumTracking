import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from ising import ising_energy, ARI_check
import time
    
def maxcut_value(bitstring, G):
    value = 0
    for i,j,weight in G.edges(data='weight'):
        if bitstring[i] != bitstring[j]:
            value += weight            
    return value


def qaoa(hit_coords : np.ndarray,  G,  w : np.ndarray,  lambda_bal : float):
    N = len(hit_coords)
    qreg_q = QuantumRegister(N, 'q')
    creg_c = ClassicalRegister(N, 'c')
    grid_counts = 10
    no_of_shots = 1000
    backend = Aer.get_backend('aer_simulator')
    
    best_cut = 0
    best_gamma = 0
    best_beta = 0
    best_counts = 0
        
    start = time.time()
    
    for gamma in np.linspace(-np.pi, np.pi, grid_counts):
        for beta in np.linspace(-np.pi, np.pi, grid_counts):
            circuit = QuantumCircuit(qreg_q, creg_c)
            
            circuit.h(qreg_q)
            
            for i in range(N):
                for j in range(i+1, N):
                    J = 2*lambda_bal - w[i][j]
                    circuit.rzz(2*gamma*J, qreg_q[i], qreg_q[j])
                    
            circuit.rx(2 * beta, qreg_q)
            
            for i in range(N):
                circuit.measure(qreg_q[i], creg_c[i])
                    
            result = backend.run(circuit, shots=no_of_shots).result()
            counts = result.get_counts()
            
            avg_cut_value = 0 
            for partition, count in counts.items():
                partition = partition[::-1]                         #Corrects for qiskit endian convention (qubits are ordered in reverse)         
                cut_value = maxcut_value(partition, G)              #Evaluate how good the cut is. 
                avg_cut_value += cut_value * (count / no_of_shots)      #Average cut value sum
                
            if avg_cut_value > best_cut:
                best_cut = avg_cut_value
                best_gamma = gamma
                best_beta = beta
                best_counts = counts
    
    end = time.time()
    
    return best_counts, best_gamma, best_beta, (end-start)



def qaoa_results(hit_coords, W, true_groundstate, true_groundstate_energy, lambda_bal, G):
    best_counts, best_gamma, best_beta, time = qaoa(hit_coords, G, W, lambda_bal)
    config = max(best_counts, key=best_counts.get)
    config = np.array(list(config), dtype=int)
    
    config_energy = ising_energy(W, config, lambda_bal)
    rel_energy = abs((config_energy - true_groundstate_energy) / true_groundstate_energy)
    ari = ARI_check(true_groundstate, np.array([config]))
    
    return config, rel_energy, ari, best_gamma, best_beta, time


    