import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import Aer
from ising import ising_energy, ARI_check
from plotting import plot_energy_hist
import time

def qaoa(W : np.ndarray,  lambda_bal : float, no_of_shots : int) -> tuple[tuple, tuple]:
    '''
    The QAOA is a hybrid QC algorithm. The variational part where parameters are tweaked is controlled by the classical 
    computer.
    
    Two algorithms for this tweaking are used: Grid Search and COBYLA minimisation.
    '''
    
    backend = Aer.get_backend('aer_simulator')
    backend.set_options(seed_simulator=44)
    
    gamma = Parameter('g')
    beta = Parameter('b')
    circuit = build_qaoa_circuit(W, lambda_bal, gamma, beta, p)
    
    grid_best_energy, grid_best_gamma, grid_best_beta, grid_best_counts, grid_runtime = grid_optimised_qaoa(W, circuit,  backend,  gamma, beta,  lambda_bal,  no_of_shots)
    
    cobyla_best_gamma, cobyla_best_beta, cobyla_best_energy, cobyla_best_counts, cobyla_runtime = cobyla_optimised_qaoa(W, circuit,  backend,  gamma,  beta,  lambda_bal,  no_of_shots)
    

    grid_results = grid_best_energy, grid_best_gamma, grid_best_beta, grid_best_counts, grid_runtime
    cobyla_results = cobyla_best_gamma, cobyla_best_beta, cobyla_best_energy, cobyla_best_counts, cobyla_runtime
    
    return grid_results, cobyla_results
    



def cobyla_optimised_qaoa(W, circuit,  backend,  gamma,  beta,  lambda_bal,  no_of_shots):

    def eval_cobyla(params):
        gamma_value, beta_value = params
        return evaluate(W, circuit,  backend,  gamma,  beta,  gamma_value, beta_value, lambda_bal,  no_of_shots)
    
    x0 = [0.5, 0.5]
    cobyla_runtime_start = time.time()
    result = minimize(eval_cobyla, x0, method='COBYLA', options={'maxiter' : 50})
    cobyla_runtime_end = time.time()

    cobyla_best_gamma = result.x[0]
    cobyla_best_beta = result.x[1]
    cobyla_best_energy = result.fun
    
    paramed_circuit = circuit.assign_parameters({gamma: cobyla_best_gamma, beta: cobyla_best_beta})  
    cobyla_best_counts = run_qaoa(backend, paramed_circuit, no_of_shots)
    
    return cobyla_best_gamma, cobyla_best_beta, cobyla_best_energy, cobyla_best_counts, (cobyla_runtime_end - cobyla_runtime_start)
    
    
    

def grid_optimised_qaoa(W, circuit,  backend,  gamma,  beta,  lambda_bal,  no_of_shots):
    '''
    Grid search optimisation. Compare to COBYLA.
    '''
    best_energy = np.inf
    grid_best_gamma = 0
    grid_best_beta = 0
    grid_counts = 20
    gamma_range = np.linspace(0, np.pi, grid_counts)
    beta_range = np.linspace(0, np.pi/2, grid_counts)
    
    grid_runtime_start = time.time()
    for gamma_value in gamma_range:              #Grid search optimisation for tuning parameters beta and gamma.
        for beta_value in beta_range:
            avg_energy = evaluate(W,  circuit,  backend,  gamma,  beta,  gamma_value, beta_value, lambda_bal,  no_of_shots)   
                        
            if avg_energy < best_energy:
                #If this 'shot' measures a better expectation value of the energy, it replaces the previous best shot.
                grid_best_energy = avg_energy
                grid_best_gamma = gamma_value
                grid_best_beta = beta_value
    grid_runtime_end = time.time() 
    
    paramed_circuit = circuit.assign_parameters({gamma: grid_best_gamma, beta: grid_best_beta})  
    grid_best_counts = run_qaoa(backend, paramed_circuit, no_of_shots)
    
    return grid_best_energy, grid_best_gamma, grid_best_beta, grid_best_counts, (grid_runtime_end - grid_runtime_start)
                


def evaluate(W,  circuit,  backend,  gamma,  beta, gamma_value, beta_value, lambda_bal,  no_of_shots)  ->  float:
    '''
    Evaluation step. Returns the average energy and the counts of the current grid search point.
    '''
    paramed_circuit = circuit.assign_parameters({gamma: gamma_value, beta: beta_value})          
    counts = run_qaoa(backend, paramed_circuit, no_of_shots)
            
    avg_energy = 0 
    for rev_config, count in counts.items():
        config = rev_config[::-1]                         #Corrects for qiskit endian convention (qubits are ordered in reverse)  
        config = np.array(list(config), dtype=int)        #counts is a dictionary of bitstrings and their corresponding frequencies. The bitstrings are given as strings so convert to a np array
        config_energy = ising_energy(W, config, lambda_bal)               
        avg_energy += config_energy * (count / no_of_shots)
        
    return avg_energy



def build_qaoa_circuit(W, lambda_bal, gamma, beta):
    '''
    Builds the p=1 QAOA circuitw using the generalised parameters. 
    Only one circuit is ever built, only the RZZ and RX gate input angle parameters change. 
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
    no_of_shots = 4096
    p = 1           #Number of layers
    grid_results, cobyla_results = qaoa(W, lambda_bal, no_of_shots)
    
    
    
    '''
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
    return best_config, rel_energy, ari, best_gamma, best_beta, 0
    '''


    