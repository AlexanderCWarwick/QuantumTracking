import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import Aer
from ising import ising_energy, ARI_check
from plotting import plot_energy_hist
import time

def qaoa(W : np.ndarray,  lambda_bal : float, no_of_shots : int, p : int) -> tuple[float, float, float, dict, float]:
    '''
    The QAOA is a hybrid QC algorithm. The variational part where parameters are tweaked is controlled by the classical 
    computer.
    
    Two algorithms for this tweaking are used: Grid Search and COBYLA minimisation. (Week 5) we move forward from Grid Search 
    p=1 circuit to COBYLA p >= 1.
    '''
    
    backend = Aer.get_backend('aer_simulator')
    backend.set_options(seed_simulator=44)
    
    gamma = [Parameter(f'g{i+1}') for i in range(p)]
    beta = [Parameter(f'b{i+1}') for i in range(p)]
    
    
    circuit = build_qaoa_circuit(W, lambda_bal, gamma, beta, p)
    
    return cobyla_optimised_qaoa(W,  circuit,  backend,  gamma,  beta,  lambda_bal,  no_of_shots,  p)    



def cobyla_optimised_qaoa(W, circuit,  backend,  gamma,  beta,  lambda_bal,  no_of_shots, p):

    def eval_cobyla(params):
        gamma_values = params[:p]
        beta_values = params[p:]
        return evaluate(W, circuit,  backend,  gamma,  beta,  gamma_values, beta_values, lambda_bal,  no_of_shots, p)
    
    x0 = np.concatenate([np.random.uniform(0,np.pi,p), np.random.uniform(0,np.pi/2,p)])
    
    cobyla_runtime_start = time.time()
    result = minimize(eval_cobyla, x0, method='COBYLA', options={'maxiter' : 50})
    cobyla_runtime_end = time.time()

    cobyla_best_gammas = result.x[:p]
    cobyla_best_betas = result.x[p:]
    cobyla_best_energy = result.fun
    
    best_paramed_circuit = circuit.assign_parameters({gamma[i]: cobyla_best_gammas[i] for i in range(p)} |
                                                     {beta[i]: cobyla_best_betas[i] for i in range(p)})
    cobyla_best_counts = run_qaoa(backend, best_paramed_circuit, no_of_shots)
    
    return cobyla_best_gammas, cobyla_best_betas, cobyla_best_energy, cobyla_best_counts, (cobyla_runtime_end - cobyla_runtime_start)
    


def evaluate(W,  circuit,  backend,  gamma,  beta, gamma_values, beta_values, lambda_bal,  no_of_shots, p)  ->  float:
    '''
    Evaluation step. Returns the average energy and the counts of the current grid search point.
    '''

    paramed_circuit = circuit.assign_parameters({gamma[i]: gamma_values[i] for i in range(p)} |
                                                {beta[i]: beta_values[i] for i in range(p)})
     
    counts = run_qaoa(backend, paramed_circuit, no_of_shots)
            
    avg_energy = 0 
    for rev_config, count in counts.items():
        config = rev_config[::-1]                         #Corrects for qiskit endian convention (qubits are ordered in reverse)  
        config = np.array(list(config), dtype=int)        #counts is a dictionary of bitstrings and their corresponding frequencies. The bitstrings are given as strings so convert to a np array
        config_energy = ising_energy(W, config, lambda_bal)               
        avg_energy += config_energy * (count / no_of_shots)
        
    return avg_energy



def build_qaoa_circuit(W, lambda_bal, gamma, beta, p):
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
    for layer in range(p):
        
        for i in range(N):
            for j in range(i+1, N):             #Start at i+1 since we don't want to double count the similarity measures.

                J = 2*lambda_bal - W[i][j]                                  #Effective coupling matrix. Equivalent to classical Ising energy.
                circuit.rzz(2*gamma[layer]*J, qreg_q[i], qreg_q[j])                #Cost layer. Applies RZZ gates to all connected vertices. Factor of 2 cancels the qiskit convention of a gamma/2.
        circuit.barrier() 
                    
        circuit.rx(2 * beta[layer], qreg_q)            #Mixer layer. Applies RX gates to every qubit. Allows for interference between qubit phases.
        circuit.barrier()   
                 
    circuit.measure(qreg_q, creg_c)
    
    circuit.draw('mpl')
    plt.show()
    
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
    p = 2           #Number of layers
    cobyla_best_gammas, cobyla_best_betas, cobyla_best_energy, cobyla_best_counts, cobyla_runtime = qaoa(W, lambda_bal, no_of_shots, p)
    
    cobyla_best_config = max(cobyla_best_counts, key=cobyla_best_counts.get)             #This is the configuration with the highest measurement frequency.
    
    cobyla_best_config = cobyla_best_config[::-1]                                 #Qiskit endian correction. Reverses configuration order (not inverting)
    cobyla_best_config = np.array(list(cobyla_best_config), dtype=int)            #Convert string to numpy array of integers.
    groundstate_prob = get_groundstate_prob(cobyla_best_counts, true_groundstate, no_of_shots)
    
    cobyla_best_config_energy = ising_energy(W, cobyla_best_config, lambda_bal)
    cobyla_best_rel_energy = abs((cobyla_best_config_energy - true_groundstate_energy) / true_groundstate_energy)
    cobyla_best_ari = ARI_check(true_groundstate, np.array([cobyla_best_config]))
    energy_data(cobyla_best_counts, W, lambda_bal, true_groundstate_energy)
    
    print(groundstate_prob)
    return cobyla_best_config, cobyla_best_rel_energy, cobyla_best_ari, cobyla_best_gammas, cobyla_best_betas, cobyla_runtime 
    


    