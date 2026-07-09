import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import Aer
from ising import ising_energy, ARI_check
from plotting import plot_energy_hist
from itertools import product
import time

def qaoa_pipeline(W : np.ndarray, lambda_bal : float, no_of_shots : int, seed : int, p : int, circuit, gamma, beta):
    '''
    The QAOA is a hybrid QC algorithm. The variational part where parameters are tweaked is controlled by the classical 
    computer.
    
    Two algorithms for this tweaking are used: Grid Search (see Week 4) and COBYLA minimisation. (Week 5) we move forward from Grid Search 
    p=1 circuit to COBYLA p >= 1.
    '''
    
    backend = Aer.get_backend('aer_simulator')
    backend.set_options(seed_simulator=seed)
    
    
    grid_gammas, grid_betas, grid_runtime = grid_optimised_qaoa(W, circuit, backend, gamma, beta, lambda_bal, no_of_shots, p)
    best_grid_paramed_circuit = circuit.assign_parameters({gamma[i]: grid_gammas[i] for i in range(p)} |
                                                          {beta[i]: grid_betas[i] for i in range(p)})
    grid_best_counts = run_qaoa(backend, best_grid_paramed_circuit, no_of_shots)
    
    
    
    cobyla_result, cobyla_runtime = cobyla_optimised_qaoa(W,  circuit,  backend,  gamma,  beta,  lambda_bal,  no_of_shots,  p)
    cobyla_gammas = cobyla_result.x[:p]
    cobyla_betas = cobyla_result.x[p:]

    best_cobyla_paramed_circuit = circuit.assign_parameters({gamma[i]: cobyla_gammas[i] for i in range(p)} |
                                                            {beta[i]: cobyla_betas[i] for i in range(p)})
    cobyla_best_counts = run_qaoa(backend, best_cobyla_paramed_circuit, no_of_shots)
    
    return grid_best_counts, grid_runtime, cobyla_best_counts, cobyla_runtime
        
        

def get_counts_data(best_counts, W, true_groundstate, true_groundstate_energy, lambda_bal, no_of_shots):
    best_config = max(best_counts, key=best_counts.get)             #This is the configuration with the highest measurement frequency.
    best_config = best_config[::-1]                                 #Qiskit endian correction. Reverses configuration order (not inverting)
    best_config = np.array(list(best_config), dtype=int)            #Convert string to numpy array of integers.
    groundstate_prob = get_groundstate_prob(best_counts, true_groundstate, no_of_shots)
    
    best_config_energy = ising_energy(W, best_config, lambda_bal)
    
    best_rel_energy = abs((best_config_energy - true_groundstate_energy) / true_groundstate_energy)
    best_ari = ARI_check(true_groundstate, np.array([best_config]))        
        
    return best_config, best_rel_energy, best_ari, groundstate_prob
        
        
        
def grid_optimised_qaoa(W, circuit, backend, gamma, beta, lambda_bal, no_of_shots, p):
    grid_counts = 3
    gamma_range = np.linspace(0, np.pi, grid_counts)
    beta_range = np.linspace(0, np.pi/2, grid_counts)
    
    A = [gamma_range for _ in range(p)]
    B = [beta_range for _ in range(p)]
    
    product_space = list(product(*A, *B))
    
    best_params = None
    best_energy = np.inf
    grid_runtime_start = time.time()
    for param_state in product_space:
        gamma_values = param_state[:p]
        beta_values = param_state[p:]
        state_avg_energy = evaluate(W, circuit,  backend,  gamma,  beta,  gamma_values, beta_values, lambda_bal,  no_of_shots, p)

        if state_avg_energy < best_energy:
            best_params = param_state
            best_energy = state_avg_energy
    grid_runtime_end = time.time()
    
    return best_params[:p], best_params[p:], (grid_runtime_end - grid_runtime_start)
    
    
            
    

def cobyla_optimised_qaoa(W, circuit,  backend,  gamma,  beta,  lambda_bal,  no_of_shots, p)  ->  tuple:

    def eval_cobyla(params):
        gamma_values = params[:p]
        beta_values = params[p:]
        return evaluate(W, circuit,  backend,  gamma,  beta,  gamma_values, beta_values, lambda_bal,  no_of_shots, p)
    
    cobyla_starts = 2
    cobyla_avg_energy = np.inf
    best_result = None
    cobyla_best_time = None
    for _ in range(cobyla_starts):
        x0 = np.concatenate([np.random.uniform(0,np.pi,p), np.random.uniform(0,np.pi/2,p)])
    
        cobyla_runtime_start = time.time()
        result = minimize(eval_cobyla, x0, method='COBYLA', options={'maxiter' : 50})
        cobyla_runtime_end = time.time()
        
        if result.fun < cobyla_avg_energy:
            cobyla_avg_energy = result.fun
            best_result = result
            cobyla_best_time = (cobyla_runtime_end - cobyla_runtime_start) 
    
    return best_result, cobyla_best_time
    


def evaluate(W,  circuit,  backend,  gamma,  beta, gamma_values, beta_values, lambda_bal,  no_of_shots, p)  ->  float:
    '''
    Evaluation step. Binds parameter inputs to the general QAOA circuit.
    Returns the average energy of the such circuit after no_of_shots samples.
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
    
    
def metric_stats(rel_energies, aris, runtimes, groundstate_probs):
    metrics = [rel_energies, aris, runtimes, groundstate_probs]
    means = [np.mean(metric) for metric in metrics]
    std_metrics = [np.mean(metric) for metric in metrics]

    return np.array(means), np.array(std_metrics)
        
        
        
def qaoa_results(W, true_groundstate, true_groundstate_energy, lambda_bal):
    no_of_shots = 100
    p = 1           #Number of layers
    seed_lim = 2
    
    cobyla_configs =[]
    cobyla_rel_energies = []
    cobyla_aris = []
    cobyla_runtimes = []
    cobyla_groundstate_probs = []
    
    grid_configs =[]
    grid_rel_energies = []
    grid_aris = []
    grid_runtimes = []
    grid_groundstate_probs = []
    
    gamma = [Parameter(f'g{i+1}') for i in range(p)]
    beta = [Parameter(f'b{i+1}') for i in range(p)]
        
    circuit = build_qaoa_circuit(W, lambda_bal, gamma, beta, p)
    
    for seed in range(seed_lim):
        grid_best_counts, grid_runtime, cobyla_best_counts, cobyla_runtime = qaoa_pipeline(W, 
                                                                                           lambda_bal, 
                                                                                           no_of_shots, 
                                                                                           seed,
                                                                                           p, 
                                                                                           circuit, gamma, beta)
        
        grid_best_config, grid_best_rel_energy, grid_best_ari, grid_groundstate_prob = get_counts_data(grid_best_counts,
                                                                                                        W, 
                                                                                                        true_groundstate, 
                                                                                                        true_groundstate_energy, 
                                                                                                        lambda_bal, 
                                                                                                        no_of_shots)
        
        cobyla_best_config, cobyla_best_rel_energy, cobyla_best_ari, cobyla_groundstate_prob = get_counts_data(cobyla_best_counts,
                                                                                                        W, 
                                                                                                        true_groundstate, 
                                                                                                        true_groundstate_energy, 
                                                                                                        lambda_bal, 
                                                                                                        no_of_shots)
        
        grid_configs.append(grid_best_config)
        grid_aris.append(grid_best_ari)
        grid_rel_energies.append(grid_best_rel_energy)
        grid_runtimes.append(grid_runtime)
        grid_groundstate_probs.append(grid_groundstate_prob)
        
          
        cobyla_configs.append(cobyla_best_config)
        cobyla_aris.append(cobyla_best_ari)
        cobyla_rel_energies.append(cobyla_best_rel_energy)
        cobyla_runtimes.append(cobyla_runtime)
        cobyla_groundstate_probs.append(cobyla_groundstate_prob)
        
        
    grid_means, grid_stds = metric_stats(np.array(grid_rel_energies),
                                         np.array(grid_aris),
                                         np.array(grid_runtimes),
                                         np.array(grid_groundstate_probs))
        
    cobyla_means, cobyla_stds = metric_stats(np.array(cobyla_rel_energies), 
                                             np.array(cobyla_aris),
                                             np.array(cobyla_runtimes), 
                                             np.array(cobyla_groundstate_probs))
    
    return (cobyla_configs, cobyla_means[0], cobyla_means[1], cobyla_means[2], cobyla_means[3], 
            cobyla_stds[0], cobyla_stds[1], cobyla_stds[2], cobyla_stds[3])



    