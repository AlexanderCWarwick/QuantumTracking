import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import Aer
from ising import ising_energy, ARI_check
from plotting import plot_energy_hist
from itertools import product
import time

def qaoa_pipeline(W : np.ndarray,  lambda_bal : float,  no_of_shots : int,  seed : int,  p : int,  backend,  optimiser,  circuit,  beta,  gamma):
    '''
    The QAOA is a hybrid QC algorithm. The variational part where parameters are tweaked is controlled by the classical 
    computer.
    
    Two algorithms for this tweaking are used: Grid Search (see Week 4) and COBYLA minimisation. (Week 5) we move forward from Grid Search 
    p=1 circuit to COBYLA p >= 1.
    '''
    gamma_range = (0,np.pi)
    beta_range = (0,np.pi/2)
    
    backend.set_options(seed_simulator=seed)            
    best_gammas, best_betas, best_runtime = optimiser(W, circuit, backend, gamma, beta, lambda_bal, no_of_shots, seed, p, 
                                                        gamma_range, beta_range)
        
    best_paramed_circuit = bind_params(circuit, gamma, beta, best_gammas, best_betas, p)
    best_counts = run_qaoa(backend, best_paramed_circuit, no_of_shots)
    return best_counts, best_runtime
    
        
        

def get_counts_data(best_counts, W, true_groundstate, true_groundstate_energy, lambda_bal, no_of_shots):
    best_config = max(best_counts, key=best_counts.get)             #This is the configuration with the highest measurement frequency.
    best_config = best_config[::-1]                                 #Qiskit endian correction. Reverses configuration order (not inverting)
    best_config = np.array(list(best_config), dtype=int)            #Convert string to numpy array of integers.
    groundstate_prob = get_groundstate_prob(best_counts, true_groundstate, no_of_shots)
    
    best_config_energy = ising_energy(W, best_config, lambda_bal)
    
    best_rel_energy = abs((best_config_energy - true_groundstate_energy) / true_groundstate_energy)
    best_ari = ARI_check(true_groundstate, np.array([best_config]))        
        
    return best_config, best_rel_energy, best_ari, groundstate_prob
        
        
        
def grid(W, circuit, backend, gamma, beta, lambda_bal, no_of_shots, seed, p, gamma_range, beta_range):
    grid_counts = 3
    gamma_range = np.linspace(*gamma_range, grid_counts)
    beta_range = np.linspace(*beta_range, grid_counts)
    
    A = [gamma_range for _ in range(p)]
    B = [beta_range for _ in range(p)]
    
    best_params = None
    best_energy = np.inf
    
    grid_runtime_start = time.time()
    
    for param_state in product(*A, *B):
        gamma_values = param_state[:p]
        beta_values = param_state[p:]
        state_avg_energy = evaluate(W, circuit,  backend,  gamma,  beta,  gamma_values, beta_values, lambda_bal,  no_of_shots, p)

        if state_avg_energy < best_energy:
            best_params = param_state
            best_energy = state_avg_energy
    grid_runtime_end = time.time()
    
    return best_params[:p], best_params[p:], (grid_runtime_end - grid_runtime_start)
    
    
            
    

def cobyla(W, circuit,  backend,  gamma,  beta,  lambda_bal,  no_of_shots, seed, p, gamma_range, beta_range)  ->  tuple:

    def eval_cobyla(params):
        gamma_values = params[:p]
        beta_values = params[p:]
        return evaluate(W, circuit,  backend,  gamma,  beta,  gamma_values, beta_values, lambda_bal,  no_of_shots, p)
    
    cobyla_starts = 2
    cobyla_avg_energy = np.inf
    best_result = None
    cobyla_best_time = None
    
    rng = np.random.default_rng(seed)
    
    for _ in range(cobyla_starts):
        x0 = np.concatenate([rng.uniform(*gamma_range,p), rng.uniform(*beta_range,p)])
    
        cobyla_runtime_start = time.time()
        result = minimize(eval_cobyla, x0, method='COBYLA', options={'maxiter' : 100})
        cobyla_runtime_end = time.time()
        
        if result.fun < cobyla_avg_energy:
            cobyla_avg_energy = result.fun
            best_result = result
            cobyla_best_time = (cobyla_runtime_end - cobyla_runtime_start) 
    
    return best_result.x[:p], best_result.x[p:], cobyla_best_time
    


def bind_params(circuit, gamma, beta, gamma_values, beta_values, p):
    return circuit.assign_parameters({gamma[i]: gamma_values[i] for i in range(p)} |
                                    {beta[i]: beta_values[i] for i in range(p)})



def evaluate(W,  circuit,  backend,  gamma,  beta, gamma_values, beta_values, lambda_bal,  no_of_shots, p)  ->  float:
    '''
    Evaluation step. Binds parameter inputs to the general QAOA circuit.
    Returns the average energy of the such circuit after no_of_shots samples.
    '''
    
    paramed_circuit = bind_params(circuit, gamma, beta, gamma_values, beta_values, p)
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
    J = 2*lambda_bal - W                    #Effective coupling matrix. Equivalent to classical Ising energy.
    for layer in range(p):
        
        for i in range(N):
            for j in range(i+1, N):             #Start at i+1 since we don't want to double count the similarity measures.
                circuit.rzz(2*gamma[layer]*J[i][j], qreg_q[i], qreg_q[j])                #Cost layer. Applies RZZ gates to all connected vertices. Factor of 2 cancels the qiskit convention of a gamma/2.
        circuit.barrier() 
                    
        circuit.rx(2 * beta[layer], qreg_q)            #Mixer layer. Applies RX gates to every qubit. Allows for interference between qubit phases.
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
    
    
    
def metric_stats(metrics_dict : dict):

    metrics = metrics_dict.values()
    means = [np.mean(metric_values) for metric_values in metrics]
    std_metrics = [np.std(metric_values) for metric_values in metrics]
    
    return np.array(means), np.array(std_metrics)
    
    
    
def qaoa_results(W, true_groundstate, true_groundstate_energy, lambda_bal, no_of_shots, p, seed_lim, optimiser):
    
    metrics_dict = {'rel_errors' : [],
               'aris' : [],
               'runtimes' : [],
               'gs_prob' : []}
    
    gamma = [Parameter(f'g{i+1}') for i in range(p)]
    beta = [Parameter(f'b{i+1}') for i in range(p)]
        
    circuit = build_qaoa_circuit(W, lambda_bal, gamma, beta, p)
    
    backend = Aer.get_backend('aer_simulator')
    for seed in range(seed_lim):
        best_counts, runtime = qaoa_pipeline(W,  lambda_bal,  no_of_shots,  seed,  p,  backend,  optimiser,  circuit,  beta,  gamma)
        
        
        best_config, best_rel_energy, best_ari, gs_prob = get_counts_data(best_counts,
                                                                                    W, 
                                                                                    true_groundstate, 
                                                                                    true_groundstate_energy, 
                                                                                    lambda_bal,
                                                                                    no_of_shots)
        
        metrics_dict['rel_errors'].append(best_rel_energy)
        metrics_dict['aris'].append(best_ari)
        metrics_dict['runtimes'].append(runtime)
        metrics_dict['gs_prob'].append(gs_prob)
    return metric_stats(metrics_dict)
