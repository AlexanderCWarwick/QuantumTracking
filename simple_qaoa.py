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
    '''
    Input: A sample of the pdf from a circuit, best_counts. In total no_of_shots independent measurements made.
    Output: ari, rel_energy and gsp. best_config is found from which configuration was the most sampled (highest frequency)
    '''
    
    best_config = max(best_counts, key=best_counts.get)             #This is the configuration with the highest measurement frequency.
    best_config = best_config[::-1]                                 #Qiskit endian correction. Reverses configuration order (not inverting)
    best_config = np.array(list(best_config), dtype=int)            #Convert string to numpy array of integers.
    groundstate_prob = get_groundstate_prob(best_counts, true_groundstate, no_of_shots)
    
    best_config_energy = ising_energy(W, best_config, lambda_bal)
    
    best_rel_energy = abs((best_config_energy - true_groundstate_energy) / true_groundstate_energy)
    best_ari = ARI_check(true_groundstate, np.array([best_config]))        
        
    return best_config, best_rel_energy, best_ari, groundstate_prob
        
        
        
def grid(W, circuit, backend, gamma, beta, lambda_bal, no_of_shots, _seed, p, gamma_lims, beta_lims):
    '''
    Basic iterative search in hypercuboid of 2p dimensional parameter space. 
    seed is unused here but is needed for general optimiser call in qaoa function.
    '''
    
    grid_counts = 5                                             #Number of points along each axes to sample from. In total 2*2p points.
    gamma_range = np.linspace(*gamma_lims, grid_counts)
    beta_range = np.linspace(*beta_lims, grid_counts)
     
    A = [gamma_range for _ in range(p)]
    B = [beta_range for _ in range(p)]
    #A and B are the subspaces of the parameter space A x B. 
    
    
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
    '''
    COBYLA optimised QAOA. 
    COBYLA minimisation does not use the gradient (since we don't know the ising hmailtonian gradient)
    
    Instead, working through the evaluate function output space (the avg energy of the returned sampled distribution)
    it uses a shrinking trust region to estimate better values for the tuning parameters to get a better estimate.
    ''' 
    
    def eval_cobyla(params):
        gamma_values = params[:p]
        beta_values = params[p:]
        return evaluate(W, circuit,  backend,  gamma,  beta,  gamma_values, beta_values, lambda_bal,  no_of_shots, p)
    
    cobyla_starts = 5                       #Number of random restarts
    cobyla_avg_energy = np.inf
    best_result = None
    cobyla_best_time = None
    
    rng = np.random.default_rng(seed)
    
    for i in range(cobyla_starts):
        x0 = np.concatenate([rng.uniform(*gamma_range,p), rng.uniform(*beta_range,p)])
    
        cobyla_runtime_start = time.time()
        result = minimize(eval_cobyla, x0, method='COBYLA', options={'maxiter' : 250})
        cobyla_runtime_end = time.time()
        
        if result.fun < cobyla_avg_energy:
            cobyla_avg_energy = result.fun
            best_result = result
            cobyla_best_time = (cobyla_runtime_end - cobyla_runtime_start)
        print(i) 
    return best_result.x[:p], best_result.x[p:], cobyla_best_time
    


def bind_params(circuit, gamma, beta, gamma_values, beta_values, p):
    '''
    Binds parameter values to the gates as in build_qaoa_circuit.
    '''
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
    
    We are using the RBF similarity matrix so we can simply iterate over the entire (upper right triangle) matrix when building
    the cost layers.
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
    '''
    Input: The simulator standing in for the quantum computer, the binded circuit and the number of shots.
    Output: The sampled probability distribution for that circuit with those specific parameter values.
    '''
    
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
  
    
    
def metric_stats(metrics_dict : dict) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    '''
    Input: metrics_dict contains lists of seed_lim values for each metric.
    Compute means and errors for each different metric given
    Output: Two seperate arrays for means and errors for each metric. The ordering is kept the same as the dictionary.
    relative energy -> ari -> runtime -> gsp
    '''
    
    metrics = metrics_dict.values()
    means = [np.mean(metric_values) for metric_values in metrics]
    std_metrics = [np.std(metric_values) for metric_values in metrics]
    
    return np.array(means), np.array(std_metrics)
    
    
    
def qaoa_results(W : np.ndarray[float],  true_groundstate : np.ndarray[int], 
                 true_groundstate_energy : float,  lambda_bal : float, 
                 no_of_shots : int,  p : int,  seed_lim : int,  optimiser) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    '''
    Build the generalised circuit wih parameters gamma and beta (for each layer). Each time we generate parameter values
    e.g. iterating through points in the grid search or adaptive optimiser finds a new parameter set, we bind them to the circuit.
    
    Then for seed_lim different random shot noise seeds, obtain means and errors for all metrics. 
    Order of metrics is the same throughout this code:
    - relative energy error = (estimated gs energy - true_gs_energy) / true_gs_energy 
    - ari = ari of the estimate gs configuration
    - runtime = total runtime of the algorithm (can be split for the qaoa since it's hybrid)
    - gsp = raw gs probability. Depth scan will use gs prob relative to the uniformly random probability (2 / (2^N)). 
    '''
    
    metrics_dict = {'rel_error' : [],
               'ari' : [],
               'runtime' : [],
               'gsp' : []}
    
    gamma = [Parameter(f'g{i+1}') for i in range(p)]
    beta = [Parameter(f'b{i+1}') for i in range(p)]
        
    circuit = build_qaoa_circuit(W, lambda_bal, gamma, beta, p)
    
    backend = Aer.get_backend('aer_simulator')
    for seed in range(seed_lim):
        best_counts, runtime = qaoa_pipeline(W,  lambda_bal,  no_of_shots,  seed,  p,  backend,  optimiser,  circuit,  beta,  gamma)
        
        
        _, best_rel_energy, best_ari, gs_prob = get_counts_data(best_counts,
                                                                                    W, 
                                                                                    true_groundstate, 
                                                                                    true_groundstate_energy, 
                                                                                    lambda_bal,
                                                                                    no_of_shots)
        
        metrics_dict['rel_error'].append(best_rel_energy)
        metrics_dict['ari'].append(best_ari)
        metrics_dict['runtime'].append(runtime)
        metrics_dict['gsp'].append(gs_prob)
    return metric_stats(metrics_dict)
