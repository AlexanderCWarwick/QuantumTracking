import numpy as np
from scipy.optimize import minimize

from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import QiskitRuntimeService

from qaoa.add_noisemodel import make_noise_model
from qaoa.qaoa_circuit import build_qaoa_circuit

from classical.ising import ising_energy, ARI_check
from plotting.qaoa_energy_plots import (plot_energy_hist) #optimiser_energy_trace, optimiser_result_energies, top_ten_states
from itertools import product
from time import perf_counter

def qaoa_pipeline(W : np.ndarray,  
                  lambda_bal : float,  
                  no_of_shots : int,  
                  seed : int,  
                  p : int,  
                  backend,  
                  optimiser,  
                  circuit,  
                  beta,  
                  gamma, 
                  warm_restart):
    '''
    The QAOA is a hybrid QC algorithm. The variational part where parameters are tweaked is controlled by the classical 
    computer.
    
    Two algorithms for this tweaking are used: Grid Search (see Week 4) and COBYLA minimisation. (Week 5) we move forward from Grid Search 
    p=1 circuit to COBYLA p >= 1.
    '''
    gamma_range = (0, 2*np.pi)
    beta_range = (0, np.pi)
    
    backend.set_options(seed_simulator=seed)                #To be used if the Aersimulator backend is used.         
                    
    best_gammas, best_betas, best_runtime, best_energy = optimiser(W, 
                                                      circuit, 
                                                      backend, 
                                                      gamma, 
                                                      beta, 
                                                      lambda_bal, 
                                                      no_of_shots, 
                                                      seed, 
                                                      p, 
                                                      gamma_range, 
                                                      beta_range, 
                                                      warm_restart)
        
    best_paramed_circuit = bind_params(circuit, gamma, beta, best_gammas, best_betas, p)
    best_counts = run_qaoa(backend, best_paramed_circuit, no_of_shots)
    
    return best_counts, best_gammas, best_betas, best_runtime, best_energy
    
        
        
def get_counts_data(best_counts, 
                    W, 
                    true_groundstate, 
                    true_groundstate_energy, 
                    lambda_bal, 
                    no_of_shots):
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
        

def grid(W, 
         circuit, 
         backend,  
         gamma, 
         beta, 
         lambda_bal, 
         no_of_shots, 
         _seed, 
         p, 
         gamma_lims, 
         beta_lims, 
         _warm_restart):
    
    '''
    Basic iterative search in hypercuboid of 2p dimensional parameter space. 
    seed is unused here but is needed for general optimiser call in qaoa function.
    '''
    
    grid_counts = 10                                #Number of points along each parameter axes to sample from. In total 2*2p points.
    gamma_range = np.linspace(*gamma_lims, grid_counts)
    beta_range = np.linspace(*beta_lims, grid_counts)
     
    A = [gamma_range for _ in range(p)]
    B = [beta_range for _ in range(p)]
    #A and B are the subspaces of the parameter space A x B. 
    
    best_params = None
    best_energy = np.inf
    
    grid_runtime_start = perf_counter()
    
    for param_state in product(*A, *B):
        gamma_values = param_state[:p]
        beta_values = param_state[p:]
        state_avg_energy = evaluate(W, circuit,  backend,  gamma,  beta,  gamma_values, beta_values, lambda_bal,  no_of_shots, p)

        if state_avg_energy < best_energy:
            best_params = param_state
            best_energy = state_avg_energy
            
    grid_runtime = perf_counter() - grid_runtime_start
    return best_params[:p], best_params[p:], grid_runtime, best_energy

    
def expand_warm_start(warm_start, 
                      p, 
                      gamma_range, 
                      beta_range, 
                      rng):
    '''
    minimise function expects parameters in order [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
    not [gamma_1, beta_1, ... gamma_p, beta_p]. 
    '''
    
    if warm_start is None:
        return None

    old_p = len(warm_start) // 2

    old_gammas = warm_start[:old_p]
    old_betas = warm_start[old_p:]

    new_gamma = rng.uniform(*gamma_range) * 0.1             #New params are close to 0. (p)th layer can then use the (p-1)th layer solution
    new_beta = rng.uniform(*beta_range) * 0.1               #and make use of the warm restart.

    return np.concatenate([old_gammas, [new_gamma], old_betas, [new_beta]])  
      
            
def cobyla(W, 
           circuit,  
            backend,   
            gamma,  
            beta,  
            lambda_bal,  
            no_of_shots, 
            seed, 
            p, 
            gamma_range, 
            beta_range,
            warm_restart)  ->  tuple:
    '''
    COBYLA/COBYQA optimised QAOA. 
    COBYLA/COBYQA minimisation does not use the gradient (since we don't know the ising hmailtonian gradient)
    
    COBYQA uses quadratic approximation, COBYLA is only linear. 
    Expect COBYQA to perform better in ARI and relative energy error but slower.
    
    Instead, working through the evaluate function output space (the avg energy of the returned sampled distribution)
    it uses a shrinking trust region to estimate better values for the tuning parameters to get a better estimate.
    
    Warm restarts now use the best parameters from the previous p scan as a new starting point. 
    Example: If we have 3 restarts and are (depth) scanning over p=[1,2] , the pattern is: 
    p = 1
    restart:
    1 - random
    2 - random
    3 - random
    
    p = 2
    restart:
    1 - warmstart using best p=1 gamma_1 and beta_1 values.
    2 - random
    3 - random
    ''' 
    restarts = 4                       #Number of random restarts
    best_avg_energy = np.inf
    best_result = None
    
    histories = []                       #List to hold the evolution of each random restarts energy. 
    final_energies = []                  #List to hold the returned best energies from each restart.
    
    param_bounds = [gamma_range] * p + [beta_range] * p
    rng = np.random.default_rng(seed)
    
    seed_runtime_start = perf_counter()
    
    for restart_idx in range(restarts):
        if restart_idx == 0 and warm_restart is not None:
            #First restart = warm start, the rest callare normal random restarts.
            x0 = expand_warm_start(warm_restart, p, gamma_range, beta_range, rng)
        else:
            # Remaining restarts = random
            x0 = np.concatenate([rng.uniform(*gamma_range, p), rng.uniform(*beta_range, p)])
            
        restart_energies = []
        
        def eval(params):
            gamma_values = params[:p]
            beta_values = params[p:]
            energy = evaluate(W, circuit,  backend,  gamma,  beta,  gamma_values, beta_values, lambda_bal,  no_of_shots, p)
            
            restart_energies.append(energy)
            return energy
    
        result = minimize(eval, x0, method='COBYLA', bounds=param_bounds, options={'maxiter' : 100})
        result_energy = result.fun
    
        if result_energy < best_avg_energy:
            best_avg_energy = result_energy
            best_result = result
            final_energies.append(result_energy)  
        
        else:
            final_energies.append(best_avg_energy) 
            
        if restart_idx == 0 and warm_restart is not None:
            print(f'Warm Restart {restart_idx+1} / {restarts} complete')
        else:
            print(f'Restart {restart_idx+1} / {restarts} complete')
        histories.append(restart_energies)
    
    seed_time = perf_counter() - seed_runtime_start
    #Plot energy trace of each restart.
    #optimiser_energy_trace(restarts, histories, best_history_idx, method, p, len(W))

    #Plot how the cobyla_avg_energy changes through the cobyla_restarts number of repitions.
    #optimiser_result_energies(final_energies, method, p, len(W))

    return best_result.x[:p], best_result.x[p:], seed_time, best_avg_energy
    
    
def bind_params(circuit, 
                gamma, 
                beta,
                gamma_values, 
                beta_values, 
                p):
    '''
    Binds parameter values to the gates as in build_qaoa_circuit.
    '''
    return circuit.assign_parameters({gamma[i]: gamma_values[i] for i in range(p)} |
                                    {beta[i]: beta_values[i] for i in range(p)})

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

def bind_params(circuit, 
                gamma, 
                beta,
                gamma_values, 
                beta_values, 
                p):
    '''
    Binds parameter values to the gates as in build_qaoa_circuit.
    '''
    return circuit.assign_parameters({gamma[i]: gamma_values[i] for i in range(p)} |
                                    {beta[i]: beta_values[i] for i in range(p)})
    
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
    
    
    
def qaoa_results(W : np.ndarray[float],  
                 true_groundstate : np.ndarray[int], 
                 true_groundstate_energy : float,  
                 lambda_bal : float, 
                 no_of_shots : int,  
                 p : int,  
                 seed_lim : int,  
                 optimiser,
                 warm_restart : np.ndarray[float],
                 noise_strengths : tuple[np.float64, np.float64],
                 readout_prob : float) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
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
        
        
    
    noise_model = make_noise_model(*noise_strengths, 
                                   readout_prob)
    print(noise_model)
    
    service = QiskitRuntimeService(instance='Warwick-flex')
    
    hardware_backend = service.backend('ibm_miami')
    
    sim_backend = AerSimulator(noise_model=noise_model)
    
    
    circuit = build_qaoa_circuit(W, lambda_bal, gamma, beta, p)
    transpiled_circuit = transpile(circuit,
                                    backend=hardware_backend)
    
    #transpiled_qc_depth = sum(list(transpiled_circuit.count_ops().values()))
    transpiled_depth = transpiled_circuit.depth()
    print(f'All-to-all Circuit Depth = {circuit.depth()}')
    print(f'Transpiled Depth = {transpiled_circuit.depth()}')


    best_seed_energy = np.inf
    best_seed_gammas, best_seed_betas = None, None
    
    for seed in range(seed_lim):
        seed_start_time = perf_counter()
        best_counts, best_gammas, best_betas, runtime, seed_energy = qaoa_pipeline(W,  
                                                                                    lambda_bal,  
                                                                                    no_of_shots,  
                                                                                    seed,  
                                                                                    p,
                                                                                    sim_backend,
                                                                                    optimiser,  
                                                                                    transpiled_circuit, 
                                                                                    beta,  
                                                                                    gamma,  
                                                                                    warm_restart)
        if seed_energy < best_seed_energy:
            best_seed_energy = seed_energy
            best_seed_gammas, best_seed_betas = best_gammas, best_betas
            
        
        #energy_data(best_counts, W, lambda_bal, true_groundstate_energy)
        
        #top_ten_states(best_counts, true_groundstate)
                
        
        
        _, best_rel_energy, best_ari, gs_prob = get_counts_data(best_counts,
                                                                W, true_groundstate, true_groundstate_energy, lambda_bal,
                                                                no_of_shots)
        
        metrics_dict['rel_error'].append(best_rel_energy)
        metrics_dict['ari'].append(best_ari)
        metrics_dict['runtime'].append(runtime)
        metrics_dict['gsp'].append(gs_prob)
        print(f'Seed {seed + 1} / {seed_lim} complete: ({(perf_counter() - seed_start_time):.2f}s)')
        print('\n')
        
    return *metric_stats(metrics_dict), best_seed_gammas, best_seed_betas, transpiled_depth
