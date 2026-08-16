import numpy as np
import matplotlib.pyplot as plt
from global_params import gamma_range, beta_range

from qiskit.visualization import plot_histogram
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

from noise.add_noisemodel import make_noise_model
from circuits.qaoa_circuit import build_qaoa_circuit, bind_params
from analysis.counts_analysis import get_counts_data#, energy_data_plot
from analysis.metric_data import metric_stats
from plotting.qaoa_energy_plots import top_ten_states
from qaoa.q_ising_energy import run_qaoa
from circuits.transpiler import transpile_circuit

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
                  warm_restart,
                  restarts : int):
    '''
    The QAOA is a hybrid QC algorithm. The variational part where parameters are tweaked is controlled by the classical 
    computer.
    
    Two algorithms for this tweaking are used: Grid Search (see Week 4) and COBYLA minimisation. (Week 5) we move forward from Grid Search 
    p=1 circuit to COBYLA p >= 1.
    '''
    
    backend.set_options(seed_simulator=seed)                #Sets the seed to fix sampling output.         
                    
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
                                                                warm_restart,
                                                                restarts)
    
    best_paramed_circuit = bind_params(circuit, gamma, beta, best_gammas, best_betas, p)
    best_counts = run_qaoa(backend, best_paramed_circuit, no_of_shots)
    
    return best_counts, best_gammas, best_betas, best_runtime, best_energy
    
    
    
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
                 readout_prob : float,
                 restarts : int,
                 backend) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
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
    
    
    sim_backend = define_sim_backend(noise_strengths, 
                                    readout_prob)
    
    
    gamma = [Parameter(f'g{i+1}') for i in range(p)]
    beta = [Parameter(f'b{i+1}') for i in range(p)]
    ata_circuit = build_qaoa_circuit(W, lambda_bal, gamma, beta, p)
    transpiled_circuit = transpile_circuit(ata_circuit,
                                    backend=backend)
   

    print_circuit_data(backend.name, ata_circuit, transpiled_circuit)


    best_seed_energy = np.inf
    best_seed_gammas, best_seed_betas = None, None

    for seed in range(seed_lim):
        '''
        Within each iteration (seed), each restart (warm or random) finds:
        1. a set of {γ, β} parameters 
        2. the resultant counts histogram
        3. the energy of that distribution (see evaluate function)
        4. Restart runtime
        
        The best restart is the one that returns the lowest energy. This is the restart from which we take the above results. 
        Each seed returns its best restart results which we average over.
        '''
        
        seed_start_time = perf_counter()             #Timer used just to monitor progress. Times how long all the restarts together took.
        
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
                                                                                    warm_restart,
                                                                                    restarts)
        if seed_energy < best_seed_energy:
            best_seed_energy = seed_energy
            best_seed_gammas, best_seed_betas = best_gammas, best_betas
            
        _, best_rel_energy, best_ari, gs_prob = get_counts_data(best_counts,
                                                                W, true_groundstate, true_groundstate_energy, lambda_bal,
                                                                no_of_shots)
        print(f'best seed ari = {best_ari}')
        metrics_dict['rel_error'].append(best_rel_energy)
        metrics_dict['ari'].append(best_ari)
        metrics_dict['runtime'].append(runtime)
        metrics_dict['gsp'].append(gs_prob)
        
        print(f'Seed {seed + 1} / {seed_lim} complete: ({(perf_counter() - seed_start_time):.2f}s)')
        print('\n')
        #energy_data_plot(best_counts, W, lambda_bal, true_groundstate_energy)
                
        #top_ten_states(best_counts, true_groundstate)
        
    return *metric_stats(metrics_dict), gamma, beta, best_seed_gammas, best_seed_betas, best_counts, ata_circuit


def define_sim_backend(noise_strengths,
                    readout_prob):
    '''
    Simulator backend -> Used for computing the measurement counts when optimising. Can be with or without noise.
    '''
    if noise_strengths == (0.0, 0.0) and readout_prob == 0.0:
        sim_backend = AerSimulator()
    else:
        noise_model = make_noise_model(*noise_strengths, 
                                           readout_prob)
        print(noise_model)
        sim_backend = AerSimulator(noise_model=noise_model)
    
    return sim_backend


def print_circuit_data(qpu_name,
                       ata_circuit,
                       t_circuit):
    
    '''
    List of the ideal vs transpiled circuit properties. Can verify the inflated gate count and circuit depth.
    The qpu chosen (the least busy), is also given.
    '''
    
    print(f'QPU: {qpu_name}')
    print(f'All-to-all circuit gate count = {sum(list(ata_circuit.count_ops().values()))}')
    print(f'All-to-all Circuit Depth = {ata_circuit.depth()}')
    
    print(f'Transpiled circuit gate count = {sum(list(t_circuit.count_ops().values()))}')
    print(f'Transpiled Depth = {t_circuit.depth()}')
   