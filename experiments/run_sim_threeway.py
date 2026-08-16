from qaoa.q_ising_energy import run_qaoa
from noise.add_noisemodel import make_noise_model
from analysis.counts_analysis import get_counts_data
from global_params import threeway_no_of_shots
from qiskit_aer import AerSimulator
from time import perf_counter

def sim_run(name : str,
           data : dict,
           transpiled_circuit,
           lambda_bal : float,
           params,
           repeats) -> tuple[dict, dict]:
    '''
    
    '''
    metric_results = {i : {'config' : None,
                        'rel_error': None,
                        'ari': None,
                        'runtime': None,
                        'gsp': None,
                        'counts': None} for i in range(repeats)}
    
    if name == 'clean':
       sim_backend = AerSimulator()
    elif name == 'noisy':
       noise_model = make_noise_model(*data['depolar'],
                                 data['readout'])
       sim_backend = AerSimulator(noise_model = noise_model)
    else:
       raise ValueError(f"Unknown backend type: {name}")
    
    for i in range(repeats):
      start_time = perf_counter()
      counts = run_qaoa(sim_backend, 
                        transpiled_circuit,
                        threeway_no_of_shots)
      
      config, rel_error, ari, gsp = get_counts_data(counts,
                                                   *params,
                                                   lambda_bal,
                                                   threeway_no_of_shots)
      
      metric_results[i] = {'config' : config,
                        'rel_error': rel_error,
                        'ari': ari,
                        'runtime': perf_counter() - start_time,
                        'gsp': gsp,
                        'counts': counts}
      
    return metric_results
    
    


