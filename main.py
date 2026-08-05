from global_params import (similarity_type,
                           sweet_spot,
                           lambda_bal,
                            mode, 
                            experiment_option, 
                            qpu_name_OPT,
                            readout_error_probability,
                            dep_noise_strengths)

from run_experiment import run_experiment
from experiments.run_threeway import ss_run
from problem.generatesystem import generate_toyproblem_params
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from circuits.qaoa_circuit import bind_params
from submit import submit
from fetch import fetch


def main():
    service = QiskitRuntimeService(instance="Warwick-flex")
    if mode == 'OPTIMISE':
        metric_results = run_experiment(experiment_option, qpu_name_OPT, service, sweet_spot)
        
    elif mode == '3-COMP':
        if dep_noise_strengths == (0, 0):
            raise ValueError('Depolarisation strengths must be non-zero') 
        '''
        This is the threeway comparison mode. 
        
        Given a sweet spot (N,p) that is suitable for a QPU i.e. pulled back from the maximum sweet spot, optimise 
        on the noisy_simulator for our best gammas and betas. 
        Then do a single-run comparison of clean, noisy and real-hardware.
        
        We optimise to find the qaoa gamma/beta parameters using the noisy simulator.
        We use these same params in all three approaches (and with fixed global params, W, λ, etc in global_params)
        '''
        #params = (similarity_matrix, true_groundstate, true_groundstate_energy)
        #Can include before the depth_scan call since params doesn't change with p.
        params = generate_toyproblem_params(sweet_spot[0], lambda_bal, similarity_type)
        
        backend = service.least_busy(
                        simulator=False,
                        operational=True
                    )
        backend_name = backend.name                         #See doc_string above.
        optimised_qaoa_params = {'gammas' : None,
                                 'betas' : None}
         
        threeway_comp = {'clean' : {'error' : {'depolar' : (0,0), 'readout' : 0, 'counts' : None, 'metrics' : None}},
                         'noisy' : {'error' : {'depolar' : dep_noise_strengths, 'readout' : readout_error_probability}, 'counts' : None, 'metrics' : None},
                         'real' : {'error' : None, 'counts' : None, 'metrics' : None}}
    
        
        gamma, beta, ss_gammas, ss_betas, ata_circuit = run_experiment('depth', backend_name, service, sweet_spot, params)
        optimised_qaoa_params['gammas'] = ss_gammas
        optimised_qaoa_params['betas'] = ss_betas  
        print(optimised_qaoa_params)    
        
        for name, data in threeway_comp.items():
            circuit = bind_params(ata_circuit, gamma, beta, ss_gammas, ss_betas, sweet_spot[1])
            t_circuit = transpile(circuit, backend)
            
            if name == 'real':
                '''
                Submit real job
                '''
                job_submitted = submit(circuit, backend)
                if job_submitted == False:
                    real_metric_results, real_counts = fetch(params, lambda_bal, service)
                else:
                    continue
                
                threeway_comp[name]['metrics'] = real_metric_results
                threeway_comp[name]['counts'] = real_counts
                
            else:
                '''
                Run the sweet spot through the qaoa (no optimisation since it has already been done).
                Split the universal scan experiment into two explicit sub-experiments: 
                - classical appraoches
                - quantum approaches
                
                Then use the quantum approach (just the cobyla method) to run the circuit through once.
                '''
                sim_metric_results, sim_counts = ss_run(data['error'], t_circuit, lambda_bal, params)
                
                threeway_comp[name]['metrics'] = sim_metric_results
                threeway_comp[name]['counts'] = sim_counts
                
        
    else:
        raise ValueError('Unknown Mode')

if __name__ == '__main__':
    main()