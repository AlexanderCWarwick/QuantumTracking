from global_params import (mode, 
                            experiment_option, qpu_name_OPT,
                            readout_error_probability,
                            single_gate_error,
                            double_gate_error)

from run_experiment import run_experiment
from experiments.run_threeway import ss_run
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from circuits.qaoa_circuit import bind_params
from submit import submit


def main():
    service = QiskitRuntimeService(instance="Warwick-flex")
    if mode == 'OPTIMISE':
        metric_results = run_experiment(experiment_option, qpu_name_OPT, service)
        
    elif mode == '3-COMP':
        backend = service.least_busy(
                        simulator=False,
                        operational=True
                    )
        backend_name = backend.name
        optimised_qaoa_params = {'gammas' : None,
                                 'betas' : None}
         
        threeway_comp = {'clean' : {'error' : {'depolar' : (0,0), 'readout' : 0, 'counts' : None, 'metrics' : None}},
                         'noisy' : {'error' : {'depolar' : (single_gate_error, double_gate_error), 'readout' : readout_error_probability}, 'counts' : None, 'metrics' : None},
                         'real' : {'error' : None, 'counts' : None, 'metrics' : None}}
        
        '''
        We optimise to find the qaoa gamma/beta parameters using the noisy simulator.
        We use these same params in all three approaches (and with fixed global params, W, λ, etc in global_params)
        '''
        
        gamma, beta, ss_gammas, ss_betas, ata_circuit = run_experiment('depth', backend_name, service)
        optimised_qaoa_params['gammas'] = ss_gammas
        optimised_qaoa_params['betas'] = ss_betas      
        
        for name, data in threeway_comp:
            circuit = bind_params(ata_circuit, gamma, beta, ss_gammas, ss_betas)
            t_circuit = transpile(circuit, backend)
            
            if name == 'real':
                '''
                Submit real job
                '''
            else:
                '''
                Run the sweet spot through the qaoa (no optimisation since it has already been done).
                Split the universal scan experiment into two explicit sub-experiments: 
                - classical appraoches
                - quantum approaches
                
                Then use the quantum approach (just the cobyla method) to run the circuit through once.
                '''
                metric_results, counts = ss_run(threeway_comp[name]['error'], backend_name, t_circuit)
                
                threeway_comp[name]['metrics'] = metric_results
                threeway_comp[name]['counts'] = counts
                
        
    else:
        raise ValueError('Unknown Mode')

if __name__ == '__main__':
    main()