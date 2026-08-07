from global_params import (sweet_spot,
                            mode, 
                            experiment_option,
                            readout_error_probability,
                            dep_noise_strengths)
from qiskit_ibm_runtime import QiskitRuntimeService
from execution.run_experiment import run_experiment
from execution.run_threeway_comparison import run_threeway_comparison

def main():
    service = QiskitRuntimeService(instance="Warwick-flex")
    backend = service.least_busy(simulator=False,       #Chosen backend is least busy
                                operational=True)       #Same for both modes        
    
    if mode == 1:
        run_experiment(experiment_option, backend, sweet_spot, None)
        
    elif mode == 2:
        if dep_noise_strengths == (0, 0) or readout_error_probability == 0:
            raise ValueError('Depolarisation strengths and readout error must both be non-zero') 
        else:
            run_threeway_comparison(backend, service)
                  
    else:
        raise ValueError('Unknown Mode')

if __name__ == '__main__':
    main()