from global_params import mode, experiment_option, qpu_name_OPT
from optimiser import optimise

from qiskit_ibm_runtime import QiskitRuntimeService

from submit import submit


def main():
    service = QiskitRuntimeService(instance="Warwick-flex")
    if mode == 'OPTIMISE':
        optimise(experiment_option, qpu_name_OPT, service)
        print(99999999999999999999999999999999)
        
    elif mode == 'OPTIMISE-HARDWARE':
        backend = service.least_busy(
                        simulator=False,
                        operational=True
                    )
        
        ss_gammas, ss_betas, ata_circuit = optimise('depth', backend.name, service)
        print(88888888888888888888888888888888888888)
        submit(ss_gammas, ss_betas, ata_circuit, backend)
    else:
        raise ValueError('Unknown Mode')
    
    print(1111111111111111111111111111111111111111)

        

if __name__ == '__main__':
    main()