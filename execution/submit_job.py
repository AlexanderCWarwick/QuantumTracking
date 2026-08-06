from global_params import (similarity_type,
                           sweet_spot,
                           lambda_bal,
                           readout_error_probability,
                           dep_noise_strengths)
import matplotlib.pyplot as plt
from experiments.run_threeway import ss_run
from plotting.table_print import print_real_quantum_table
from problem.generatesystem import generate_toyproblem_params
from circuits.qaoa_circuit import bind_params
from circuits.transpiler import transpile_circuit
from execution import run_experiment
from execution import submit
'''
This is the threeway comparison mode. 
        
Given a sweet spot (N,p) that is suitable for a QPU i.e. pulled back from the maximum sweet spot, optimise 
on the noisy_simulator for our best gammas and betas. 
Then do a single-run comparison of clean, noisy and real-hardware.

We optimise to find the qaoa gamma/beta parameters using the noisy simulator.
We use these same params in all three approaches (and with fixed global params, W, λ, etc in global_params)
'''
def submit_job(backend):
    n = sweet_spot[0]
    p = sweet_spot[1]

    #params = (similarity_matrix, true_groundstate, true_groundstate_energy)
    #Can include before the depth_scan call since params doesn't change with p.
    params = generate_toyproblem_params(n, lambda_bal, similarity_type)
        
    threeway_comp = {'clean' : {'error' : {'depolar' : (0,0), 'readout' : 0,}, 'counts' : None, 'metrics' : None},
                        'noisy' : {'error' : {'depolar' : dep_noise_strengths, 'readout' : readout_error_probability}, 'counts' : None, 'metrics' : None},
                        'real' : {'counts' : None, 'metrics' : None}}


    gamma, beta, ss_gammas, ss_betas, ata_circuit = run_experiment('depth', backend, sweet_spot, params)

    circuit = bind_params(ata_circuit, gamma, beta, ss_gammas, ss_betas, p)
    t_circuit = transpile_circuit(circuit, backend)

    circuit.draw('mpl')
    plt.savefig(f'assest/plots/All-to-All_circuit_N{sweet_spot[0]}_p{sweet_spot[1]}')
    t_circuit.draw('mpl')
    plt.savefig(f'assest/plots/Transpiled circuit_N{sweet_spot[0]}_p{sweet_spot[1]}')


    for name, data in threeway_comp.items():
        if name == 'real':
            '''
            Submit real job
            '''
            submit.submit(t_circuit, backend)

        else:
            '''
            Run the sweet spot through the qaoa (no optimisation since it has already been done).
            '''
            sim_metric_results, sim_counts = ss_run(name, data['error'], t_circuit, lambda_bal, params)
            
            threeway_comp[name]['metrics'] = sim_metric_results
            threeway_comp[name]['counts'] = sim_counts
            
    print_real_quantum_table(similarity_type, lambda_bal, n, p, backend.name, threeway_comp)