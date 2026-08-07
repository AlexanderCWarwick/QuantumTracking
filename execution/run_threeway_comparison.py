from global_params import (similarity_type,
                           sweet_spot,
                           lambda_bal,
                           job_repeats,
                           readout_error_probability,
                           dep_noise_strengths,
                           threeway_no_of_shots)
import numpy as np
import matplotlib.pyplot as plt
from analysis import threeway_data_analysis
from experiments.run_sim_threeway import sim_run
from problem.generatesystem import generate_toyproblem_params
from circuits.qaoa_circuit import bind_params
from circuits.transpiler import transpile_circuit
from execution.run_experiment import run_experiment
from execution.submit import submit
from execution.fetch import fetch
'''
This is the threeway comparison mode. 
        
Given a sweet spot (N,p) that is suitable for a QPU i.e. pulled back from the maximum sweet spot, optimise 
on the noisy_simulator for our best gammas and betas. 
Then do a multi-ran comparison of clean, noisy and real-hardware. This is done using a Batch session for the real hardware.

We optimise to find the qaoa gamma/beta parameters using the noisy simulator.
We use these same params in all three approaches (and with fixed global params, W, λ, etc in global_params)
'''
def run_threeway_comparison(backend, service):
    
    n = sweet_spot[0]
    p = sweet_spot[1]
    #params = (similarity_matrix, true_groundstate, true_groundstate_energy)
    #Can include before the depth_scan call since params doesn't change with p.
    params = generate_toyproblem_params(n, lambda_bal, similarity_type)
        
    threeway_comp = {'clean' : {'error' : {'depolar' : (0,0), 'readout' : 0}, 'results' : None},
                     
                    'noisy' : {'error' : {'depolar' : dep_noise_strengths, 'readout' : readout_error_probability}, 'results' : None},
                    
                    'real' : {'results' : None}}


    gamma, beta, ss_gammas, ss_betas, ata_circuit = run_experiment('depth', backend, sweet_spot, params)

    circuit = bind_params(ata_circuit, gamma, beta, ss_gammas, ss_betas, p)
    t_circuit = transpile_circuit(circuit, backend)
    '''
    fig = circuit.draw('mpl')
    plt.savefig(f'assets/plots/All-to-All_circuit_N{sweet_spot[0]}_p{sweet_spot[1]}')
    plt.close(fig)
    fig = t_circuit.draw('mpl')
    plt.savefig(f'assets/plots/Transpiled circuit_N{sweet_spot[0]}_p{sweet_spot[1]}')
    plt.close(fig)
    '''
    for name, data in threeway_comp.items():
        job_ids = None
        if name == 'real':
            '''
            Submit real job
            '''
            job_ids = submit(t_circuit, backend, repeats=job_repeats)
            
            real_results = fetch(params, lambda_bal, service, job_ids)       
            threeway_comp[name]['results'] = real_results

        else:
            '''
            Run the sweet spot through the qaoa (no optimisation since it has already been done).
            '''
            sim_metric_results = sim_run(name, data['error'], t_circuit, lambda_bal, params, repeats=job_repeats)
            threeway_comp[name]['results'] = sim_metric_results
            
    
    info = (f'(N,p) = {sweet_spot}\n'
            f'Similarity Type: {similarity_type}\n'
            f'Lambda = {lambda_bal}\n'
            f'#Shots = {threeway_no_of_shots}\n'
            f'ReadoutProb = {readout_error_probability}\n'
            f'Depolar error (1q,2q) = {dep_noise_strengths}')
    
   
    with open('history/experiment_log.txt', 'a') as file:
        file.write(f'{info}\n')
        file.write(f'{threeway_comp}\n')
        file.write(f'{backend.name}\n')
        file.write(f'{job_ids}\n')
        file.write('\n')
        
    path = 'history/experiment_history.npz'
    try:
        with np.load(path, allow_pickle=True) as data:
            history = data['history'].item()

    except FileNotFoundError:
        history = {}
        
    if sweet_spot not in history:
        history[sweet_spot] = []

    history[sweet_spot].append({'job_ids': job_ids,
                                'similarity': similarity_type,
                                'lambda_bal': lambda_bal,
                                'shots': threeway_no_of_shots,
                                'readoutprob': readout_error_probability,
                                'depolar': dep_noise_strengths,
                                'backend': backend.name,
                                'threeway_comp': threeway_comp})
                            
    np.savez(path, history=history)
    
    summary_statistics = threeway_data_analysis.threeway_data_analysis(threeway_comp)
    threeway_data_analysis.plot_threeway_metrics(sweet_spot,
                                                similarity_type, 
                                                 lambda_bal, 
                                                 threeway_no_of_shots, 
                                                 readout_error_probability,
                                                dep_noise_strengths,
                                                 summary_statistics)
    
    