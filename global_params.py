mode = 2

experiment_option = 'class'
lambda_bal = 0.3
similarity_type = 'KNN'

no_of_shots = 2048
seed_lim = 5
restarts = 5           


single_gate_noise = 1e-3                       
double_gate_noise = 5 * single_gate_noise
dep_noise_strengths = (single_gate_noise, double_gate_noise)
readout_error_probability = 1e-2


sweet_spot = (3, 1)
threeway_no_of_shots = 2048
job_repeats = 3

'''
Parameter descriptions:
mode -> Should take two values:
1. mode = 1 provides all experiments (depth, scale and universal scans), the points (N,p) to test are changed in run_experiment.py
2. mode = 2 is the 3-way COMParison (see Week 8 in readme).
 

experiment_option
This is the identifier for which experiment we want to do.
1. 'scale'
2. 'depth'
3. 'class'
If mode == 'OPTIMISE-HARDWARE' then this is overrided to 'depth'. 

lambda_bal
Hamiltonian balance term prefactor. No pre-optimisation.

similarity_type 
Type of similarity matrix we use: Choice is KNN and RBF.

no_of_shots 
Number of measurements the quantum simulator will make of the circuit in each restart.

seed_lim
Number of runs of the QAOA to calculate means and errors.

restarts
Number of restarts to use per seed.


noise params:
single_gate_noise -> Single qubit gate depolarisation error.
double_gate_noise -> Two qubit gate depolarisation error. Estimate ratio to be 5 : 1.
readout_error_probability -> Readout error probability. Same for all qubit measurements.

sweet_spot
(N, p) sweet spot to test in the three way comparison.

threeway_no_of_shots
Number of shots to use on the real hardware jobs. 

job_repeats
Number of jobs to put into each submitted batch.
'''
