from numpy import pi
mode = 1

experiment_option = 'scale'

depth_hits = 6
depth_layers = range(1, 3) 

scale_hits = range(3,7)
scale_layers = 2

uni_hits = 5
uni_layers = 1

track_seed = 45
intersection_allowed = False            
track_noise = 1e-2           

nearneighb_n = 3
rbf_sigma = 0.3          
lambda_bal = 0.3
similarity_type = 'KNN'
graph_switch = False
classical_loops = 10

no_of_shots = 5000
seed_lim = 3
restarts = 4           
gamma_range = (0, 2*pi)
beta_range = (0, 2*pi)

single_gate_noise = 1e-3                       
double_gate_noise = 5 * single_gate_noise
dep_noise_strengths = (single_gate_noise, double_gate_noise)
readout_error_probability = 1e-2

sweet_spot = (4, 1)
threeway_no_of_shots = 10000
job_repeats = 3

'''
Parameter descriptions:
mode : str
1. mode = 1 provides all experiments (depth, scale and universal scans), the points (N,p) to test are changed in run_experiment.py
2. mode = 2 is the 3-way COMParison (see Week 8 in readme).
 

experiment_option : str
This is the identifier for which experiment we want to do.
1. 'scale'
2. 'depth'
3. 'class'
If mode = 2 then this is overrided to 'depth'. 


depth/scale/uni_hits and depth/scale/uni_layers are used when mode = 1.

depth_hits, depth_layers : int, np.ndarray[int]
The points (N,p) the user should input to test in the depth scan.

scale_hits, scale_layers : np.ndarray[int], int
The points (N,p) the user should input to test in the scale scan.

uni_hits, uni_layers : int, int
The point (N,p) the user should input to test in the universal scan.

intersection_allowed : bool
#Control whether linear particles tracks intersect.

track_noise : float
Standard error modelling detector uncertainty.

lambda_bal : float
Hamiltonian balance term prefactor. No pre-optimisation.

similarity_type : str
Type of similarity matrix we use: Choice is KNN and RBF.

graph_switch : bool
Boolean toggle if the user wishes to visualise choice similarit matrix as a graph.

classical_loops : int
Number of times the classical benchmark algorithms are looped for averages/errors. Only used in universal scan.

no_of_shots : int
Number of measurements the quantum simulator will make of the circuit in each restart. Used when mode = 1.

seed_lim : int
Number of runs of the QAOA to calculate means and errors.

restarts : int
Number of restarts to use per seed.

gamma/beta_range : tuple[0, float]
Defines the optimiser search domain for each pair of cost/mixer gamma/beta pairs.
Hence total tuning parameter domain is (gamma_range x beta_range)^p


noise params : all float
single_gate_noise -> Single qubit gate depolarisation error.
double_gate_noise -> Two qubit gate depolarisation error. Estimate ratio to be 5 : 1.
readout_error_probability -> Readout error probability. Same for all qubit measurements.

sweet_spot : tuple[int, int]
(N, p) sweet spot to test in the three way comparison.

threeway_no_of_shots : int
Number of shots to use on real hardware (and in the optimisation prior to job submission). Used when mode = 2.

job_repeats : int
Number of jobs to put into each submitted batch.
'''
