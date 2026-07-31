
option = 'depth-scale'                  #This is the identifier for which 'task' we want to do.
no_of_shots =  30               #Number of measurements the quantum simulator will make of the circuit (all independent).
seed_lim = 3                    #Number of runs of the QAOA to calculate means and errors.
lambda_bal = 0.25                  #Lambda_balance parameter values to be used in the Hamiltonian. Modelled as a constant.
similarity_type = 'KNN'
    
    
single_qubit_noise = 0                                            #Single qubit gate depolarisation error.
double_qubit_noise = 5 * single_qubit_noise                           #Two qubit gate depolarisation error. Estimate ratio to be 5 : 1.
noise_strengths = (single_qubit_noise, double_qubit_noise)
readout_prob = 1e-3                                                   #Readout error probability. Same for all qubit measurements.