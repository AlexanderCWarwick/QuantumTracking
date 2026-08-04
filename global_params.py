mode = 'OPTIMISE'
experiment_option = 'depth'                  #This is the identifier for which experiment we want to do.
                                             #If mode == 'OPTIMISE-HARDWARE' then this is overrided to 'depth'.

lambda_bal = 0.3                  #Lambda_balance parameter values to be used in the Hamiltonian. Modelled as a constant.
similarity_type = 'KNN'

no_of_shots =  2048                #Number of measurements the quantum simulator will make of the circuit in each restart.
seed_lim = 5                    #Number of runs of the QAOA to calculate means and errors.
restarts = 5                    #Number of restarts to use per seed.
single_qubit_noise = 0                                            #Single qubit gate depolarisation error.
double_qubit_noise = 5 * single_qubit_noise                           #Two qubit gate depolarisation error. Estimate ratio to be 5 : 1.
noise_strengths = (single_qubit_noise, double_qubit_noise)
readout_prob = 0                                                   #Readout error probability. Same for all qubit measurements.


sweet_spot = (6, 1)
qpu_name_OPT = 'ibm_miami'              #Name of the qpu that user wants to optimise to. If the user wants to test real hardware than this is
                                        #overrided to whichever qpu is the least busy. 
hardware_no_of_shots = 2048