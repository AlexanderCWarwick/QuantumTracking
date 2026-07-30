from qiskit_aer.noise import NoiseModel, depolarizing_error

def add_depolarizing_noise(single_noise_strength : float,
                           double_noise_strength : float) -> NoiseModel:
    '''
    Add a simple source of noise to quantum circuit using single depolarizing channel.
    This means there is a small probability that each gate output is randomised.
    
    noise_strength is the tunable noise parameter controlling how much noise the circuit experiences
    '''
    
    noisemodel = NoiseModel()
    
    single_qubit_error = depolarizing_error(single_noise_strength, 1)  #Error to add to all single qubit gates (RX, H)
    two_qubit_error = depolarizing_error(double_noise_strength, 2)     #Error to add to all two qubit gates (RZZ)
    
    noisemodel.add_all_qubit_quantum_error(single_qubit_error, ["h", "rx"])   
    noisemodel.add_all_qubit_quantum_error(two_qubit_error, ["rzz"])
    
    return noisemodel
    
    