from qiskit_aer.noise import NoiseModel, depolarizing_error

def add_depolarizing_noise(noise_strength : float):
    '''
    Add a simple source of noise to quantum circuit using single depolarizing channel.
    This means there is a small probability that each gate output is randomised.
    
    noise_strength is the tunable noise parameter controlling how much noise the circuit experiences
    '''
    
    noisemodel = NoiseModel()
    
    single_qubit_error = depolarizing_error(noise_strength, 1)
    two_qubit_error = depolarizing_error(noise_strength, 2)
    
    noisemodel.add_all_qubit_quantum_error(single_qubit_error, ["h", "rx", "rz"])   #Error to add to all single qubit gates (RX, H, RZ)

    noisemodel.add_all_qubit_quantum_error(two_qubit_error, ["rzz"])    #Error to add to all two qubit gates (RZZ)
    
    return noisemodel
    
    