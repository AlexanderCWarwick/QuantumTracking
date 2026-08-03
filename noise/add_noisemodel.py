from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError, thermal_relaxation_error

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
    
    noisemodel.add_all_qubit_quantum_error(single_qubit_error, ['h', 'rx', 'sx', 'rz'])   
    noisemodel.add_all_qubit_quantum_error(two_qubit_error, ['rzz', 'cx'])
    return noisemodel


def add_readout_error(noisemodel, 
                      probability: float) -> NoiseModel:
    
    '''
    Adds a probability to all measurement gates.
    There is then a (probability) chance that qubit n will be read out wrong.
    '''

    readout_error = ReadoutError([[1 - probability, probability],
                                  [probability, 1 - probability]])
    
    noisemodel.add_all_qubit_readout_error(readout_error)
    return noisemodel


def add_T1_T2_error(noisemodel, T1, T2, one_qubit_gate_time, two_qubit_gate_time):
    '''
    T1 error = excited state amplitude is lost. Amplitude information is lost.
    T2 error = Qubit state relative phase information is lost. Destroys coherence -> Removes interference 
    '''
    
    single_error = thermal_relaxation_error(T1, T2, one_qubit_gate_time)
    
    two_qubit_ind_error = thermal_relaxation_error(T1, T2, two_qubit_gate_time)
    two_error = two_qubit_ind_error.tensor(two_qubit_ind_error)
    
    noisemodel.add_all_qubit_quantum_error(single_error, ['h', 'rx', 'sx', 'rz'])
    noisemodel.add_all_qubit_quantum_error(two_error, ['rzz', 'cz'])
    
    return noisemodel


def make_noise_model(single_dep_noise_strength : float,
                    double_dep_noise_strength : float,
                    readout_prob : float) -> NoiseModel:
    
    noisemodel = NoiseModel()
    
    noisemodel = add_depolarizing_noise(single_dep_noise_strength, double_dep_noise_strength)
    noisemodel = add_readout_error(noisemodel, readout_prob)
    
    
    return noisemodel
    
    