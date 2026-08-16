from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
def build_qaoa_circuit(W, lambda_bal, gamma, beta, p):
    '''
    Builds the p layer QAOA circuit using the generalised parameters. 
    Only one circuit is ever built, only the RZZ and RX gate input angle parameters change. 
    '''
    
    N = len(W)
    qreg_q = QuantumRegister(N, 'q')
    creg_c = ClassicalRegister(N, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)
    
    circuit.h(qreg_q)                       #Superposition layer
    
    J = 2*lambda_bal - W                    #Effective coupling matrix. Equivalent to classical Ising energy.
    for layer in range(p):                  #p Repeated Cost+Mixer layers.
        for i in range(N):
            for j in range(i+1, N):                                              #Start at i+1 since we don't want to double count the similarity measures.
                circuit.rzz(2*gamma[layer]*J[i][j], qreg_q[i], qreg_q[j])                #Cost layer. Applies RZZ gates to all connected vertices. Factor of 2 cancels the qiskit convention of a gamma/2.
        
        circuit.rx(2 * beta[layer], qreg_q)            #Mixer layer. Applies RX gates to every qubit. Allows for interference between qubit phases.
           
    circuit.measure(qreg_q, creg_c)
        
    return circuit

def bind_params(circuit, 
                gamma, 
                beta,
                gamma_values : float, 
                beta_values : float, 
                p):
    '''
    Binds parameter values to the gates as in build_qaoa_circuit.
    '''
    return circuit.assign_parameters({gamma[i]: gamma_values[i] for i in range(p)} |
                                    {beta[i]: beta_values[i] for i in range(p)})
