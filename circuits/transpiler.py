from qiskit import transpile

def transpile_circuit(circuit, backend):
    '''
    Transpiler function with fixed seed and optimisation level (1 is moderate complexity level)
    '''
    return transpile(circuit,
                    backend=backend,
                    seed_transpiler=42,
                    optimization_level=1)