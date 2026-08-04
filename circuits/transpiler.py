from qiskit import transpile

def transpile_circuit(circuit, backend):
    return transpile(
                    circuit,
                    backend=backend,
                    seed_transpiler=42,
                    optimization_level=1
                    )