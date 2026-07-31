from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import matplotlib.pyplot as plt

service = QiskitRuntimeService(
    instance="Warwick-flex"
)

backend = service.least_busy(
    simulator=False,
    operational=True
)

print(backend.name)

# Create circuit with classical register
qc = QuantumCircuit(2, 2)

qc.h(0)
qc.cx(0, 1)
# Measure qubits into classical bits
qc.measure([0, 1], [0, 1])

# Compile for hardware
qc_transpiled = transpile(
    qc,
    backend=backend
)

print(qc_transpiled.count_ops())

qc_transpiled.draw('mpl')
plt.show()

sampler = Sampler(mode=backend)

job = sampler.run(
    [qc_transpiled],
    shots=1024
)

print("Job ID:", job.job_id())

with open("jobs.log", "a") as f:
    f.write(f"{job.job_id()},{backend.name}\n")

with open("last_job.txt", "w") as f:
    f.write(job.job_id())
