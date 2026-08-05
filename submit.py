from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler
import matplotlib.pyplot as plt
from global_params import threeway_no_of_shots

def submit(ata_circuit, 
           backend):
    
    skip_job = True

    qpu_name = backend.name
    print(qpu_name)
    
    # Compile for hardware
    transpiled_circuit = transpile(
        ata_circuit,
        backend=backend
    )

    print(transpiled_circuit.count_ops())

    transpiled_circuit.draw('mpl')
    plt.show()
    
    if skip_job:
        print('Skip job submission')
        return skip_job
    
    sampler = Sampler(mode=backend)

    job = sampler.run(
        [transpiled_circuit],
        shots=threeway_no_of_shots
    )

    print("Job ID:", job.job_id())

    with open("jobs.log", "a") as f:
        f.write(f"{job.job_id()},{qpu_name}\n")

    with open("last_job.txt", "w") as f:
        f.write(job.job_id())
        
    return 
