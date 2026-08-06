from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler
import matplotlib.pyplot as plt
from global_params import threeway_no_of_shots

def submit(ata_circuit,
            transpiled_circuit, 
           backend):

    qpu_name = backend.name
    print(qpu_name)
    

    print(transpiled_circuit.count_ops())

    ata_circuit.draw('mpl')
    transpiled_circuit.draw('mpl')
    plt.show()
    
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
        
    return job.job_id()
