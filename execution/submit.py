from qiskit_ibm_runtime import Batch, SamplerV2 as Sampler
from global_params import threeway_no_of_shots

def submit(transpiled_circuit, 
           backend,
           repeats):

    qpu_name = backend.name
    print(qpu_name)
    
    print(transpiled_circuit.count_ops())
    jobs = []

    with Batch(backend=backend) as batch:
        sampler = Sampler(mode=batch)

        for i in range(repeats):
            job = sampler.run([transpiled_circuit],
                            shots=threeway_no_of_shots)

            jobs.append(job)

            print(f'Repeat {i}: Job ID = {job.job_id()}')
                  
            with open('history/jobs.log', 'a') as f:
                f.write(f'repeat={i},{job.job_id()},{qpu_name}\n')

    with open('history/last_batch_jobs.txt', 'w') as f:
        for i, job in enumerate(jobs):
            f.write(f'repeat={i},{job.job_id()},{qpu_name}\n')
            
            
    return [job.job_id() for job in jobs]