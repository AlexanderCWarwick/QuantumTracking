import sys
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    instance="Warwick-flex"
)

if len(sys.argv) > 1:
    job_id = sys.argv[1]
else:
    with open("last_job.txt") as f:
        job_id = f.read().strip()

job = service.job(job_id)

print("Status:", job.status())

result = job.result()

counts = result[0].data.c.get_counts()

print(counts)
