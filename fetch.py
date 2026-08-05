import sys
from analysis.counts_analysis import get_counts_data
from time import perf_counter
from global_params import threeway_no_of_shots

def fetch(params, lambda_bal, service):
    metric_results = {'config' : None,
                        'rel_error': [],
                        'ari': [],
                        'runtime': [],
                        'gsp': []}
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    else:
        with open("last_job.txt") as f:
            job_id = f.read().strip()

    job = service.job(job_id)

    print("Status:", job.status())

    result = job.result()

    counts = result[0].data.c.get_counts()
    config, rel_error, ari, gsp = get_counts_data(counts, 
                                                   *params,
                                                   lambda_bal,
                                                   threeway_no_of_shots
                                                           )
    job_metrics = job.metrics()
    runtime = job_metrics["usage"]["quantum_seconds"]                   #runtime is how long the qpu took to execute the job.
                                                                        #Is not time from when job created to finished.
    
    metric_results['config'].append(config)
    metric_results['rel_error'].append(rel_error)
    metric_results['ari'].append(ari)
    metric_results['runtime'].append(runtime)
    metric_results['gsp'].append(gsp)
    
    return counts

