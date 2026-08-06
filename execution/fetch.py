import sys
from analysis.counts_analysis import get_counts_data
from global_params import threeway_no_of_shots

def fetch(params, lambda_bal, service):
    real_metric_results = {'config' : None,
                        'rel_error': None,
                        'ari': None,
                        'runtime': None,
                        'gsp': None}
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
    runtime = job.usage()                   #runtime is how long the qpu took to execute the job. #Is not time from when job created to finished.
                                                                    
    real_metric_results['config'] = config
    real_metric_results['rel_error'] = rel_error
    real_metric_results['ari'] = ari
    real_metric_results['runtime'] = runtime
    real_metric_results['gsp'] = gsp
    
    return real_metric_results, counts, job_id

