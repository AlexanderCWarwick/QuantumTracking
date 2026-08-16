from analysis.counts_analysis import get_counts_data
from global_params import threeway_no_of_shots

def fetch(params, lambda_bal, service, job_ids):
    real_metric_results = {}
    
    for i, job_id in enumerate(job_ids):

        job = service.job(job_id)

        print(f"Repeat {i + 1}")
        print(f"Job ID: {job_id}")
        print(f"Status: {job.status()}")

        result = job.result()

        counts = result[0].data.c.get_counts()

        config, rel_error, ari, gsp = get_counts_data(counts,
                                                    *params,
                                                    lambda_bal,
                                                    threeway_no_of_shots)

        runtime = job.usage()

        real_metric_results[i + 1] = {'job_id': job_id,
                                        'config': config,
                                        'rel_error': rel_error,
                                        'ari': ari,
                                        'runtime': runtime,
                                        'gsp': gsp,
                                        'counts': counts}

    return real_metric_results
    
