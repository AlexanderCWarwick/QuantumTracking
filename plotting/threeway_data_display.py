def print_real_quantum_table(similarity_type : str,
                             lambda_bal : float,
                             N : int,
                             p : int,
                             qpu_name : str,
                             threeway_comp : dict,
                             repeats : int):
    
    header = (f'{'App':<10}'
                f'{'N':<7}'
                f'{'p':<7}'
                f'{'CONFIG':<25}'
                f'{'Relative Energy Error':<30}'
                f'{'ARI':<20}'
                f'{'QPU RunTime (s)':<25}'
                f'{'GS Prob':<25}')
    
    print(f'Parameters : sim_matrix={similarity_type}, λ={lambda_bal}')
    print(f'Backend = {qpu_name}, Batch Repeats = {repeats}')
    print(header)
    print('-' * len(header))
    
    for name, data in threeway_comp.items():
        metric_results = data['metrics']
        
        print(f'{name:<10}'
            f'{2*N:<7}'
            f'{p:<7}'
            f'{str(metric_results['config']):<25}'
            f'{metric_results['rel_error']:<30}'
            f'{metric_results['ari'][0]:<20}'
            f'{metric_results['runtime']:<25}'
            f'{metric_results['gsp']:<25}')