def format_metric(metric):
    return f'{metric['mean']:.4f} \u00b1 {metric['error']:.4f}'
    
def print_benchmark_table(hits : int, 
                          similarity_type : str, 
                          lambda_bal : float, 
                          classical_results : dict[dict]):
    header = (f'{'Hits':<7}'
              f'{'Layers':<7}'
        f'{'Classical Algorithm':<30}'
        f'{'Relative Energy Error':<30}'
        f'{'ARI':<20}'
        f'{'Time (s)':<20}'
        f'{'Convergence Fraction':<30}')
    
    print(f'Parameters : sim_matrix={similarity_type}, λ={lambda_bal}')
    print('\n')
    print(header)
    print('-' * len(header))
    
    for optimiser_name, metrics in classical_results.items():
        if optimiser_name == 'Simulated Annealing':
            print(f'{2*hits:<7}'
                f'{optimiser_name:<30}'
                f'{format_metric(metrics['rel_error']):<30}'
                f'{format_metric(metrics['ari']):<20}'
                f'{format_metric(metrics['runtime']):<20}'
                f'{format_metric(metrics['conv_frac']):<30}')
        else:
            print(f'{2*hits:<7}'
                f'{optimiser_name:<30}'
                f'{format_metric(metrics['rel_error']):<30}'
                f'{format_metric(metrics['ari']):<20}'
                f'{format_metric(metrics['runtime']):<20}'
                '-')
            
        
        
def print_quantum_table(similarity_type : str, 
                        lambda_bal : float, 
                        hits : int,
                        p : int,
                        quantum_results : dict,
                        noise_strengths : tuple[float], 
                        readout_prob : float, 
                        scan_type : str):
    header = (f'{'N':<7}'
              f'{'p':<7}'
                f'{'Optimiser':<12}'
                f'{'Relative Energy Error':<30}'
                f'{'ARI':<20}'
                f'{'Seed RunTime (s)':<20}'
                f'{'GS Prob':<20}')
    
    if scan_type == 'depth':
        print(f'Parameters : sim_matrix={similarity_type}, λ={lambda_bal}')
        print(f'Depolarising Noise (1q, 2q) = {noise_strengths}, Readout Error = {readout_prob}')
        print('\n')
        print(header)
        print('-' * len(header))
        for optimiser_name, metrics in quantum_results[p].items():
            print(f'{2*hits:<7}'
                  f'{p:<7}'
                f'{optimiser_name:<12}'
                f'{format_metric(metrics['rel_error']):<30}'
                f'{format_metric(metrics['ari']):<20}'
                f'{format_metric(metrics['runtime']):<20}'
                f'{format_metric(metrics['gsp']):<20}')
        print('\n')
        
    elif scan_type == 'scale':
        print(f'Parameters : sim_matrix={similarity_type}, λ={lambda_bal}')
        print(f'Depolarising Noise (1q, 2q) = {noise_strengths}, Readout Error = {readout_prob}')
        print('\n')
        print(header)
        print('-' * len(header))
        for optimiser_name, metrics in quantum_results[hits].items():
            print(f'{2*hits:<7}'
                  f'{p:<7}'
                    f'{optimiser_name:<12}'
                    f'{format_metric(metrics['rel_error']):<30}'
                    f'{format_metric(metrics['ari']):<20}'
                    f'{format_metric(metrics['runtime']):<20}'
                    f'{format_metric(metrics['gsp']):<20}')
        print('\n')
        
    elif scan_type == 'uni':
        print(f'Parameters : sim_matrix={similarity_type}, λ={lambda_bal}')
        print(f'Depolarising Noise (1q, 2q) = {noise_strengths}, Readout Error = {readout_prob}')
        print('\n')
        print(header)
        print('-' * len(header))
        
        for optimiser_name, metrics in quantum_results[hits][p].items():
            print(f'{2*hits:<7}'
                  f'{p:<7}'
                    f'{optimiser_name:<12}'
                    f'{format_metric(metrics['rel_error']):<30}'
                    f'{format_metric(metrics['ari']):<20}'
                    f'{format_metric(metrics['runtime']):<20}'
                    f'{format_metric(metrics['gsp']):<20}')
        print('\n')            
            
