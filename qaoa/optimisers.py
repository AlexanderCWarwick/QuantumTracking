import numpy as np
from time import perf_counter
from scipy.optimize import minimize
from itertools import product

from qaoa.q_ising_energy import evaluate
    
        
def expand_warm_start(warm_start, 
                        gamma_range, 
                        beta_range, 
                        rng):
    '''
    minimise function expects parameters in order [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
    not [gamma_1, beta_1, ... gamma_p, beta_p]. 
    '''
    
    if warm_start is None:
        return None

    old_p = len(warm_start) // 2

    old_gammas = warm_start[:old_p]
    old_betas = warm_start[old_p:]

    new_gamma = rng.uniform(*gamma_range) * 0.1             #New params are close to 0. (p)th layer can then use the (p-1)th layer solution
    new_beta = rng.uniform(*beta_range) * 0.1               #and make use of the warm restart.

    return np.concatenate([old_gammas, [new_gamma], old_betas, [new_beta]])  
          
            
def cobyla(W, 
           circuit,  
            backend,   
            gamma,  
            beta,  
            lambda_bal,  
            no_of_shots, 
            seed, 
            p, 
            gamma_range, 
            beta_range,
            warm_restart,
            restarts):
    '''
    COBYLA optimised QAOA. 
    COBYLA minimisation does not use the gradient (since we do not have that information)
    
    Alternatively there is COBYQA which uses optimises quadratically (rather than linearly).
    COBYQA uses quadratic approximation, COBYLA is only linear. We can expect COBYQA to outperform COBYLA in 
    each metrics, except in runtime.
    
    Instead, working through the evaluate function output space (the avg energy of the returned sampled distribution)
    it uses a shrinking trust region to estimate better values for the tuning parameters to get a better estimate.
    
    Warm restarts now use the best parameters from the previous p scan as a new starting point. 
    Example: If we have 3 restarts and are (depth) scanning over p=[1,2] , the pattern is: 
    p = 1
    restart:
    1 - random
    2 - random
    3 - random
    
    p = 2
    restart:
    1 - warmstart using best p=1 gamma_1 and beta_1 values.
    2 - random
    3 - random
    ''' 
    best_avg_energy = np.inf
    best_result = None
    
    histories = []                       #List to hold the evolution of each random restarts energy. Plotted in optimiser_energy_trace
    final_energies = []                  #List to hold the returned best energies from each restart. Plotted in optimiser_result_energies
    
    param_bounds = [gamma_range] * p + [beta_range] * p
    rng = np.random.default_rng(seed)
    
    seed_runtime_start = perf_counter()
    
    for restart_idx in range(restarts):
        if restart_idx == 0 and warm_restart is not None:
            #First restart = warm start, the rest callare normal random restarts.
            x0 = expand_warm_start(warm_restart, p, gamma_range, beta_range, rng)
        else:
            # Remaining restarts = random
            x0 = np.concatenate([rng.uniform(*gamma_range, p), rng.uniform(*beta_range, p)])
            
        restart_energies = []
        
        def eval(params):
            gamma_values = params[:p]
            beta_values = params[p:]
            energy = evaluate(W, circuit,  backend,  gamma,  beta,  gamma_values, beta_values, lambda_bal,  no_of_shots, p)
            
            restart_energies.append(energy)
            return energy
    
        result = minimize(eval, x0, method='COBYLA', bounds=param_bounds, options={'maxiter' : 100})
        result_energy = result.fun
    
        if result_energy < best_avg_energy:
            best_avg_energy = result_energy
            best_result = result
            final_energies.append(result_energy)  
        
        else:
            final_energies.append(best_avg_energy) 
            
        if restart_idx == 0 and warm_restart is not None:
            print(f'(Warm) Restart {restart_idx+1} / {restarts} complete')
        else:
            print(f'Restart {restart_idx+1} / {restarts} complete')
        histories.append(restart_energies)
    
    seed_time = perf_counter() - seed_runtime_start
    #Plot energy trace of each restart.
    #optimiser_energy_trace(restarts, histories, best_history_idx, method, p, len(W))

    #Plot how the cobyla_avg_energy changes through the cobyla_restarts number of repitions.
    #optimiser_result_energies(final_energies, method, p, len(W))

    return best_result.x[:p], best_result.x[p:], seed_time, best_avg_energy