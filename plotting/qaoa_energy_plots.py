import numpy as np
import matplotlib.pyplot as plt

def optimiser_energy_trace(restarts : int,  histories : np.ndarray[float],  best_history_idx : int, optimiser : str,
                           p: int, hits : int):
    fig, ax = plt.subplots(restarts, 1, figsize=(14,14))
    for j, history in enumerate(histories):
        ax[j].plot(history)
        ax[j].set_ylabel('Energy')
        
        if j == best_history_idx:
            best_energy = history[-1]

            ax[j].annotate('Best result',
                            xy=(len(history) - 1, best_energy),
                            xytext=(-60, 20),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='->'))
            
    fig.suptitle(f'{optimiser} Energy traces for ($p={p}$, $N={hits}$)')
    fig.supxlabel('COBYLA iteration')
    
    
    
def optimiser_result_energies(final_energies, optimiser, p, hits):
    plt.figure()
    x = np.arange(1,len(final_energies) + 1)
    plt.title(f'{optimiser} best energy evolution for ($p={p}$, $N={hits}$).')
    plt.xlabel('Restart iteration')
    plt.xticks(x)
    plt.ylabel('Energy')
    plt.plot(x, final_energies)
    
    
def plot_energy_hist(energies, true_groundstate_energy):
    plt.figure()
    plt.hist(energies, bins=30)
    plt.axvline(true_groundstate_energy, color='red', linestyle='--', label=f'Exact GS energy = {true_groundstate_energy:.2f}')
    plt.xlabel('Ising energy')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
        
            
def top_ten_states(counts : dict, true_gs : np.ndarray[int]):
    top_10 = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10])      #Gets top 10 states in {state : count} form.
    states = list(top_10.keys())                                                            #List conversion of states
    frequencies = list(top_10.values())                                                     #List conversion of counts
    
    inv_true_gs = ''.join((true_gs^1).astype(str))                      #Inverse true_gs as str
    true_gs = ''.join(true_gs.astype(str))                              #true_gs as str
    
    bar_colors = ["red" if state == true_gs or state == inv_true_gs else "blue" for state in states]
        
    plt.bar(states, frequencies, color = bar_colors)
    plt.xlabel("Measured configuration")
    plt.ylabel("Counts")
    plt.title("10 Most Frequently Measured Configurations")
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
                
        
    
    