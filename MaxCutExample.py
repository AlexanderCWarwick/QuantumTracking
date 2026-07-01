import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

import time

'''
MaxCut optimisation example using Quantum Computing!

Given a graph network G = (V,E), how can we cut the graph into two disjoint subsets of nodes such 
that the cut maximise the total sum weight of all the cut edges. If the graph isn't weighted then we are just maximising 
the number of edges the cut traverses.

In this example we take a simple |V| = 4 vertex network placed at the corners of a square. 
There are three edges forming a Z pattern. The MaxCut is then the cut that goes straight through the middle and hence cuts all
3 edges. 

We then implement QAOA (for now with 1 layer). Each vertex is 'assigned' one qubit and prepared in the groundstate |0>. We then 
feed the qubits into our quantum circuit: Superposition  -->   Cost Layer  -->   Mixer Layer  -->   Measurement. The tunable parameters
are β and γ, which we optimise by a simple grid search (better methods exist, e.g. COBYLA). 
We use the output PDF to pick which partitions are likely the best.

The cost layer hamiltonian in this problem involves only a pairwise wZZ term (adjacency/similarity matrix w) but the tracking problem 
will also involve a balance term.
'''

G = nx.Graph()
G.add_nodes_from(range(4))

edges = [(0,1), (1,2), (2,3)]
weights = [1,1,1]
G.add_weighted_edges_from([(u, v, w) for (u, v), w in zip(edges, weights)])


node_coords = {0 : (0,0), 1 : (1,0), 2 : (0,1), 3 : (1,1)}

nx.draw_networkx_edges(G, node_coords, edgelist=edges, alpha=weights, edge_color='black')        
nx.draw_networkx_nodes(G, node_coords, node_size=200)                                                   
nx.draw_networkx_labels(G, node_coords)


def maxcut_value(bitstring, G):
    value = 0
    for i,j,weight in G.edges(data='weight'):
        if bitstring[i] != bitstring[j]:
            value += weight            
    return value

qreg_q = QuantumRegister(4, 'q')
creg_c = ClassicalRegister(4, 'c')
grid_counts = 10
no_of_shots = 4096
backend = Aer.get_backend('aer_simulator')

best_gamma = 0
best_beta = 0
best_cut = 0
best_counts = None

start = time.time()

for gamma in np.linspace(-np.pi, np.pi, grid_counts):                #Grid search through possible gamma, beta parameters.
    for beta in np.linspace(-np.pi, np.pi, grid_counts):
        circuit = QuantumCircuit(qreg_q, creg_c)
        
        circuit.h(qreg_q) 
        
        for i, j, weight in G.edges(data='weight'):
            circuit.rzz(2 * gamma * weight, qreg_q[i], qreg_q[j])       #Cost layer.
        
        circuit.rx(2*beta, qreg_q)                                      #Mixer layer   
        
        for i in range(4):
            circuit.measure(qreg_q[i], creg_c[i])               #Measurement of each qubit and store in classical register bit.
            

        result = backend.run(circuit, shots=no_of_shots).result()
        counts = result.get_counts()
        
        avg_cut_value = 0 
        for partition, count in counts.items():
            partition = partition[::-1]                         #Corrects for qiskit endian convention (qubits are ordered in reverse)         
            cut_value = maxcut_value(partition, G)              #Evaluate how good the cut is. 
            avg_cut_value += cut_value * (count / no_of_shots)      #Average cut value sum
            
        if avg_cut_value > best_cut:
            best_cut = avg_cut_value
            best_gamma = gamma
            best_beta = beta
            best_counts = counts
        
end = time.time()
print("QAOA Max-Cut results:")
print(best_cut)
print(best_counts)
print(best_gamma)
print(best_beta)
print(f'Time-to-run = {(end - start):.4f}')
plot_histogram(best_counts)
plt.show()
