# Quantum optimization for particle tracking problems

## Abstract

This project investigates the application of quantum optimization algorithms to the problem of particle tracking in high-energy physics experiments. Such solutions are invaluable in high-energy particle experiments, such as the [LHCb](https://home.cern/science/experiments/lhcb/) experiment at [CERN](https://home.cern/), as they can be used to identify and characterize final-state particles. 
A quantum based optimization method may offer an alternative to established classical approaches in the case of such combinatorially complex problems.

The project will run for 8 weeks. Each week developing an optimisation toolset and building the combinartorial complexity of the problem. We begin with a classical formulation of a simplified particle tracking simulation. Then, once the classical approach is clear, we will move to studying quantum approaches, implementing simple quantum optimisation algorithms such as the Quantum Approximate Optimization Algorithm (QAOA) or quantum variational circuits applied to combinatorial optimisation problems.

## Prerequisites 
- Python version 3.12.0 or higher

![LHCb Proton Tracks](assets/Images/ProtonCollisionTracks.jpg)

*Image from LHCb experiment, CERN.*

## Week 1

We create a simple toy data set with which we can build example similarity matrices. We use heat maps and graph networks as visualisation tools.

The simplest tracking problem, requiring some level of optimisation, is two non-intersecting tracks in 2-dimensional space. We construct this system so that we have 6 equally spaced detectors on the interval $[0,1]$ as shown. External noise is modelled as a Gaussian for each hit. 

![2D track setup](assets/plots/ClassicalPlots/Toytracks_12.png)

The similarity matrix elements $W_{ij}$ quantify the compatibility or correlation between hits i and j. The more correlated two hits, the more likely it is that they correspond to the same particle track. Intuitively, a simialrity matrix should be symmetric. 
In this project, two main types used are:

1. k-nearest neighbours (KNN)
KNN matrices are discrete. Given a hit $i$, we find its k nearest neighbours $j = α,β...$. Such elements are taken as      1, otherwise it is 0.
   
2. Radial Basis Function (RBF)
RBF matrices are continuous and depends on the distance between hits, $d(i,j)$. Here we choose the standard $L_{2}$ metric.
The RBF formula uses an exponential with values taken between $0$ and $1$. The standard deviation parameter $\sigma$ models the leaniency over which compatibility applies.

$$
W_{ij} = \exp\bigl(-\frac{d(i,j)^2}{(2\sigma^2)}\bigr)
$$
   
The smaller the hit seperation, the greater their RBF correlation. 


## Week 2

Having built a simple toy system of two non-intersecting particle tracks, we now need to determine which hit correponds to which track (assuming we don't already know). This can be viewed as an optimisation problem!

In order to employ optimisation technqiues we need to map each possible way of assigning hits to tracks a unique configuration in state or 'label' space. This we accomplish by map each configuration to a binary string of twelve 0's and 1's where 0 and 1 distinguish which of the two tracks each hit belongs to. 

E.g. The configuration $x = 010111010111$ says: hit 1 is in track 0, hit 2 is in track 1, hit 3 is in track 0 etc. 

We then map this binary string space into a spin sequence of $+1$ and $-1$. Mathematically this is done via,

$$
z_{i} = 2x_{i} - 1
$$

The constraint we place on the system is that the best configuration is the one that minimises the 'energy'. We develop an Ising-style Hamiltonian objective which admits the configuration spin sequence. Crucially, the Hamiltonian depends on the similarity matrix $W_{ij}$ in the same way as the magnetic coupling matrix does in the magnetism forumlation of the Ising model. 

$$
H = -\sum_{i<j} W_{ij} z_i z_j + \lambda \bigl(\sum_i z_i\bigr)^2
$$


Because we force $W$ to be symmetrical, the Hamiltonian is symmetric about flipping the signs of all spins in any sequence. This means that each energy level, whether we use the KNN or RBF matrix, will be at least two-fold degeneracy.
The second term acts as a penalty term, with the $\lambda > 0$ parameter enforcing how strict the peanlty term should be. This penalty term balances the first term by discouraging the all-in-one configurations of
111111111111 and 000000000000. 

Our expectation is of course that 000000111111 and its flipped state 111111000000 are the optimal groundstates. 

Visualisation of the system's energy landscape can be achieved by converting each configuration's binary string into its decimal equivalent and ordering the states numerically. 

## Week 3

The exhaustive brute force method from Week 2 works well for small values of N. If $N=12$ then there are only $2^{N} = 4096$ possible cluster configurations, but this expoential growth severely caps performance. Even at $N=20$, there would be over one million possible configurations! 

We now implement and compare three new and targeted approaches to finding the best clustering for our simple tracking problem. These will serve as our classical baselines that we will compare with later quantum-based algorithms.

1. Greedy Algorithm
2. Spectral Clustering
3. Simulated Annealing

Performance metrics used to numerically compare each algorithm are runtime, ARI (of estimated groundstate against true groundstate) and relative error to the true groundstate energy. We expect Greedy to perform the worst. Being myopic means decisions are made locally which can severely bias the resultant configuration early in the algorithm leading to near-poor clusterings. Spectral Clustering on the other hand is a graph-minded approach with global decision making based on Graph topology.

Simulated Annealing can be seen as the classical twin of our later quantum methods. Thus, it will be the most important of our classical comparatives. Based on the well-known Metropolis Acceptance Criterion, a random initial configuration is selected and state-space is traversed stochatsically, all while being tempered by a cooling scheme. Ideally it approaches the global minimum but can very easily get trapped in a local minimum. Convergence traces, shown below, detail the energy evolution through the algorithm. Hence optimisation depends on the temperature, $T$, evolution. As $T$ decreases, overcoming energy barriers becomes harder leading to the trapping phenomenon in the energy landscape. 

## Week 4

We have formulated and compared four classical baseline algorithms. In our small scale problem Simulated Annealing and Spectral Clustering perform brilliantly compared to Greedy Clustering, both returning the correct clusterings at the cost of slightly slower overall runtimes. Now we turn to formulating a simple quantum algorithm to tackle our problem! 

Our approach relies on the well-known Quantum Approximation Optimisation Algorithm (QAOA). Established by Farhi et al. in 2014, QAOA was applied to the Maximum Cut problem and originated from the Quantum Adiabatic Algorithm, (the mathematical structure of the circuit used comes from a first order Trotter-Suzuki approximation of the adiabatic Hamiltonian). 
In the MaxCut problem, we are given a weighted undirected graph where our aim is to find the best way to partition the graph into two compementary node sets. The best cut is the one that maximises the sum of weights of the cut edges. 
To gain familiarity with QAOA, I used the MaxCut problem as a starting point. 


## QAOA - An overview 

QAOA uses quantum mechanics to build up an optimal solution using tuning parameters. The biggest departure from classical algorithms, like those discussed above, lies in how the current 'state' moves in through the energy landscape. 
In a system of N hits, we have $2^N$ possible configurations. 
The Simulated Annealing state will move through configuration space step-by-step based on both an energy criterion and a cooling scheme. 
QAOA takes a fundamentally different approach. The QAOA state is described by a statevector with $2^N$ amplitudes, one for each configuration. Upon measuring the statevector we collapse it to a single confiugration. Which state we observe is determined by a probability distribution function (PDF) that depends directly on the squared magnitudes of the complex statevector amplitudes.
Our job as the QAOA circuit designer is to malnipulate the statevector, using a careful arrangement of quantum gates and tuning parameters, so that the probability of observing the desirable low-energy configuration is amplified, whereas higher energy configurations are suppressed.

How is this careful arrangement of gates and tuning parameters acheived? Each hit corresponds to one qubit. Measuring in the Z-basis, a $|0\rangle$ assigns the hit in group 0, and |1\rangle assigns to group 1. Preparing our quantum system in the $|00\ldots0\rangle$, we proceed in layers:

# Hadamard Layer 
We begin by applying a Hadamard gate to every qubit. This puts the statevector into a uniform superposition where the statevector PDF is a uniform distribution: 

$$ 
|+\rangle^{\otimes N} = \frac{1}{\sqrt{2^N}} \sum_{x \in \{0,1\}^N} |x\rangle. 
$$

The probability of measuring any confiugraiton is the same value of $\frac{1}{2^N}$. At this point measuring the state wouldn't do us any good! We need to construct a way of building a more favourable PDF.

# Cost Layer - The Cost Hamiltonian
Recall the classical Hamiltonian from Week 2, which I denote $H_{c}$. This Hamiltonian defines our energy landscape so it makes sense to also use it in the QAOA. However we must tranlate from classical variables into quantum (unitary) operators as such:

$$
H_{c} = - \sum_{i < j} W_{ij} Z_{i} Z_{j} + \lambda \bigl(\sum_{i} Z_{i}\bigr)^2.
$$

Which we can simplify to make computation easier by expanding the balance term:

$$
H_{c} = - \sum_{i < j} J_{ij} Z_{i} Z_{j},
$$

where $J_{ij} = 2\lambda - W_{ij}$ defines an effective coupling matrix between spins. $H_{c}$ is the Cost Hamiltonian. Being constructed purely of independent $Z$ operators, each particle track configuration is an eigenstate. Crucially this means the energy landscape is encoded in the this Hamiltonian. 

We implement this Hamiltonian into the QAOA through a series of parameterised two-qubit $R_{ZZ}$ gates, which capture the pairwise qubit interactions. Mathematically $R_{ZZ}$ take the form 

$$
\exp\left(-i\gamma J_{ij}\left(Z_{i} Z_{j}\right)\right),
$$

for rotation angle $2\gamma$. The phases of like spins ($|00\rangle$ and $|11\rangle$) are rotated equal and oppositely to opposing spins ($|10\rangle$ and $|01\rangle$). 
Now we are in a position to construct a cost layer with tuning parameter $\gamma$. We apply an $R_{ZZ}$ gate to all compatible hits as give by the similarity matrix $W_{ij}$. The effect of this is to multiply each amplitude $a(z)$, for all configurations z, by a phase factor that depends on the configuration energy, $E(z)$. Concretely each amplitude shifts as:

$$
a(z) \rightarrow a(z) \exp\left(-i \gamma E\left(z \right)\right).
$$

This operation does not change the configuration probability distribution. Instead, each complex amplitude is rotated by an angle $\gamma E\left(z\right)$. The angle of rotation is proportional the configuration energy. 
All we have done is written each configuration's energy into the corresponding amplitudes phase, setting up the mixer layer.


# Mixer Layer - The Mixer Hamiltonian
The Cost Layer alone will not produce anything computationally useful. It encodes configuration energies to amplitudes. We need a way of changing these amplitudes to increase the probability of getting the lowest-energy configuration.

The Mixer Layer is also based a Hamiltonian function called the Mixer Hamiltonian, $H_{M}$, taking the form

$$
H_{M} = \sum_{i} X_{i}.
$$

The reasoning behind using this form is that we can use **interference**, (and also because $H_{M}$ and $H_{C}$ do not commute).
Implementing this requires single-qubit R_{X} gates which rotates each qubit by an angle $2 \beta$. This $\beta$ is our other tuning parameter. 
The effect of the mixer layer is to blend amplitudes using interference. Specifically, amplitudes can flow between configurations that differ by exactly one bit. For example, the amplitudes of $|000\rangle$ can interact and mix with that of $|001\rangle$, $|010\rangle$ and $|100\rangle$. The amplitude 'flow' means that similar phases tend to constructively interfere while opposing phases tend to destructively interfere, based on the vector-like geometry of complex numbers. 

Which amplitudes are amplified and suppressed depends on the energy-dependent relative phases encoded in the Cost Layer through $\gamma$, and on the strength of the amplitude mixing controlled by $\beta$ in the Mixer Layer. This is where we must optimise for $\gamma$ and $\beta$. Neither Cost or Mixer Layer can act alone, and poorly selected tuning parameter values will inevitably damage the delicate interplay between these operations.

Following the mixer layer we can now perform measurment on each qubit and ideally observe the lowest energy configuration.

# Multi-layered QAOA
The QAOA circuit consisting of:

$$
|\psi(\gamma, \beta)\rangle = U_{M}(\beta) U_{C}(\gamma) |\psi_{0}\rangle
$$

can be generalised to $p$ cost+mixer layers.

$$
|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{k=1}^{p} U_M(\beta_k) U_C(\gamma_k) |\psi_0\rangle
$$

Where now we have $2p$ tuning parameters. The benefit of adding more layers is expressivity, the more fine-tuned we can make the QAOA the more flexible we can make our final state and potentially concentrate more probability on desirable configurations. However, as we shall explore in week 6, in the current NISQ era of quantum computing, we encounter a lot of noise, creating a noise vs expressivity tradeoff. 









