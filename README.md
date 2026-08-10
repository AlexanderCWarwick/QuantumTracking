# Quantum optimization for particle tracking problems

## Abstract

This project investigates the application of quantum optimization algorithms to the problem of particle tracking in high-energy physics experiments. Such solutions are invaluable in high-energy particle experiments, such as the [LHCb](https://home.cern/science/experiments/lhcb/) experiment at [CERN](https://home.cern/), as they can be used to identify and characterize final-state particles. 
A quantum based optimization method may offer an alternative to established classical approaches in the case of such combinatorially complex problems.
Critically, this project does not seek algorithms that find an unknown answer to the tracking problem. We construct a simplfied problem for which the answer is known, allowing comparison of the different optimisation methods. 
Instead, the aim is to:
- Learn the principles of QAOA optimisation.
- Learn whether a quantum approach might be valueable in solving this specific combinatorial problem.
- Compare the performance of well-established classical methods against a quantum based algorithm.

The project will run for 8 weeks. Each week developing an optimisation toolset and building the combinartorial complexity of the problem. We begin with a classical formulation of a simplified particle tracking simulation. Then, once the classical approach is clear, we will move to studying quantum approaches, implementing simple quantum optimisation algorithms such as the Quantum Approximate Optimization Algorithm (QAOA) or quantum variational circuits applied to combinatorial optimisation problems.

## Prerequisites 
- Python version 3.12.0 or higher

![LHCb Proton Tracks](assets/Images/ProtonCollisionTracks.jpg)

*Image from LHCb experiment, CERN.*

## Table of Contents

- [Week 1 - Setting the Problem Up](#week-1---introduction)
- [Week 2 - Brute Force Eenrgy Groundstate Search](#week-2---classical-methods)
    - [Energy Landscape](#energy-landscape)
- [Week 3 - Ising Formulation](#week-3---ising-formulation)
- [Week 4 - Building QAOA](#week-4---building-qaoa)
    - [QAOA - An Overview](#qaoa---an-overview)
- [Week 5 - A Better QAOA Optimiser](#week-5---a-better-qaoa-optimiser)
    - [Depth, Scale and niversal Scans](#depth-scale-and-universal-scans)
- [Week 6 - Adding Noise](#week-6---adding-noise)
- [Week 7/8 - Real Hardware Comparison](#week-7/8---real-hardware-comparison)
    - [Three-way Scan](#three-way-scan)

---

## Week 1 - Setting the Problem Up

We create a simple toy data set with which we can build example similarity matrices. We use heat maps and graph networks as visualisation tools.

The simplest tracking problem, requiring some level of optimisation, is two non-intersecting tracks in 2-dimensional space. We construct this system so that we have $N=6$ equally spaced detectors on the interval $[0,1]$ as shown. External noise is modelled as a Gaussian for each hit. Later on in the project we will want to test the effect of varying $N$ on our slected algorithms.

![2D track setup](assets/plots/ClassicalPlots/Toytracks_12.png)

The similarity matrix elements $W_{ij}$ quantify the compatibility or correlation between hits i and j. The more correlated two hits, the more likely it is that they correspond to the same particle track. Intuitively, a simialrity matrix should be symmetric. 
In this project, two main types used are:

1. k-nearest neighbours (KNN)
KNN matrices are discrete. Given a hit $i$, we find its k nearest neighbours $j = α,β...$. Such elements are taken as 1, otherwise it is 0. 
   
2. Radial Basis Function (RBF)
RBF matrices are continuous and depends on the distance between hits, $d(i,j)$. Here we choose the standard $L_{2}$ metric.
The RBF formula uses an exponential with values taken between $0$ and $1$. The standard deviation parameter $\sigma$ models the leaniency over which compatibility applies.

$$
W_{ij} = \exp\bigl(-\frac{d(i,j)^2}{(2\sigma^2)}\bigr)
$$
   
The smaller the hit seperation, the greater their RBF correlation. 

---

## Week 2 - Brute Force Eenrgy Groundstate Search

Having built a simple toy system of two non-intersecting particle tracks, we now use the similarity matrix to determine which hits correpond to which track (assuming we don't already know). This can be viewed as an optimisation problem!

### Energy Landscape

In order to employ optimisation technqiues we need to map each possible way of assigning hits to tracks a unique configuration in state or 'label' space. This we accomplish by map each configuration to a binary string of twelve 0's and 1's where 0 and 1 distinguish which of the two tracks each hit belongs to. 

E.g. The configuration $x = 010111010111$ says: hit 1 is in track 0, hit 2 is in track 1, hit 3 is in track 0 etc. 

We then map this binary string space into a spin sequence of $+1$ and $-1$. Mathematically this is done via,

$$
z_{i} = 2x_{i} - 1
$$

The constraint we place on the system is that the best configuration is the one that minimises the 'energy'. We develop an Ising-style Hamiltonian objective which admits the configuration spin sequence. Crucially, the Hamiltonian depends on the similarity matrix $W_{ij}$ in the same way as the magnetic coupling matrix does in the magnetism forumlation of the Ising model. 

$$
H = -\sum_{i<j} W_{ij} z_i z_j + \lambda \left(\sum_i z_i\right)^2 \tag_{1}
$$

Because we force $W$ to be symmetric, the Hamiltonian is symmetric about flipping the signs of all spins in any sequence. This means that each energy level, whether we use the KNN or RBF matrix, will be at least two-fold degeneracy. Crucially, there are two degenerate groundstate configurations.
The second term acts as a penalty term, with the $\lambda > 0$ parameter enforcing how strict the peanlty term should be. This penalty term balances the first term by discouraging the all-in-one configurations of $111111111111$ and $000000000000$. Our expectation is of course that $000000111111$ and its flipped state $111111000000$ are the optimal groundstates. 

### Brute force

Naively, we can find the system ground state by a brute force search. Each bitstring is mapped to an energy via Eq. (1), if we know the entire mapping we simply find the minimum, gauranteed for any $N$. Although, the size of our problem scales exponentially as $2^N$. Hence for small N, roughly $N \leq 12$, we can employ this brute force method with a reasonably small runtime. This is the method through which we find the true groundstates and its implementation is the purpose of Week 2.
(Insert Energy landscape plot)

---

## Week 3 - Classical Benchmark and Metrics

The exhaustive brute force method from Week 2 works well for small values of N. If $N=12$ then there are only $2^{N} = 4096$ possible cluster configurations, but this expoential growth severely caps performance. Even at $N=20$, there would be over one million possible configurations! 

We now implement and compare three new and targeted approaches to finding the best clustering for our simple tracking problem. These will serve as our classical baselines that we will compare with later quantum-based algorithms.

1. Greedy Algorithm
2. Spectral Clustering
3. Simulated Annealing

Performance metrics used to numerically compare each algorithm are runtime, ARI (of estimated groundstate against true groundstate) and relative error to the true groundstate energy. We expect Greedy to perform the worst. Being myopic means decisions are made locally which can severely bias the resultant configuration early in the algorithm leading to near-poor clusterings. Spectral Clustering on the other hand is a graph-minded approach with global decision making based on Graph topology.

Simulated Annealing can be seen as the classical twin of our later quantum methods. Thus, it will be the most important of our classical comparatives. Based on the well-known Metropolis Acceptance Criterion, a random initial configuration is selected and state-space is traversed stochatsically, all while being tempered by a cooling scheme. Ideally it approaches the global minimum but can very easily get trapped in a local minimum. Convergence traces, shown below, detail the energy evolution through the algorithm. Hence optimisation depends on the temperature, $T$, evolution. As $T$ decreases, overcoming energy barriers becomes harder leading to the trapping phenomenon in the energy landscape. 

Numerical performance analysis of each algorithm is realised through the following metrics:

1. Relative Energy Error (REE)
2. Runtime
3. Adjusted Random Index (ARI)

A successful run should return an REE of 0, and ARI of 1. Performance of the simulated annelaing (SA) algorithm can also be measured by the convergence success ratio. If we run the algorithm $m$ times, seeing as we know the true groundstates and their energies, we can test how many times, $n$, SA successfully converged to the true groundstate. The success fraction is $\frac_{n}{m}$.

---

## Week 4 - Building the QAOA!

We have formulated and compared four classical baseline algorithms. In our small scale problem Simulated Annealing and Spectral Clustering perform brilliantly compared to Greedy Clustering, both returning the correct clusterings at the cost of slightly slower overall runtimes. Now we turn to formulating a simple quantum algorithm to tackle our problem! Implementation is done all via [Qiskit](https://www.ibm.com/quantum/qiskit).


Our approach relies on the well-known Quantum Approximation Optimisation Algorithm (QAOA). Established by [Farhi et al.](https://arxiv.org/abs/1411.4028) in 2014, QAOA was applied to the [MaxCut Problem](https://en.wikipedia.org/wiki/Maximum_cut) and is closely related to the Quantum Adiabatic Algorithm.

This and following weeks are dedicated to building up the complexity of our QAOA. Beginning with the simplest case, using Qiskit's simulator `AerSimulator`, generalising to multi-layered circuits (Week 5) as well as implementing noise models (Week 6) to improve realism. To gain familiarity with QAOA, I used a simplified MaxCut problem as this weeks starting point, see MaxCutExmaple.py, using a simple grid search parameter optimisation. Once I had implemented a working QAOA MaxCut solution, I adapted it to the particle tracking problem. 
Seeing as QAOA is at the heart of this project, I have given an overview below.


### QAOA - An overview 

QAOA uses quantum mechanics to build up an optimal solution using **tuning parameters**. The biggest departure from classical algorithms, like those discussed above, lies in how the current 'state' moves in through the energy landscape. 
In a system of N hits, we have $2^N$ possible configurations. 
The Simulated Annealing state will move through configuration space step-by-step based on both an energy criterion and a cooling scheme. 
QAOA takes a fundamentally different approach. The QAOA state is described by a statevector with $2^N$ amplitudes, one for each configuration. Upon measuring the statevector we collapse it to a single confiugration. Which state we observe is determined by a probability distribution function (PDF) that depends directly on the squared magnitudes of the complex statevector amplitudes.
Our job as the QAOA circuit designer is to malnipulate the statevector, using a careful arrangement of quantum gates and tuning parameters, so that the probability of observing the desirable low-energy configuration is amplified, whereas higher energy configurations are suppressed.

How is this careful arrangement of gates and tuning parameters acheived? Each hit corresponds to one qubit. Measuring in the Z-basis, a $|0\rangle$ assigns the hit in group 0, and |1\rangle assigns to group 1. Preparing our quantum system in the $|00\ldots0\rangle$, we proceed in layers:

#### Hadamard Layer 
We begin by applying a Hadamard gate to every qubit. This puts the statevector into a uniform superposition where the statevector PDF is a uniform distribution: 

$$ 
|+\rangle^{\otimes N} = \frac{1}{\sqrt{2^N}} \sum_{x \in \{0,1\}^N} |x\rangle. 
$$

The probability of measuring any confiugraiton is the same value of $\frac{1}{2^N}$. At this point measuring the state wouldn't do us any good! We need to construct a way of building a more favourable PDF.

#### Cost Layer - The Cost Hamiltonian
Recall the classical Hamiltonian from Week 2, which I denote $H_{c}$. This Hamiltonian defines our energy landscape so it makes sense to also use it in the QAOA. However we must tranlate from classical variables into quantum (unitary) operators as such:

$$
H_{c} = - \sum_{i < j} W_{ij} Z_{i} Z_{j} + \lambda \bigl(\sum_{i} Z_{i}\bigr)^2 = \sum_{i<j} J_{ij} Z_{i} Z_{j}
\tag{1}
$$

where $J = 2\lambda - W$ defines an effective coupling matrix between spins. $H_{c}$ is the Cost Hamiltonian. Being constructed purely of independent $Z$ operators, each particle track configuration is an eigenstate. Crucially this means the energy landscape is encoded in the this Hamiltonian. 

Implementing Eq. (1) into the QAOA is acheived through a series of parameterised two-qubit $R_{ZZ}$ gates, which capture the pairwise qubit interactions. Mathematically $R_{ZZ}$ take the form 

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


#### Mixer Layer - The Mixer Hamiltonian
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

### Multi-layered QAOA
The QAOA circuit consisting of:

$$
|\psi(\gamma, \beta)\rangle = U_{M}(\beta) U_{C}(\gamma) |\psi_{0}\rangle
$$

can be generalised to $p$ cost+mixer layers.

$$
|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{k=1}^{p} U_M(\beta_k) U_C(\gamma_k) |\psi_0\rangle
$$

Where now we have $2p$ tuning parameters. The benefit of adding more layers is expressivity, the more fine-tuned we can make the QAOA the more flexible we can make our final state and potentially concentrate more probability on desirable configurations. However, as we shall explore in week 6, in the current NISQ era of quantum computing, we encounter a lot of noise, creating a noise vs expressivity tradeoff. 

### Choosing tuning parameters
Before we can actually run the circuit, we need to chose the tuning parameters $\boldsymbol{\gamma}$, $\boldsymbol_{\beta}$ to assign to cost and mixer layer gates. In Week 4, we optimise for their values with a simple grid search in the $2p$-dimensional parameter space. In this project I restrict the search domain $\gamma_{i} \in [0, 2\pi)$ and $\beta_{i} \in [0, \pi)$ for all $i$. 
The primary drawback of the Grid Search is that it is non-adaptive; it tests fixed points in the parameter space instead of using previous results to find a better region to test. In addition, using nested `for` loops over a uniformly $K$ spaced grid, adding more QAOA layers increases the Grid Search domain as $K^(2p)$. Hence it is hugely profitable to replace Grid Search with an adaptive optimiser if we are going to explore the effetc of adding more layers. Many such strategies exist and our choice is explored in Week 5.

### QAOA as a Hybrid Algorithm

Above in Week 4 I explained the structure of the QAOA machinery. But QAOA is actually implemented as a hybrid quantum-classical algorithm. This means that parts of the QAOA workflow where a quantum computation isn't suitable are delegated to a classical computer. While the quantum computer runs the circuits and returns the sampled probability distribution, the classical computer will perform the $\boldsymbol{\gamma}$, $\boldsymbol_{\beta}$ parameter optimisation. Optimisation is **not** carried out on a quantum computer. The algorithm workflow looks like this:

![QAOA_hybrid_workflow](assets/Images/QAOA_hybrid_worklow.png)

*Hybrid workflow of QAOA with p layers. 
Reproduced from Figure 3 of [Blekos et al.](https://arxiv.org/abs/2306.09198), licensed under CC BY 4.0.*

---

## Week 5 - A New Optimiser and Scans
In Week 4 I used the simplest possible parameter optimisation method, selecting a fixed set of independent points to test. However, as the parameter space grows, Grid Search becomes increasingly inefficient. If we want to explore the effect of adding more layers to the circuit, we need to consider a new optimiser. For this project, we will be analysing the gradient-free **COBYLA** (Constrained Optimisation BY Linear Approximation) method, available through SciPy's `minimize` function. Please see SciPy [COBYLA documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html). 
A wide range of other gradient-free methods are available that would also work in the context of this project, and include popular methods such as [Nelder-Mead](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html) and [CMA-ES](https://cma-es.github.io/apidocs-pycma/index.html). 


### Depth, Scale and Universal Scans
Having chosen a new optimiser, we can now explore how QAOA performance changes when we change the problem **scale** $N$, and QAOA circuit **depth** $p$, without relying on the expensive runtime of a Grid Search. We can also compare QAOA performance to the previous classical algorithms implemented in Week 3 using the same metrics: REE, ARI and runtime. Note however that because of QAOA's hybrid nature the runtime can be split into classical runtime and quantum runtime, here we simply measure the total algorithm runtime. All performance metric results are given as distributions, rather than single anecdotal points, computing the mean and standard deviation to give a more representative finding. 

We also introduce a new performance metric: Groundstate Probability (GSP). Given the number of shots, we compute an estimate of the probability of obtaining the groundstate upon measurement.

$$
\mathrm{GSP} = \frac{n_{\mathrm{GS}_{1}} + n_{\mathrm{GS}_{2}}}{N_{\mathrm{shots}}}
$$

where $n_{\mathrm{GS}_{1}$ and $n_{\mathrm{GS}_{2}}$ are the number of times each degenerate groundstate is sampled. This GSP metric is only applicable to the QAOA, but is mathematically similar to the convergence fraction we used in when studying SA. Please see Week 6 below for a plot of my results.

To provide an complete comparison between all classical and quantum algorithms, I also implemented a **Universal** Scan for fixed $N$ and $p$. The output table is a clear way to view the differences between each algorithm. 
**Importantly**, we should not expect QAOA to 'win'. 

---

## Week 6 - Adding Noise
Up until now we have worked on a noiseless simulator. As of now in 2026, real quantum hardware carries plenty of noise. Gates are imperfect, qubits can easily decohere and measurement is prone to error. Adding more operations increases the effect of error.
This week we implement a basic noise model using Qiskit's `NoiseModel` object, adding a single noise source through separate [depolarisation channels](https://en.wikipedia.org/wiki/Quantum_depolarizing_channel) for one- and two-qubit gates. We only consider the effects of one noise channel to study the impact of noise on the QAOA performance when $N$ or $p$ are varied.
Other models of quantum noise can be added including T1/T2 error and readout error to make the simulator more realistic and will be considered later in Week 7/8 when we actually use real hardware.


### Scale Scan: Clean vs Noisy

### Depth Scan: Clean vs Noisy



## Week 7/8 - Real Hardware Comparison
We have created a noisy simulator using Qiskit's `AerSimulator`, and we have compared it to our classical benchmarks in the universal scan. But how accurately does it compare to real quantum hardware? Seeing as we are using Qiskit, we can execute our circuit using IBM's Quantum Porcessing Units (QPUs) through the [IBM Quantum Platform](https://www.ibm.com/quantum?utm_content=SRCWW&p1=Search&p4=318569543695&p5=e&p9=194522864622&gclid=6bf19520fd951efefb05ff591a3581b2&gclsrc=3p.ds&msclkid=6bf19520fd951efefb05ff591a3581b2). Running a quantum circuitry on a QPU introduces hardware constraints that were absent in our idealised simulated circuit. Limited connectivity between qubits meaning a two-qubit operation between two arbitrary qubits cannot be performed directly between them. To account for connectivity overhead we transpile our ideal circuit, through Qiskit's `transpile` function. Mapping the logical qubits and gates in our idealised circuit to the physical qubits and native gates used by the real QPU, we taylor our circuit to a specific QPU architecture.
This transpiled circuit can infact be very different from the original circuit, potentially increasing depth and gate count substantially, thereby increasing exposure to noise. The table below highlights the effect of transpilation.

### Table summarising ideal vs transpiled circuit differcnes.


Importantly, the deliverable for Weeks 7 and 8 is the difference between noisy simulator and real hardware: does the noise model I built actually predict what real hardware does? We can analyse this difference through a three-way scan. A discrepency between the results can tell us which physical effects our noisy model doesn't factor in.

### Three-way Scan

