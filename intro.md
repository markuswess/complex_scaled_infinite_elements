# Implementation of frequency-dependent complex-scaled infinite elements

### *M. Wess, 2020*


***

In this notebook we explain the [Netgen/NgSolve](https://ngsolve.org)-implementation of the methods presented in the PhD-Thesis

M. Wess, *Frequency-Dependent Complex-Scaled Infinite Elements for Exterior Helmholtz Resonance Problems*, TU Wien, 2020, [reposiTUm](https://repositum.tuwien.at/handle/20.500.12708/15095) [pdf](https://repositum.tuwien.at/bitstream/20.500.12708/15095/2/Wess%20Markus%20-%202020%20-%20Frequency-dependent%20complex-scaled%20infinite%20elements%20for...pdf) [doi](https://doi.org/10.34726/hss.2020.78903)

which is concerned with numerical methods for the Helmholtz resonance problem to
\begin{align}
\text{find }(\omega,p),\text{ such that}\\
-\Delta p(x) -\omega^2 p(x) &= 0,& x&\in\Omega,\\
p&\text{ fulfills some b.c},&x&\in\partial\Omega,\\
p&\text{ is radiating},&\|x\|&\to\infty,
\end{align}
on unbounded domains $\Omega\in\mathbb R^d$.






## Abstract

Resonance phenomena occur, when waves in a given system are excited at certain frequencies.
For the mathematical analysis of resonances, time-harmonic waves (i.e., waves that are periodic in time, with respect to a given angular frequency) can be considered.
In this work we are concerned with the numerical analysis of the Helmholtz equation -- a partial differential equation that models, for example, time-harmonic acoustic waves -- on unbounded domains and the numerical analysis of the according resonance problems.

A popular method for treating such problems is the so-called complex scaling. The idea of this method is to introduce an artificial damping of the waves outside of a chosen computational interior domain in a way that no additional reflections are induced. When so-called perfectly matched layers are used, the exterior domain (i.e., the part of the domain where the damping is introduced) is truncated to a bounded layer and discretized using, for instance, finite elements. 
In the work at hand, we analyze, implement, and test a number of improvements to the method described above. 

To obtain a larger number of equally-well approximated resonances, we use method parameters that depend on the unknown resonance frequency.
This approach leads to non-linear eigenvalue problems, instead of linear ones. 

For the discretization of the problem, we use a method based on the decomposition of a wave into a propagating radial and an oscillating transversal part.
The discrete ansatz functions for the propagating part are functions with unbounded support and are closely related to the ansatz functions of Hardy space infinite element methods and spectral element methods.
Due to the use of these functions, we avoid the artificial truncation of the exterior domain and obtain super-algebraic approximation properties. Moreover, this decomposition makes it straightforward to adapt the method to the specific geometry of the given problem.

Lastly, we present an efficient method to approximate the eigenvalues of the resulting discrete, non-linear eigenvalue problems, which requires no significant extra computational effort, compared to similar methods for linear eigenvalue problems.

Numerical experiments underline our findings and exhibit the advantages of our method described above.

## Contents

In particular, we explain how the three main ideas of the thesis, namely

- [infinite elements](infinite_elements.ipynb)
- [tensor-product exterior discretizations by the use of exterior coordinates](tp_disc.ipynb)
- [frequency-dependent scaling functions and non-linear eigenvalue problems](fq_scaling.ipynb)

can be implemented using the finite element library [Netgen/NgSolve](https://www.ngsolve.org) and provide the following examples:

- [1. circular scalings](circular.ipynb)
- [2. star-shaped scalings](star-shaped.ipynb)
- [3. curvilinear scalings](curvilinear.ipynb)
- [4. different frequency-dependencies](fq_dep.ipynb)

All of the examples approximate resonances of the complement of a circle which are given by the zeros of the derivatives of Hankel functions of the first kind, and can therefore be computed semi-analytically.


## Some notes on system requirements 

*All of the examples require at least NGSolve-6.2.2008, a sufficiently recent version of [numpy](https://numpy.org) and [matplotlib](https://matplotlib.org) and, to run as a [jupyter-notebook](https://ngsolve.org/docu/latest/install/usejupyter.html), an installed [webgui](https://ngsolve.org/docu/latest/i-tutorials/index.html#installation).*

**[download code as .zip-file](fq_dp_cs_ie.zip)**

***

*by M. Wess, 2020, last edited 2025-06-11*

