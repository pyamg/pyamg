---
title: 'pyAMG: Algebraic Multigrid Solvers in Python'
tags:
  - Python
  - algebraic multigrid
  - preconditioning
  - sparse matrix
  - Krylov
  - iterative methods
authors:
  - name: Luke N. Olson
    orcid: 0000-0002-5283-6104
    affiliation: 1
  - name: Jacob Schroder
    orcid: 0000-0002-1076-9206
    affiliation: 2
affiliations:
 - name: Department of Computer Science, University of Illinois at Urbana-Champaign, Urbana, IL USA 61801
   index: 1
 - name: Department of Mathematics and Statistics, University of New Mexico, Albuquerque, NM USA 87131
   index: 2
date: 22 January 2022
bibliography: paper.bib
---

# Summary

`pyAMG` is a Python package of algebraic multigrid (AMG) solvers and supporting
tools for approximating the solution to large, sparse linear systems of
algebraic equations,
$$A x = b,$$
where $A$ is an $n\times n$ sparse matrix.
Sparse linear systems arise in a range of
problems in the science, from fluid flows to solid mechanics to data analysis.
While the direct solvers available in SciPy's sparse linear algebra package
(`scipy.sparse.linalg`) are highly efficient, in many cases *iterative* methods
are preferred due to overall complexity.  Likewise, the iterative methods in
SciPy, such as CG and GMRES, often require an efficient preconditioner in order
to improve convergence.  `pyAMG` is constructs multigrid solvers for use as a
preconditioner in this setting.  A summary of multigrid and algebraic multigrid
solvers can be found in [@encmg,@encamg]; a detailed description can be found
in [@mgtutorial,@mgbook].

The overarching goals of `pyAMG` include both readability and performance.
This includes readable implementations of popular variations of AMG (see the
Methods section), the ability to reproduce results in the literature, and a useable
interface to AMG allowing straightforward access to the variety of parameters
in the method(s). At the same time, pure Python may not be efficient for sparse matrix
operations that are not immediately expressed as efficient SciPy operations like a sparse
matrix-vector multiply `A @ x`.  For many of the cases in `pyAMG`, the method
interface and error handling is handled directly in Python, while compute (or
memory) intensive kernels are expressed in C++ and wrapped through PyBind11.
(more in the next section).

In the end, the goal of `pyAMG` is provide quick access, rapid prototyping,
and performant execution of AMG methods.

# Design

The central data model in `pyAMG` is that of a `MultiLevel` object, which is
constructed in the *setup* phase of AMG.  The multigrid hierarchy is expressed
in this object along with details of the *solve* phase (which can be executed
on various input data, $b$).

The `MultiLevel` object consists of a list of `Level`s and diagnostic
information.  For example, a `MultiLevel` object named `ml` contains
`ml.levels`.  Then, the data on level `i` (with the fine level denoted `i=0`),
in `ml.levels[i]`, includes
  - `A`: the sparse matrix operator, in CSR format, on level `i`;
  - `P`: a sparse matrix interpolation operator to transfer grid vectors from level `i+1` to `i`;
  - `R`: a sparse matrix restriction operator to transfer grid vectors from level `i` to `i+1`; and
  - `presmoother`, `postsmoother`: functions that implement pre/post-relaxation in the solve phase, such as weighted Jacobi or Gauss-Seidel.
Other information, may be contained for additional diagnostics, such as grid
splitting information, aggregation information, etc.

Specific multigrid methods (next section) in `pyAMG` and their parameters are generally described
and constructed in Python, while key components of both the setup and solve phase
are written in C++.  Heavy looping that cannot be accomplished with vectorized
or efficient calls to NumPy or sparse matrix operations that are not readily
expressed as ScyPy sparse (CSR or CSC) operations are contained in short,
templated C++ functions.  The templates are used to avoid type recasting the variety
of input arrays, and the direct wrapping to Python is handled through another layer
with PyBind11.

# Methods

`pyAMG` implements several base methods, each with a range of options.  The base forms
for a solver include

- `ruge_stuben_solver()`: the classical form of C/F-type AMG [@cfamg:1987];
- `smoothed_aggregation_solver()`: smoothed aggregation based AMG as introduced in [@aggamg:1996];
- `adaptive_sa_solver()`: a so-called adaptive form of smoothed aggregation from [@adaptiveamg:2005]; and
- `rootnode_solver()`: the root-node AMG method from [@rootnodeamg:2017].

In each of these, the *base* algorithm is available but defaults may be
modified for robustness.  Options such as the default smoother or smoothing the
input candidate vectors (in the case of smoothed aggregation AMG), can be
modified to tune the solver.  In addition, several cycles are available,
including the standard V and W cycles, for the solve phase.  The resulting
method can also be used in the form of a preconditioner within the Krylov
methods available in `pyAMG` or with SciPy's Krylov methods.  The methods in
`pyAMG` (generally) support for complex data types and non-symmetric matrices.  

# Example

As an example, consider a five-point finite difference approximation to a
Poisson problem, $-\Delta u = f$, given in matrix form as $A x = b$.  The
AMG setup phase is called with
```python
A = pyamg.gallery.poisson((10000,10000), format='csr')
ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
```
For this case, with 100M unknowns, the following multilevel hierarchy
is generated for smoothed aggregation:
```
MultilevelSolver
Number of Levels:     9
Operator Complexity:  1.338
Grid Complexity:      1.188
Coarse Solver:        'pinv'
  level   unknowns     nonzeros
     0   100000000    499960000 [74.76%]
     1    16670000    149993328 [22.43%]
     2     1852454     16670676 [ 2.49%]
     3      205859      1852805 [ 0.28%]
     4       22924       208516 [ 0.03%]
     5        2539        23563 [ 0.00%]
     6         289         2789 [ 0.00%]
     7          34          332 [ 0.00%]
     8           4           16 [ 0.00%]
```
In this case, the hierarchy consists of nine levels, with SciPy's pseudoinverse ('pinv')
being used on the coarsest level.

The solve phase, using standard V-cycles, is executed with the object's solve:
```python
x0 = np.random.rand(A.shape[0])
b = np.zeros(A.shape[0])
res = []
x = ml.solve(b, x0, tol=1e-8, residuals=res)
```
This leads to the residual history shown in \autoref{fig:example}.

![Algebraic multigrid convergence (relative residual).\label{fig:example}](example.pdf)

# References
