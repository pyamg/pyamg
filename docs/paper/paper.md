---
title: 'PyAMG: Algebraic Multigrid Solvers in Python'
tags:
  - Python
  - algebraic multigrid
  - preconditioning
  - sparse matrix
  - Krylov
  - iterative methods
authors:
  - name: Nathan Bell
    affiliation: 1
  - name: Luke N. Olson
    orcid: 0000-0002-5283-6104
    affiliation: 2
  - name: Jacob Schroder
    orcid: 0000-0002-1076-9206
    affiliation: 3
affiliations:
 - name: Google, Mountain View, CA, USA
   index: 1
 - name: Department of Computer Science, University of Illinois at Urbana-Champaign, Urbana, IL USA 61801
   index: 2
 - name: Department of Mathematics and Statistics, University of New Mexico, Albuquerque, NM USA 87131
   index: 3
software_repository_url:
 - https://github.com/pyamg/pyamg
date: 22 January 2022
bibliography: paper.bib
---
\fvset{frame=lines}

# Statement of need

`PyAMG` is a Python package of algebraic multigrid (AMG) solvers and supporting
tools for approximating the solution to large, sparse linear systems of
algebraic equations,
$$A x = b,$$
where $A$ is an $n\times n$ sparse matrix.
Sparse linear systems arise in a range of
problems in science, from fluid flows to solid mechanics to data analysis.
While the direct solvers available in SciPy's sparse linear algebra package
(`scipy.sparse.linalg`) are highly efficient, in many cases *iterative* methods
are preferred due to overall complexity.  However, the iterative methods in
SciPy, such as CG and GMRES, often require an efficient preconditioner in order
to achieve a lower complexity.  Preconditioning is a powerful tool whereby the 
conditioning of the linear system and convergence rate of the iterative method
are both dramatically improved.
`PyAMG` constructs multigrid solvers for use as a
preconditioner in this setting.  A summary of multigrid and algebraic multigrid
solvers can be found in @encamg, in @encmg, and in @amgintro; a detailed description can be found
in @mgtutorial and @mgbook.

# Summary

The overarching goals of `PyAMG` include both readability and performance.
This includes readable implementations of popular variations of AMG (see the
Methods section), the ability to reproduce results in the literature, and a user-friendly
interface to AMG allowing straightforward access to the variety of AMG parameters
in the method(s). Additionally, pure Python implementations are not efficient for many sparse matrix
operations not already available in `scipy.sparse` --- e.g., the sparse matrix graph 
coarsening algorithms needed by AMG. For such cases in `PyAMG`, the compute (or
memory) intensive kernels are typically expressed in C++ and wrapped through PyBind11, while the method
interface and error handling is implemented directly in Python (more in the next section). 
\medskip

In the end, the goal of `PyAMG` is to provide quick access, rapid prototyping of new AMG solvers,
and performant execution of AMG methods.  The extensive PyAMG 
[Examples](https://github.com/pyamg/pyamg-examples) page highlights many of the package's
advanced AMG capabilities, e.g., for Hermitian, complex, nonsymmetric, and other challenging system types. 
It is important to note that many other AMG packages exist, mainly with a focus on parallelism and performance, rather than quick access and rapid prototyping.
This includes BoomerAMG in hypre [@henson2002155;@hypre], MueLu in Trilinos [@muelu-website;@trilinos-website], and GAMG within PETSc [@petsc-web-page], along with other packages focused on accelerators [@amggpu], such as AmgX [@amgx], CUSP [@cusp], and AMGCL [@amgcl].

# Design

The central data model in `PyAMG` is that of a `MultiLevel` object, which is
constructed in the *setup* phase of AMG.  The multigrid hierarchy is expressed
in this object (details below) along with information for the *solve* phase, which can be executed
on various input data, $b$, to solve $A x = b$.

The `MultiLevel` object consists of a list of multigrid `Level` objects and diagnostic
information.  For example, a `MultiLevel` object named `ml` contains the list
`ml.levels`.  Then, the data on level `i` (with the finest level denoted `i=0`)
accessible in `ml.levels[i]` includes the following information:

- `A`: the sparse matrix operator, in CSR or BSR format, on level `i`;
- `P`: a sparse matrix interpolation operator to transfer grid vectors from level `i+1` to `i`;
- `R`: a sparse matrix restriction operator to transfer grid vectors from level `i` to `i+1`; and
- `presmoother`, `postsmoother`: functions that implement pre/post-relaxation in the solve phase, such as weighted Jacobi or Gauss-Seidel.

Other data may be retained for additional diagnostics, such as grid
splitting information, aggregation information, etc., and would be included
in each level.

Specific multigrid methods (next section) in `PyAMG` and their parameters are generally described
and constructed in Python, while key performance components of both the setup and solve phase
are written in C++.  Heavy looping that cannot be accomplished with vectorized
or efficient calls to NumPy or sparse matrix operations that are not readily
expressed as SciPy sparse (CSR or CSC) operations are contained in short,
templated C++ functions.  The templates are used to avoid type recasting the variety
of input arrays. The direct wrapping to Python is handled through another layer
with PyBind11.  Roughly 26\% of PyAMG is in C++, with the rest in Python.

# Methods

`PyAMG` implements several base AMG methods, each with a range of options.  The base forms
for a solver include

- `ruge_stuben_solver()`: the classical form of C/F-type AMG [@cfamg:1987];
- `smoothed_aggregation_solver()`: smoothed aggregation based AMG as introduced in [@aggamg:1996];
- `adaptive_sa_solver()`: a so-called adaptive form of smoothed aggregation from [@adaptiveamg:2005]; and
- `rootnode_solver()`: the root-node AMG method from [@rootnodeamg:2017], applicable also to some nonsymmetric systems.

In each of these, the *base* algorithm is available but defaults may be
modified for robustness.  Options such as the default pre/postsmoother or smoothing the
input candidate vectors (in the case of smoothed aggregation AMG), can be
modified to tune the solver.  In addition, several cycles are available,
including the standard V and W cycles, for the solve phase.  The resulting
method can also be used as a preconditioner within the Krylov
methods available in `PyAMG` or with SciPy's Krylov methods.  The methods in
`PyAMG` (generally) support complex data types and nonsymmetric matrices.  

# Example

As an example, consider a five-point finite difference approximation to a
Poisson problem, $-\Delta u = f$, given in matrix form as $A x = b$.  The
AMG setup phase is called with
```{.python .numberLines}
import pyamg
A = pyamg.gallery.poisson((10000,10000), format='csr')
ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
```
For this case, with 100M unknowns, the following multilevel hierarchy
is generated for smoothed aggregation (using `print(ml)`):
```
MultilevelSolver
Number of Levels:     9
Operator Complexity:  1.338
Grid Complexity:      1.188
Coarse Solver:        'pinv'
  level   unknowns     nonzeros
     0   100000000    499960000 [74.76%]
     1    16670000    149993328 [22.43%]
     2     1852454     16670676 [2.49%]
     3      205859      1852805 [0.28%]
     4       22924       208516 [0.03%]
     5        2539        23563 [0.00%]
     6         289         2789 [0.00%]
     7          34          332 [0.00%]
     8           4           16 [0.00%]
```
In this case, the hierarchy consists of nine levels, with SciPy's pseudoinverse (`pinv`)
being used on the coarsest level. Also displayed is the ratio of unknowns (nonzeros) on all levels
compared to the fine level, also known as the grid (operator) complexity.

The solve phase, using standard V-cycles, is executed with the object's solve:
```{.python .numberLines}
import numpy as np
x0 = np.random.rand(A.shape[0])
b = np.zeros(A.shape[0])
res = []
x = ml.solve(b, x0, tol=1e-8, residuals=res)
```
This leads to the residual history shown in \autoref{fig:example}.

![Algebraic multigrid convergence (relative residual).\label{fig:example}](example.pdf)

# References
