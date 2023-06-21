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
  - name: Ben Southworth
    orcid: 0000-0002-0283-4928
    affiliation: 4
affiliations:
 - name: Google, Mountain View, CA, USA
   index: 1
 - name: Department of Computer Science, University of Illinois at Urbana-Champaign, Urbana, IL USA 61801
   index: 2
 - name: Department of Mathematics and Statistics, University of New Mexico, Albuquerque, NM USA 87131
   index: 3
 - name: Los Alamos National Laboratory, Los Alamos, NM USA 87545
   index: 4
software_repository_url:
 - https://github.com/pyamg/pyamg
date: 17 April 2023
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
in @mgtutorial and @mgbook. PyAMG provides a comprehensive suite of AMG
solvers (see Methods), which is beneficial because many AMG
solvers are specialized for particular problem types.

# Summary

The overarching goals of `PyAMG` include both readability and performance.
This includes readable implementations of many popular variations of AMG (see the
Methods section), the ability to reproduce results in the literature, and a user-friendly
interface to AMG allowing straightforward access to the variety of AMG parameters
in the method(s). Additionally, pure Python implementations are not efficient for many sparse matrix
operations not already available in `scipy.sparse` --- e.g., the sparse matrix graph 
coarsening algorithms needed by AMG. For such cases in `PyAMG`, the compute (or
memory) intensive kernels are typically expressed in C++ and wrapped through PyBind11, while the method
interface and error handling is implemented directly in Python (more in the next section). 
\medskip

In the end, the goal of `PyAMG` is to provide quick access, rapid prototyping of new AMG solvers, including easy comparison with many existing variations of AMG in the literature,
and performant execution of AMG methods. The extensive PyAMG 
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

`PyAMG` implements among the most wide-ranging suites of base AMG methods, each with a range of options.  The base forms
for a solver include

- `ruge_stuben_solver()`: the classical form of C/F-type AMG [@cfamg:1987];
- `smoothed_aggregation_solver()`: smoothed aggregation based AMG as introduced in [@aggamg:1996];
- `pairwise_solver()`: pairwise (unsmoothed) aggregation based AMG as introduced in [@notay:2010];
- `adaptive_sa_solver()`: a so-called adaptive form of smoothed aggregation from [@adaptiveamg:2005]; and
- `rootnode_solver()`: the root-node AMG method from [@rootnodeamg:2017], applicable also to some nonsymmetric systems.
- `air_solver()`: the nonsymmetric AMG method based on approximate ideal restriction (AIR) from [@air1; @air2], which is highly effective for many upwind discretizations of advection-dominated problems.


In each of these, the *base* algorithm is available but defaults may be
modified for robustness.  Options such as the default pre/postsmoother or smoothing the
input candidate vectors (in the case of smoothed aggregation or root-node AMG), can be
modified to tune the solver.  In addition, several cycles are available,
including the standard V, F, and W cycles, for the solve phase.  The resulting
method can also be used as a preconditioner within the Krylov
methods available in `PyAMG` or with SciPy's Krylov methods.  The methods in
`PyAMG` generally support complex data types and nonsymmetric matrices. 
All `MultiLevel` objects provide a detailed measure of the grid complexity
(number of unknowns on all levels / number of unknowns on the finest level),
operator complexity (number of nonzeros in the matrix on all levels /
number of nonzeros in the matrix on the finest level), and cycle complexity 
(approximate cost in floating point operations (FLOPs) of a single
multigrid cycle relative to a single matrix-vector multiply). 

# Example

As an example, consider a five-point finite difference approximation to a
Poisson problem, $-\Delta u = f$, given in matrix form as $A x = b$.  The
AMG setup phase is called with
```{.python .numberLines}
import pyamg
A = pyamg.gallery.poisson((1000,1000), format='csr')
ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
```
For this case, with 1M unknowns, the following multilevel hierarchy
is generated for smoothed aggregation (using `print(ml)`):
```
MultilevelSolver
Number of Levels:     7
Operator Complexity:   1.338
Grid Complexity:       1.188
Coarse Solver:        'pinv'
  level   unknowns     nonzeros
     0     1000000      4996000 [74.75%]
     1      167000      1499328 [22.43%]
     2       18579       167051 [2.50%]
     3        2086        18870 [0.28%]
     4         233         2109 [0.03%]
     5          28          248 [0.00%]
     6           3            9 [0.00%]
```
In this case, the hierarchy consists of seven levels, with SciPy's pseudoinverse (`pinv`)
being used on the coarsest level. Also displayed is the ratio of unknowns (nonzeros) on all levels
compared to the fine level, also known as the grid (operator) complexity.

The solve phase, using standard V-cycles, is executed with the object's solve:
```{.python .numberLines}
import numpy as np
x0 = np.random.rand(A.shape[0])
b = np.zeros(A.shape[0])
res = []
x = ml.solve(b, x0, tol=1e-10, residuals=res)
```
This leads to the residual history shown in \autoref{fig:example}.
Additional examples can be found at
[github.com/pyamg/pyamg-examples](https://github.com/pyamg/pyamg-examples),
including examples with classical AMG using AIR, building solvers with rootnode, and nonsymmetric use cases.

![Algebraic multigrid convergence (relative residual).\label{fig:example}](example.pdf)

# References
