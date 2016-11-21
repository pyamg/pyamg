"""
Relaxation methods
------------------

The multigrid cycle is formed by two complementary procedures: relaxation and
coarse-grid correction.  The role of relaxation is to rapidly damp oscillatory
(high-frequency) errors out of the approximate solution.  When the error is
smooth, it can then be accurately represented on the coarser grid, where a
solution, or approximate solution, can be computed.

Iterative methods for linear systems that have an error smoothing property
are valid relaxation methods.  Since the purpose of a relaxation method is
to smooth oscillatory errors, its effectiveness on non-oscillatory errors
is not important.  This point explains why simple iterative methods like
Gauss-Seidel iteration are effective relaxation methods while being very
slow to converge to the solution of Ax=b.


PyAMG implements relaxation methods of the following varieties:
    1. Jacobi iteration
    2. Gauss-Seidel iteration
    3. Successive Over-Relaxation
    4. Polynomial smoothing (e.g. Chebyshev)
    5. Jacobi and Gauss-Seidel on the normal equations (A.H A and A A.H)
    6. Krylov methods: gmres, cg, cgnr, cgne
    7. No pre- or postsmoother

Refer to the docstrings of the individual methods for additional information.

"""

# TODO: explain separation of basic methods from interface methods.
# TODO: explain why each class of methods exist
# (parallel vs. serial, SPD vs. indefinite)

postpone_import = 1
