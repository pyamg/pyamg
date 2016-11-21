"""Basic PyAMG demo showing AMG standalone convergence versus preconditioned CG
with AMG"""
from __future__ import print_function

import scipy as sp
import numpy as np
from pyamg.gallery import poisson
from pyamg.aggregation import smoothed_aggregation_solver

__all__ = ['demo']


def demo():
    A = poisson((100, 100), format='csr')  # 2D FD Poisson problem
    B = None                               # no near-null spaces guesses for SA
    b = sp.rand(A.shape[0], 1)          # a random right-hand side

    # use AMG based on Smoothed Aggregation (SA) and display info
    mls = smoothed_aggregation_solver(A, B=B)
    print(mls)

    # Solve Ax=b with no acceleration ('standalone' solver)
    standalone_residuals = []
    x = mls.solve(b, tol=1e-10, accel=None, residuals=standalone_residuals)

    # Solve Ax=b with Conjugate Gradient (AMG as a preconditioner to CG)
    accelerated_residuals = []
    x = mls.solve(b, tol=1e-10, accel='cg', residuals=accelerated_residuals)
    del x

    # Compute relative residuals
    standalone_residuals = \
        np.array(standalone_residuals) / standalone_residuals[0]
    accelerated_residuals = \
        np.array(accelerated_residuals) / accelerated_residuals[0]

    # Compute (geometric) convergence factors
    factor1 = standalone_residuals[-1]**(1.0/len(standalone_residuals))
    factor2 = accelerated_residuals[-1]**(1.0/len(accelerated_residuals))

    print("                     MG convergence factor: %g" % (factor1))
    print("MG with CG acceleration convergence factor: %g" % (factor2))

    # Plot convergence history
    try:
        import pylab
        pylab.figure()
        pylab.title('Convergence History')
        pylab.xlabel('Iteration')
        pylab.ylabel('Relative Residual')
        pylab.semilogy(standalone_residuals, label='Standalone',
                       linestyle='-', marker='o')
        pylab.semilogy(accelerated_residuals, label='Accelerated',
                       linestyle='-', marker='s')
        pylab.legend()
        pylab.show()
    except ImportError:
        print("\n\nNote: pylab not available on your system.")
