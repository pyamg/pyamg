"""Basic PyAMG demo showing AMG standalone convergence versus preconditioned CG with AMG."""

import numpy as np

from .laplacian import poisson
from ..aggregation.aggregation import smoothed_aggregation_solver


def demo():
    """Outline basic demo."""
    A = poisson((100, 100), format='csr')  # 2D FD Poisson problem
    B = None                               # no near-null spaces guesses for SA
    b = np.random.rand(A.shape[0], 1)      # a random right-hand side

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
    standalone_residuals = np.array(standalone_residuals) / standalone_residuals[0]
    accelerated_residuals = np.array(accelerated_residuals) / accelerated_residuals[0]

    # Compute (geometric) convergence factors
    factor1 = standalone_residuals[-1]**(1.0/len(standalone_residuals))
    factor2 = accelerated_residuals[-1]**(1.0/len(accelerated_residuals))

    print(f'                     MG convergence factor: {factor1}')
    print(f'MG with CG acceleration convergence factor: {factor2}')

    # Plot convergence history
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        plt.figure()
        plt.title('Convergence History')
        plt.xlabel('Iteration')
        plt.ylabel('Relative Residual')
        plt.semilogy(standalone_residuals, label='Standalone', linestyle='-', marker='o')
        plt.semilogy(accelerated_residuals, label='Accelerated', linestyle='-', marker='s')
        plt.legend()
        plt.show()
    except ImportError:
        print('\nNote: matplotlib is needed for plotting.')
