"""Test classical AMG."""
import warnings

import numpy as np

from numpy.testing import TestCase
from scipy.sparse import csr_matrix, bsr_matrix, diags
from pyamg.classical.air import air_solver
from pyamg.gallery import poisson

# 2-level reduction on 1d upwind advection should be exact in
# one iteration; test on scalar and block bidiagonal matrix.
class TestAIR(TestCase):
    def test_upwind_advection(self):
        sizes = []
        sizes.append(100)
        sizes.append(275)

        for n in sizes:
            # CSR case
            A = diags([np.ones((n,)), -1*np.ones((n-1,))],[0,-1]).tocsr()
            f_relax = ('fc_jacobi', {'iterations': 1,'f_iterations': 1,
                      'c_iterations': 0} )
            ml = air_solver(A, postsmoother=f_relax, max_levels=2)

            res = []
            x = np.random.rand(A.shape[0])
            b = A*np.random.rand(A.shape[0])  # zeros_like(x)
            x_sol = ml.solve(b, x0=x, maxiter=1, tol=1e-12, residuals=res)
            del x_sol
            assert(res[1] < 1e-12)

            # BSR case
            A.sort_indices()
            b1 = np.array([1,-1,1,1])
            b2 = -1*b1
            bb = np.concatenate((b1,b2))
            data = np.concatenate( (np.tile(bb,n-1),b1)).reshape((2*n-1,2,2))
            rowptr = A.indptr
            colinds = A.indices
            Ab = bsr_matrix((data,colinds,rowptr),blocksize=[2,2])
            f_relax = ('fc_block_jacobi', {'iterations': 1,'f_iterations': 1,
                      'c_iterations': 0} )
            ml = air_solver(Ab, postsmoother=f_relax, max_levels=2)

            res = []
            xb = np.random.rand(Ab.shape[0])
            bb = Ab*np.random.rand(Ab.shape[0])  # zeros_like(x)
            x_sol = ml.solve(bb, x0=xb, maxiter=1, tol=1e-12, residuals=res)
            del x_sol
            assert(res[1] < 1e-12)

    def test_poisson(self):
        cases = []

        cases.append(((500,), 0.001))
        cases.append(((50,50), 0.7))
        for case, expected_speed in cases:
            A = poisson(case, format='csr')

            for interp in ['direct', 'inject', 'one_point',
                           ('classical', {'modified': False}),
                           ('classical', {'modified': True})]:
                for restrict in [('air', {'theta': 0.05, 'degree': 1}), 
                                 ('air', {'theta': 0.05, 'degree': 2})]:

                    np.random.seed(0)  # make tests repeatable
                    x = np.random.rand(A.shape[0])
                    b = A*np.random.rand(A.shape[0])  # zeros_like(x)

                    ml = air_solver(A, interpolation=interp, restrict=restrict, max_coarse=50)

                    res = []
                    x_sol = ml.solve(b, x0=x, maxiter=20, tol=1e-12,
                                     residuals=res)
                    del x_sol

                    avg_convergence_ratio = (res[-1]/res[0])**(1.0/len(res))
                    assert (avg_convergence_ratio < expected_speed)
