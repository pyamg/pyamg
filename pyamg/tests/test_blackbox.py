"""Teset blackbox.py solver"""
import warnings
import numpy as np
from numpy.testing import TestCase
from pyamg.gallery import poisson, load_example
from pyamg.blackbox import solve

warnings.filterwarnings(action='ignore', module='scipy', message='^internal gelsd')


class TestBlackbox(TestCase):
    def setUp(self):
        self.cases = []

        np.random.seed(2828777142)
        # Poisson problems in 1D and 2D
        N = 20
        self.cases.append((poisson((2*N,), format='csr'), np.random.rand(2*N,)))     # 0
        self.cases.append((poisson((N, N), format='csr'), np.random.rand(N*N,)))  # 1
        # Boxed examples
        A = load_example('recirc_flow')['A'].tocsr()                      # 2
        self.cases.append((A, np.random.rand(A.shape[0],)))
        A = load_example('bar')['A'].tobsr(blocksize=(3, 3))               # 3
        self.cases.append((A, np.random.rand(A.shape[0],)))

    def test_blackbox(self):
        np.random.seed(3105563891)
        for A, b in self.cases:
            x = solve(A, b, verb=False, maxiter=A.shape[0])
            norm1 = np.linalg.norm(b - A*x)
            norm2 = np.linalg.norm(b - A*np.random.rand(b.shape[0],))
            assert(norm1 / norm2 < 1e-4)

        # Special tests
        # (1) Make sure BSR format is preserved, and B is multiple vecs
        A, b = self.cases[-1]
        (x, ml) = solve(A, b, return_solver=True, verb=False)
        assert(ml.levels[0].B.shape[1] == 3)
        assert(ml.levels[0].A.format == 'bsr')

        # (2) Run with solver and make sure that solution is still good
        x = solve(A, b, existing_solver=ml, verb=False)
        norm1 = np.linalg.norm(b - A*x)
        norm2 = np.linalg.norm(b - A*np.random.rand(b.shape[0],))
        assert(norm1 / norm2 < 1e-4)

        # (3) Convert to CSR, make sure B is a single vector
        (x, ml) = solve(A.tocsr(), b, return_solver=True, verb=False)
        assert(ml.levels[0].B.shape[1] == 1)
        assert(ml.levels[0].A.format == 'csr')

        # (4) Run with x0, maxiter and tol
        x = solve(A, b, existing_solver=ml, x0=np.zeros_like(b), tol=1e-8,
                  maxiter=300, verb=False)
        norm1 = np.linalg.norm(b - A*x)
        norm2 = np.linalg.norm(b - A*np.random.rand(b.shape[0],))
        assert(norm1 / norm2 < 1e-7)

        # (5) Run nonsymmetric example, make sure BH isn't None
        A, b = self.cases[2]
        (x, ml) = solve(A, b, return_solver=True, verb=False,
                        maxiter=A.shape[0])
        assert(ml.levels[0].BH is not None)
