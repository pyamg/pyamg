"""Test MultilevelSolver class."""
import numpy as np
from numpy.testing import TestCase, assert_almost_equal, assert_equal
from scipy import sparse

from pyamg.gallery import poisson
from pyamg.multilevel import coarse_grid_solver, MultilevelSolver


def precon_norm(v, ml):
    """Calculate preconditioner norm of v."""
    v = np.ravel(v)
    w = ml.aspreconditioner()*v
    return np.sqrt(np.dot(v.conjugate(), w))


class TestMultilevel(TestCase):
    def test_coarse_grid_solver(self):
        cases = []

        cases.append(sparse.csr_matrix(np.diag(np.arange(1, 5, dtype=float))))
        cases.append(poisson((4,), format='csr'))
        cases.append(poisson((4, 4), format='csr'))

        from pyamg.krylov import cg

        def fn(A, b):
            return cg(A, b)[0]

        # method should be almost exact for small matrices
        for A in cases:
            for solver in ['splu', 'pinv', 'pinv2', 'lu', 'cholesky',
                           'cg', fn]:
                s = coarse_grid_solver(solver)

                b = np.arange(A.shape[0], dtype=A.dtype)

                x = s(A, b)
                assert_almost_equal(A*x, b)

                # subsequent calls use cached data
                x = s(A, b)
                assert_almost_equal(A*x, b)

    def test_aspreconditioner(self):
        from pyamg import smoothed_aggregation_solver
        from scipy.sparse.linalg import cg
        from pyamg.krylov import fgmres
        np.random.seed(1331277597)

        A = poisson((50, 50), format='csr')
        b = np.random.rand(A.shape[0])

        ml = smoothed_aggregation_solver(A)

        for cycle in ['V', 'W', 'F']:
            M = ml.aspreconditioner(cycle=cycle)
            x, info = cg(A, b, tol=1e-8, maxiter=30, M=M, atol='legacy')
            # cg satisfies convergence in the preconditioner norm
            assert precon_norm(b - A*x, ml) < 1e-8*precon_norm(b, ml)

        for cycle in ['AMLI']:
            M = ml.aspreconditioner(cycle=cycle)
            res = []
            x, info = fgmres(A, b, tol=1e-8, maxiter=30, M=M, residuals=res)
            # fgmres satisfies convergence in the 2-norm
            assert np.linalg.norm(b - A*x) < 1e-8*np.linalg.norm(b)

    def test_accel(self):
        from pyamg import smoothed_aggregation_solver
        from pyamg.krylov import cg, bicgstab
        np.random.seed(30459128)

        A = poisson((50, 50), format='csr')
        b = np.random.rand(A.shape[0])

        ml = smoothed_aggregation_solver(A)

        # cg halts based on the preconditioner norm
        for accel in ['cg', cg]:
            x = ml.solve(b, maxiter=30, tol=1e-8, accel=accel)
            assert precon_norm(b - A*x, ml) < 1e-8*precon_norm(b, ml)
            residuals = []
            x = ml.solve(b, maxiter=30, tol=1e-8, residuals=residuals,
                         accel=accel)
            assert precon_norm(b - A*x, ml) < 1e-8*precon_norm(b, ml)
            # print residuals
            assert_almost_equal(precon_norm(b - A*x, ml), residuals[-1])

        # cgs and bicgstab use the Euclidean norm
        for accel in ['bicgstab', 'cgs', bicgstab]:
            x = ml.solve(b, maxiter=30, tol=1e-8, accel=accel)
            assert np.linalg.norm(b - A*x) < 1e-8*np.linalg.norm(b)
            residuals = []
            x = ml.solve(b, maxiter=30, tol=1e-8, residuals=residuals,
                         accel=accel)
            assert np.linalg.norm(b - A*x) < 1e-8*np.linalg.norm(b)
            # print residuals
            assert_almost_equal(np.linalg.norm(b - A*x), residuals[-1])

    def test_cycle_complexity(self):
        # four levels
        levels = []
        levels.append(MultilevelSolver.Level())
        levels[0].A = sparse.csr_matrix(np.ones((10, 10)))
        levels[0].P = sparse.csr_matrix(np.ones((10, 5)))
        levels.append(MultilevelSolver.Level())
        levels[1].A = sparse.csr_matrix(np.ones((5, 5)))
        levels[1].P = sparse.csr_matrix(np.ones((5, 3)))
        levels.append(MultilevelSolver.Level())
        levels[2].A = sparse.csr_matrix(np.ones((3, 3)))
        levels[2].P = sparse.csr_matrix(np.ones((3, 2)))
        levels.append(MultilevelSolver.Level())
        levels[3].A = sparse.csr_matrix(np.ones((2, 2)))

        # one level hierarchy
        mg = MultilevelSolver(levels[:1])
        assert_equal(mg.cycle_complexity(cycle='V'), 100.0/100.0)  # 1
        assert_equal(mg.cycle_complexity(cycle='W'), 100.0/100.0)  # 1
        assert_equal(mg.cycle_complexity(cycle='AMLI'), 100.0/100.0)  # 1
        assert_equal(mg.cycle_complexity(cycle='F'), 100.0/100.0)  # 1

        # two level hierarchy
        mg = MultilevelSolver(levels[:2])
        assert_equal(mg.cycle_complexity(cycle='V'), 225.0/100.0)  # 2,1
        assert_equal(mg.cycle_complexity(cycle='W'), 225.0/100.0)  # 2,1
        assert_equal(mg.cycle_complexity(cycle='AMLI'), 225.0/100.0)  # 2,1
        assert_equal(mg.cycle_complexity(cycle='F'), 225.0/100.0)  # 2,1

        # three level hierarchy
        mg = MultilevelSolver(levels[:3])
        assert_equal(mg.cycle_complexity(cycle='V'), 259.0/100.0)  # 2,2,1
        assert_equal(mg.cycle_complexity(cycle='W'), 318.0/100.0)  # 2,4,2
        assert_equal(mg.cycle_complexity(cycle='AMLI'), 318.0/100.0)  # 2,4,2
        assert_equal(mg.cycle_complexity(cycle='F'), 318.0/100.0)  # 2,4,2

        # four level hierarchy
        mg = MultilevelSolver(levels[:4])
        assert_equal(mg.cycle_complexity(cycle='V'), 272.0/100.0)  # 2,2,2,1
        assert_equal(mg.cycle_complexity(cycle='W'), 388.0/100.0)  # 2,4,8,4
        assert_equal(mg.cycle_complexity(cycle='AMLI'), 388.0/100.0)  # 2,4,8,4
        assert_equal(mg.cycle_complexity(cycle='F'), 366.0/100.0)  # 2,4,6,3


class TestComplexMultilevel(TestCase):
    def test_coarse_grid_solver(self):
        cases = []

        cases.append(sparse.csr_matrix(np.diag(np.arange(1, 5))))
        cases.append(poisson((4,), format='csr'))
        cases.append(poisson((4, 4), format='csr'))

        # Make cases complex
        cases = [G+1e-5j*G for G in cases]
        cases = [0.5*(G + G.H) for G in cases]

        # method should be almost exact for small matrices
        for A in cases:
            for solver in ['splu', 'pinv', 'pinv2', 'lu', 'cholesky', 'cg']:
                s = coarse_grid_solver(solver)

                b = np.arange(A.shape[0], dtype=A.dtype)

                x = s(A, b)
                assert_almost_equal(A*x, b)

                # subsequent calls use cached data
                x = s(A, b)
                assert_almost_equal(A*x, b)
