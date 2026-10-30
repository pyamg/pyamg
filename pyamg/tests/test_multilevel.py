"""Test MultilevelSolver class."""
import numpy as np
from numpy.testing import TestCase, assert_almost_equal, assert_equal
from scipy import sparse

from pyamg import smoothed_aggregation_solver
from pyamg.gallery import poisson
from pyamg.multilevel import coarse_grid_solver, MultilevelSolver

import pytest


def precon_norm(v, ml):
    """Calculate preconditioner norm of v."""
    v = np.ravel(v)
    w = ml.aspreconditioner() @ v
    return np.sqrt(np.dot(v.conjugate(), w))


class TestMultilevel(TestCase):
    def test_coarse_grid_solver(self):
        cases = []

        cases.append(sparse.csr_array(np.diag(np.arange(1, 5, dtype=float))))
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
                assert_almost_equal(A@x, b)

                # subsequent calls use cached data
                x = s(A, b)
                assert_almost_equal(A@x, b)

    def test_coarse_grid_solver_block(self):
        A = poisson((4, 4), format='csr')
        B = np.arange(3 * A.shape[0], dtype=A.dtype).reshape(A.shape[0], 3)

        # direct solvers support a block right-hand side
        for solver in ['splu', 'pinv', 'lu', 'cholesky']:
            X = coarse_grid_solver(solver)(A, B)
            assert_equal(X.shape, B.shape)
            assert_almost_equal(A @ X, B)

        # Krylov coarse solvers do not support multiple rhs
        with pytest.raises(ValueError, match='multiple right-hand sides'):
            coarse_grid_solver('cg')(A, B)

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
            x, _info = cg(A, b, M=M, rtol=1e-8, maxiter=30, atol=0)
            # cg satisfies convergence in the preconditioner norm
            assert precon_norm(b - A@x, ml) < 1e-8*precon_norm(b, ml)

        for cycle in ['AMLI']:
            M = ml.aspreconditioner(cycle=cycle)
            res = []
            x, _info = fgmres(A, b, tol=1e-8, maxiter=30, M=M, residuals=res)
            # fgmres satisfies convergence in the 2-norm
            assert np.linalg.norm(b - A@x) < 1e-8*np.linalg.norm(b)

    def test_accel(self):
        from pyamg import smoothed_aggregation_solver
        from pyamg.krylov import cg, bicgstab
        np.random.seed(30459128)

        A = poisson((50, 50), format='csr')
        b = np.random.rand(A.shape[0])

        ml = smoothed_aggregation_solver(A)

        # cg halts based on the preconditioner norm
        for accel in ['cg', cg]:
            residuals = []
            x = ml.solve(b, maxiter=30, tol=1e-8, residuals=residuals, accel=accel)
            assert precon_norm(b - A@x, ml) < 1e-8*precon_norm(b, ml)
            assert_almost_equal(precon_norm(b - A@x, ml), residuals[-1])

        # cgs and bicgstab use the Euclidean norm
        for accel in ['bicgstab', 'cgs', bicgstab]:
            residuals = []
            x = ml.solve(b, maxiter=30, tol=1e-8, residuals=residuals, accel=accel)
            assert np.linalg.norm(b - A@x) < 1e-8*np.linalg.norm(b)
            assert_almost_equal(np.linalg.norm(b - A@x), residuals[-1])

    def test_multiple_rhs(self):

        # Poisson with 4 right-hand sides
        s = 4
        A = poisson((50, 50), format='csr')
        b = np.random.rand(A.shape[0], s)

        # Any AMG hierarchy -- rhs independent
        ml = smoothed_aggregation_solver(A,
                                         max_coarse=10,
                                         presmoother='jacobi',
                                         postsmoother='jacobi')

        # initial guess
        residuals = []
        x0 = np.random.rand(A.shape[0], s)

        # block solve
        x = ml.solve(b, x0=x0, maxiter=50, tol=1e-8, residuals=residuals)
        print(residuals)

        # check each
        r = b - A @ x
        for i in range(s):
            assert precon_norm(r[:, i], ml) < 1e-8 * precon_norm(b[:, i], ml)

    def test_solve_shapes(self):
        A = poisson((10, 10), format='csr')
        n = A.shape[0]
        ml = smoothed_aggregation_solver(A, max_coarse=10,
                                         presmoother='jacobi', postsmoother='jacobi')
        np.random.seed(1504262891)
        B = np.random.rand(n, 3)

        # x has the shape of b
        for shape in [(n,), (n, 1), (n, 3)]:
            b = np.ones(shape)
            assert_equal(ml.solve(b, maxiter=2).shape, shape)
            assert_equal(ml.solve(b, x0=np.zeros(shape), maxiter=2).shape, shape)

        # x0 must have the shape of b
        for b, x0 in [(np.ones(n), np.zeros((n, 1))),
                      (np.ones((n, 1)), np.zeros(n)),
                      (B, np.zeros((n, 1)))]:
            with pytest.raises(ValueError, match='x0 and b must have the same shape'):
                ml.solve(b, x0=x0)

        # b must have n rows and at most two dimensions
        for b in [np.ones(n + 1), np.ones((n, 2, 2))]:
            with pytest.raises(ValueError, match='b must have shape'):
                ml.solve(b)

        # non-contiguous and F-ordered input
        assert_equal(ml.solve(B[:, 0], maxiter=2),
                     ml.solve(np.ascontiguousarray(B[:, 0]), maxiter=2))
        assert_equal(ml.solve(np.asfortranarray(B), maxiter=2), ml.solve(B, maxiter=2))

    def test_aspreconditioner_block(self):
        A = poisson((20, 20), format='csr')
        np.random.seed(2125791836)
        B = np.random.rand(A.shape[0], 4)

        # Jacobi supports a block solve
        ml = smoothed_aggregation_solver(A, max_coarse=10,
                                         presmoother='jacobi', postsmoother='jacobi')
        M = ml.aspreconditioner()
        X_cols = np.column_stack([M @ np.ascontiguousarray(B[:, i]) for i in range(4)])
        assert_almost_equal(M @ B, X_cols)
        assert_almost_equal(M @ np.asfortranarray(B), X_cols)

        # Gauss-Seidel does not
        ml = smoothed_aggregation_solver(A, max_coarse=10,
                                         presmoother='gauss_seidel',
                                         postsmoother='gauss_seidel')
        M = ml.aspreconditioner()
        match = 'gauss_seidel does not support multiple right-hand sides'
        with pytest.raises(ValueError, match=match):
            M @ B

    def test_cycle_complexity(self):
        # four levels
        levels = []
        levels.append(MultilevelSolver.Level())
        levels[0].A = sparse.csr_array(np.ones((10, 10)))
        levels[0].P = sparse.csr_array(np.ones((10, 5)))
        levels.append(MultilevelSolver.Level())
        levels[1].A = sparse.csr_array(np.ones((5, 5)))
        levels[1].P = sparse.csr_array(np.ones((5, 3)))
        levels.append(MultilevelSolver.Level())
        levels[2].A = sparse.csr_array(np.ones((3, 3)))
        levels[2].P = sparse.csr_array(np.ones((3, 2)))
        levels.append(MultilevelSolver.Level())
        levels[3].A = sparse.csr_array(np.ones((2, 2)))

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

        cases.append(sparse.csr_array(np.diag(np.arange(1, 5))))
        cases.append(poisson((4,), format='csr'))
        cases.append(poisson((4, 4), format='csr'))

        # Make cases complex
        cases = [G+1e-5j*G for G in cases]
        cases = [0.5*(G + G.T.conjugate()) for G in cases]

        # method should be almost exact for small matrices
        for A in cases:
            for solver in ['splu', 'pinv', 'pinv2', 'lu', 'cholesky', 'cg']:
                s = coarse_grid_solver(solver)

                b = np.arange(A.shape[0], dtype=A.dtype)

                x = s(A, b)
                assert_almost_equal(A@x, b)

                # subsequent calls use cached data
                x = s(A, b)
                assert_almost_equal(A@x, b)
