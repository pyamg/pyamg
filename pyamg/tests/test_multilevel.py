"""Test MultilevelSolver class."""
import numpy as np
import pytest
from numpy.testing import TestCase, assert_allclose, assert_almost_equal, assert_equal
from scipy import sparse

from pyamg import smoothed_aggregation_solver
from pyamg.aggregation.pairwise import pairwise_solver
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


@pytest.fixture(params=[(pairwise_solver, np.float64),
                       (pairwise_solver, np.float32),
                       (smoothed_aggregation_solver, np.complex128)],
                ids=['pairwise-float64', 'pairwise-float32', 'smoothed-complex128'])
def amli_solver(request):
    solver, dtype = request.param
    A = poisson((100,), dtype=dtype, format='csr')
    smoother = ('gauss_seidel', {'sweep': 'symmetric'})
    return A, solver(A, presmoother=smoother, postsmoother=smoother)


@pytest.mark.parametrize('scale', [1e-8, 1e8])
def test_amli_rhs_scaling(amli_solver, scale):
    A, ml = amli_solver
    rng = np.random.default_rng(362)
    b = rng.standard_normal(A.shape[0]).astype(A.dtype)
    if np.iscomplexobj(b):
        b += 1j * rng.standard_normal(A.shape[0])

    M = ml.aspreconditioner(cycle='AMLI')
    correction = M @ b
    tol = max(1e-10, 1000 * np.finfo(A.dtype).eps)
    assert np.linalg.norm((M @ (scale * b)) / scale - correction) < (
        tol * np.linalg.norm(correction))


def test_amli_convergence(amli_solver):
    A, ml = amli_solver
    rng = np.random.default_rng(0)
    x0 = rng.random(A.shape[0]).astype(A.dtype)
    exact = rng.random(A.shape[0]).astype(A.dtype)
    if np.iscomplexobj(exact):
        x0 += 1j * rng.random(A.shape[0])
        exact += 1j * rng.random(A.shape[0])
    b = A @ exact

    tol = max(1e-10, 100 * np.finfo(A.dtype).eps)
    x = ml.solve(b, x0=x0, maxiter=200, tol=tol, cycle='AMLI')
    assert np.linalg.norm(b - A @ x) < tol * np.linalg.norm(b)


def test_amli_zero_coarse_residual(amli_solver):
    A, ml = amli_solver
    exact = np.zeros(A.shape[0], dtype=A.dtype)
    exact[0] = 1
    b = A @ exact

    # The fine-grid smoother solves this right-hand side exactly.
    with np.errstate(divide='raise', invalid='raise'):
        x = ml.solve(b, maxiter=1, cycle='AMLI')
    assert_allclose(x, exact, rtol=0, atol=0)
