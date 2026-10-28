"""Test that solvers with different sparse index types."""
import numpy as np
from numpy.testing import TestCase, assert_array_almost_equal
from scipy import sparse

from pyamg.gallery import poisson, linear_elasticity, linear_elasticity_p1, \
    load_example, regular_triangle_mesh
from pyamg.aggregation import smoothed_aggregation_solver, rootnode_solver, \
    pairwise_solver, adaptive_sa_solver
from pyamg.classical import ruge_stuben_solver, air_solver
from pyamg.relaxation import relaxation
from pyamg.blackbox import solve


class TestSmoothedAggregation(TestCase):
    def setUp(self):
        self.cases = []

        self.cases.append((poisson((10, 10), format='csr'), None))
        self.cases.append(linear_elasticity((7, 7), format='bsr'))

        for A, _ in self.cases:
            A.indptr = A.indptr.astype(np.int64)
            A.indices = A.indices.astype(np.int64)

    def run_cases(self, opts):
        for A, B in self.cases:
            ml = smoothed_aggregation_solver(A, B, max_coarse=5, **opts)

            np.random.seed(1883275855)  # make tests repeatable

            x = np.random.rand(A.shape[0])
            b = A @ np.random.rand(A.shape[0])

            residuals = []
            ml.solve(b, x0=x, maxiter=30, tol=1e-10, residuals=residuals)
            cf = (residuals[-1] / residuals[0])**(1.0 / len(residuals))
            assert cf < 0.9

    def test_strength_of_connection(self):
        for strength in ['symmetric', 'classical', 'evolution',
                         'algebraic_distance', 'affinity']:
            self.run_cases({'strength': strength})

    def test_aggregation_method(self):
        for aggregate in ['standard', 'naive', 'lloyd', 'balanced lloyd']:
            self.run_cases({'aggregate': aggregate})

    def test_prolongation_smoother(self):
        for smooth in ['jacobi', 'richardson', 'energy']:
            self.run_cases({'smooth': smooth})

    def test_complex(self):
        A = poisson((10, 10), format='csr')
        A = A + 1.0j * sparse.eye_array(A.shape[0], A.shape[1], format='csr')
        A.indptr = A.indptr.astype(np.int64)
        A.indices = A.indices.astype(np.int64)

        for strength in ['symmetric', 'classical', 'evolution']:
            ml = smoothed_aggregation_solver(A, strength=strength, max_coarse=5)

            np.random.seed(776825606)  # make tests repeatable

            x = np.random.rand(A.shape[0]) + 1.0j * np.random.rand(A.shape[0])
            b = A @ np.random.rand(A.shape[0])

            residuals = []
            ml.solve(b, x0=x, maxiter=30, tol=1e-10, residuals=residuals)
            cf = (residuals[-1] / residuals[0])**(1.0 / len(residuals))
            assert cf < 0.9


class TestSolvers(TestCase):
    def setUp(self):
        self.A = poisson((10, 10), format='csr')
        self.A.indptr = self.A.indptr.astype(np.int64)
        self.A.indices = self.A.indices.astype(np.int64)

        self.Ab, self.Bb = linear_elasticity((7, 7), format='bsr')
        self.Ab.indptr = self.Ab.indptr.astype(np.int64)
        self.Ab.indices = self.Ab.indices.astype(np.int64)

        self.Ans = load_example('recirc_flow')['A'].tocsr()
        self.Ans.indptr = self.Ans.indptr.astype(np.int64)
        self.Ans.indices = self.Ans.indices.astype(np.int64)

        np.random.seed(1883275855)

    def test_rootnode(self):
        for A, B in [(self.A, None), (self.Ab, self.Bb)]:
            ml = rootnode_solver(A, B, max_coarse=5)
            x = np.random.rand(A.shape[0])
            b = A @ np.random.rand(A.shape[0])
            residuals = []
            ml.solve(b, x0=x, maxiter=30, tol=1e-10, residuals=residuals)
            cf = (residuals[-1] / residuals[0])**(1.0 / len(residuals))
            assert cf < 0.9

    def test_pairwise(self):
        ml = pairwise_solver(self.A, max_coarse=5)
        x = np.random.rand(self.A.shape[0])
        b = self.A @ np.random.rand(self.A.shape[0])
        residuals = []
        ml.solve(b, x0=x, maxiter=30, tol=1e-10, residuals=residuals)
        cf = (residuals[-1] / residuals[0])**(1.0 / len(residuals))
        assert cf < 0.9

    def test_adaptive_sa(self):
        ml, _ = adaptive_sa_solver(self.A, num_candidates=1, max_coarse=5)
        x = np.random.rand(self.A.shape[0])
        b = self.A @ np.random.rand(self.A.shape[0])
        residuals = []
        ml.solve(b, x0=x, maxiter=30, tol=1e-10, residuals=residuals)
        cf = (residuals[-1] / residuals[0])**(1.0 / len(residuals))
        assert cf < 0.9

    def test_ruge_stuben(self):
        for CF in ['RS', 'PMIS', 'PMISc', 'CLJP', 'CLJPc', 'CR']:
            for interpolation in ['classical', 'direct']:
                ml = ruge_stuben_solver(self.A, CF=CF, interpolation=interpolation,
                                        max_coarse=5)
                x = np.random.rand(self.A.shape[0])
                b = self.A @ np.random.rand(self.A.shape[0])
                residuals = []
                ml.solve(b, x0=x, maxiter=30, tol=1e-10, residuals=residuals)
                cf = (residuals[-1] / residuals[0])**(1.0 / len(residuals))
                assert cf < 0.9

    def test_air(self):
        ml = air_solver(self.Ans, max_coarse=5)
        x = np.random.rand(self.Ans.shape[0])
        b = self.Ans @ np.random.rand(self.Ans.shape[0])
        residuals = []
        ml.solve(b, x0=x, maxiter=30, tol=1e-10, residuals=residuals)
        cf = (residuals[-1] / residuals[0])**(1.0 / len(residuals))
        assert cf < 0.9

    def test_blackbox(self):
        for A in [self.A, self.Ab]:
            b = np.random.rand(A.shape[0])
            x = solve(A, b, verb=False, tol=1e-8)
            assert np.linalg.norm(b - A @ x) / np.linalg.norm(b) < 1e-6


class TestRelaxation(TestCase):
    def setUp(self):
        # each case is the same matrix with int32 and int64 indices
        self.cases = []

        A = poisson((10, 10), format='csr')
        A64 = A.copy()
        A64.indptr = A64.indptr.astype(np.int64)
        A64.indices = A64.indices.astype(np.int64)
        self.cases.append((A, A64))

        A, _ = linear_elasticity((7, 7), format='bsr')
        A64 = A.copy()
        A64.indptr = A64.indptr.astype(np.int64)
        A64.indices = A64.indices.astype(np.int64)
        self.cases.append((A, A64))

        np.random.seed(0)

    def test_point_relaxation(self):
        for A, A64 in self.cases:
            x0 = np.random.rand(A.shape[0])
            b = np.random.rand(A.shape[0])

            x, x64 = x0.copy(), x0.copy()
            relaxation.gauss_seidel(A, x, b)
            relaxation.gauss_seidel(A64, x64, b)
            assert_array_almost_equal(x64, x)

            x, x64 = x0.copy(), x0.copy()
            relaxation.jacobi(A, x, b, omega=0.6)
            relaxation.jacobi(A64, x64, b, omega=0.6)
            assert_array_almost_equal(x64, x)

            x, x64 = x0.copy(), x0.copy()
            relaxation.block_jacobi(A, x, b, blocksize=2)
            relaxation.block_jacobi(A64, x64, b, blocksize=2)
            assert_array_almost_equal(x64, x)

            x, x64 = x0.copy(), x0.copy()
            relaxation.block_gauss_seidel(A, x, b, blocksize=2)
            relaxation.block_gauss_seidel(A64, x64, b, blocksize=2)
            assert_array_almost_equal(x64, x)

    def test_csr_relaxation(self):
        A, A64 = self.cases[0]
        n = A.shape[0]
        x0 = np.random.rand(n)
        b = np.random.rand(n)

        x, x64 = x0.copy(), x0.copy()
        relaxation.sor(A, x, b, omega=1.2)
        relaxation.sor(A64, x64, b, omega=1.2)
        assert_array_almost_equal(x64, x)

        x, x64 = x0.copy(), x0.copy()
        relaxation.schwarz(A, x, b)
        relaxation.schwarz(A64, x64, b)
        assert_array_almost_equal(x64, x)

        x, x64 = x0.copy(), x0.copy()
        relaxation.polynomial(A, x, b, [-0.1, 0.5])
        relaxation.polynomial(A64, x64, b, [-0.1, 0.5])
        assert_array_almost_equal(x64, x)

        x, x64 = x0.copy(), x0.copy()
        relaxation.jacobi_ne(A, x, b, omega=0.1)
        relaxation.jacobi_ne(A64, x64, b, omega=0.1)
        assert_array_almost_equal(x64, x)

        x, x64 = x0.copy(), x0.copy()
        relaxation.gauss_seidel_ne(A, x, b)
        relaxation.gauss_seidel_ne(A64, x64, b)
        assert_array_almost_equal(x64, x)

        x, x64 = x0.copy(), x0.copy()
        relaxation.gauss_seidel_nr(A, x, b)
        relaxation.gauss_seidel_nr(A64, x64, b)
        assert_array_almost_equal(x64, x)

        # indices default to int64 from arange
        indices = np.arange(0, n, 2)

        x, x64 = x0.copy(), x0.copy()
        relaxation.gauss_seidel_indexed(A, x, b, indices)
        relaxation.gauss_seidel_indexed(A64, x64, b, indices)
        assert_array_almost_equal(x64, x)

        x, x64 = x0.copy(), x0.copy()
        relaxation.jacobi_indexed(A, x, b, indices)
        relaxation.jacobi_indexed(A64, x64, b, indices)
        assert_array_almost_equal(x64, x)

        Cpts = np.arange(0, n, 3)
        Fpts = np.setdiff1d(np.arange(n), Cpts)

        x, x64 = x0.copy(), x0.copy()
        relaxation.cf_jacobi(A, x, b, Cpts, Fpts)
        relaxation.cf_jacobi(A64, x64, b, Cpts, Fpts)
        assert_array_almost_equal(x64, x)

        x, x64 = x0.copy(), x0.copy()
        relaxation.fc_jacobi(A, x, b, Cpts, Fpts)
        relaxation.fc_jacobi(A64, x64, b, Cpts, Fpts)
        assert_array_almost_equal(x64, x)


class TestGallery(TestCase):
    def test_coo_assembly(self):
        # coo -> csr from numpy's default int64 row/col gives int64 indices
        A = poisson((10, 10), format='coo')
        A = sparse.coo_array((A.data, (A.row.astype(np.int64), A.col.astype(np.int64))),
                             shape=A.shape).tocsr()
        ml = smoothed_aggregation_solver(A, max_coarse=5)

        np.random.seed(0)
        b = np.random.rand(A.shape[0])
        residuals = []
        ml.solve(b, tol=1e-8, residuals=residuals)
        assert residuals[-1] / residuals[0] < 1e-8

    def test_elasticity_p1(self):
        # int64 elements give int64 indices
        V, E = regular_triangle_mesh(12, 12)
        A, B = linear_elasticity_p1(V, E.astype(np.int64), format='bsr')

        # no boundary conditions: shift to make A nonsingular
        A = (A + sparse.eye_array(A.shape[0], format='bsr')).tobsr(blocksize=(2, 2))
        ml = smoothed_aggregation_solver(A, B, max_coarse=5)

        np.random.seed(0)
        b = np.random.rand(A.shape[0])
        residuals = []
        ml.solve(b, tol=1e-8, residuals=residuals, maxiter=200)
        assert residuals[-1] / residuals[0] < 1e-8
