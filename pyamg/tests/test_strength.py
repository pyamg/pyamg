"""Test strength of connection."""
import numpy as np
from numpy.testing import TestCase, assert_equal, assert_array_almost_equal,\
    assert_array_equal, assert_allclose
from scipy import sparse
import scipy.linalg as sla

from pyamg.gallery import poisson, linear_elasticity, load_example,\
    stencil_grid
from pyamg.strength import classical_strength_of_connection,\
    symmetric_strength_of_connection, evolution_strength_of_connection,\
    distance_strength_of_connection
from pyamg.amg_core import incomplete_mat_mult_csr
from pyamg.util.linalg import approximate_spectral_radius
from pyamg.util.utils import scale_rows
from pyamg.util.params import set_tol

classical_soc = classical_strength_of_connection
symmetric_soc = symmetric_strength_of_connection
evolution_soc = evolution_strength_of_connection
distance_soc = distance_strength_of_connection


class TestStrengthOfConnection(TestCase):
    def setUp(self):
        self.cases = []

        # random matrices
        np.random.seed(222352579)
        for N in [2, 3, 5]:
            self.cases.append(sparse.csr_matrix(np.random.rand(N, N)))

        # Poisson problems in 1D and 2D
        for N in [2, 3, 5, 7, 10, 11, 19]:
            self.cases.append(poisson((N,), format='csr'))
        for N in [2, 3, 7, 9]:
            self.cases.append(poisson((N, N), format='csr'))

        for name in ['knot', 'airfoil', 'bar']:
            ex = load_example(name)
            self.cases.append(ex['A'].tocsr())

    def test_classical_strength_of_connection(self):

        for A in self.cases:
            for theta in [0.0, 0.05, 0.25, 0.50, 0.90]:
                result = classical_soc(A, theta)
                expected = reference_classical_soc(A, theta)

                assert_equal(result.nnz, expected.nnz)
                assert_array_almost_equal(result.toarray(), expected.toarray())

        # Test BSR capabilities
        import warnings
        warnings.filterwarnings(action='ignore', message='Implicit conversion*')
        CSRtest = sparse.csr_matrix(np.array([[4.0,  -1.0, -1.1,  1.0,  0.0, 0.0],
                                              [-0.9, -1.1, -1.0,  0.0,  9.5, 0.0],
                                              [-1.9,  4.1,  5.0, -4.0,  0.0, 0.0],
                                              [0.0,  -0.1, -1.0,  0.0,  0.0, 5.0],
                                              [0.0,   0.0,  1.0,  0.0, -9.5, 1.0],
                                              [0.0,   0.0,  0.0, -1.0, -1.0, 1.0]]))
        BSRtest = CSRtest.tobsr(blocksize=(2, 2))

        # Check that entry equals Frobenius norm of that block,
        # scaled by largest entry in the row
        result = classical_soc(BSRtest, 0.1, block=True, norm='fro').toarray()
        assert_allclose(result[0, 0],
                        np.sum(np.abs(BSRtest.data[0]*BSRtest.data[0]).flatten()) / 9.5**2)

        # Check that theta filters an entry
        result = classical_soc(BSRtest, 0.1, block=True, norm='fro')
        assert_equal(result.nnz, 7)
        result = classical_soc(BSRtest, 0.9, block=True, norm='fro')
        assert_equal(result.nnz, 6)

        # Check that abs chooses largest magnitue entry in a block
        # scaled by largest entry in the row
        result = classical_soc(BSRtest, 0.1, block=True, norm='abs').toarray()
        assert_allclose(result[0, 0], 4 / 9.5)
        assert_allclose(result[0, 2], 1.0)
        # Do same check for min choosing smallest entry,
        result = classical_soc(BSRtest, 0.1, block=True, norm='min').toarray()
        assert_allclose(result[2, 0], 0.0)
        assert_allclose(result[2, 1], 1./9.5)
        assert_allclose(result[1, 0], 1.9/4)

        # In such a setting, abs and fro should be the same
        result1 = classical_soc(BSRtest, 0.1, block=False, norm='abs').toarray()
        result2 = classical_soc(BSRtest, 0.1, block=False, norm='fro').toarray()
        assert_array_almost_equal(result1, result2)

        result = classical_soc(BSRtest, 0.1, block=False, norm='min').toarray()
        assert_allclose(result[0, 0], 1.0)
        assert_allclose(result[0, 1], 1.0)
        assert_allclose(result[0, 2], 0.0)

        # Result is the same with block=True|False for a CSR matrix
        result1 = classical_soc(CSRtest, 0.1, block=True, norm='min').toarray()
        result2 = classical_soc(CSRtest, 0.1, block=False, norm='min').toarray()
        assert_array_almost_equal(result1, result2)

    def test_classical_strength_of_connection_min(self):
        for A in self.cases:
            for theta in [0.0, 0.05, 0.25, 0.50, 0.90]:
                result = classical_soc(A, theta, norm='min')
                expected = reference_classical_soc(A, theta, norm='min')

                assert_equal(result.nnz, expected.nnz)
                assert_array_almost_equal(result.toarray(), expected.toarray())

    def test_symmetric_strength_of_connection(self):
        for A in self.cases:
            for theta in [0.0, 0.1, 0.5, 1.0, 10.0]:
                result = symmetric_soc(A, theta)
                expected = reference_symmetric_soc(A, theta)

                assert_equal(result.nnz, expected.nnz)
                assert_array_almost_equal(result.toarray(), expected.toarray())

    def test_distance_strength_of_connection(self):
        data = load_example('airfoil')
        cases = []
        cases.append((data['A'].tocsr(), data['vertices']))

        for (A, V) in cases:
            for theta in [1.5, 2.0, 2.5]:
                result = distance_soc(A, V, theta=theta)
                expected = reference_distance_soc(A, V, theta=theta)
                assert_equal(result.nnz, expected.nnz)
                assert_array_almost_equal(result.toarray(), expected.toarray())

        for (A, V) in cases:
            for theta in [0.5, 1.0, 1.5]:
                result = distance_soc(A, V, theta=theta, relative_drop=False)
                expected = reference_distance_soc(A, V, theta=theta,
                                                  relative_drop=False)
                assert_equal(result.nnz, expected.nnz)
                assert_array_almost_equal(result.toarray(), expected.toarray())

    def test_incomplete_mat_mult_csr(self):
        # Test a critical helper routine for evolution_soc(...)
        # We test that (A*B).multiply(mask) = incomplete_mat_mult_csr(A,B,mask)
        #
        # This function assumes square matrices, A, B and mask, so that
        # simplifies the testing
        cases = []

        # 1x1 tests
        A = sparse.csr_matrix(np.array([[1.1]]))
        B = sparse.csr_matrix(np.array([[1.0]]))
        A2 = sparse.csr_matrix(np.array([[0.]]))
        mask = sparse.csr_matrix(np.array([[1.]]))
        cases.append((A, A, mask))
        cases.append((A, B, mask))
        cases.append((A, A2, mask))
        cases.append((A2, A2, mask))

        # 2x2 tests
        A = sparse.csr_matrix(np.array([[1., 2.], [2., 4.]]))
        B = sparse.csr_matrix(np.array([[1.3, 2.], [2.8, 4.]]))
        A2 = sparse.csr_matrix(np.array([[1.3, 0.], [0., 4.]]))
        B2 = sparse.csr_matrix(np.array([[1.3, 0.], [2., 4.]]))
        mask = sparse.csr_matrix((np.ones(4), (np.array([0, 0, 1, 1]),
                                               np.array([0, 1, 0, 1]))), shape=(2, 2))
        cases.append((A, A, mask))
        cases.append((A, B, mask))
        cases.append((A2, A2, mask))
        cases.append((A2, B2, mask))

        mask = sparse.csr_matrix((np.ones(3), (np.array([0, 0, 1]),
                                               np.array([0, 1, 1]))), shape=(2, 2))
        cases.append((A, A, mask))
        cases.append((A, B, mask))
        cases.append((A2, A2, mask))
        cases.append((A2, B2, mask))

        mask = sparse.csr_matrix((np.ones(2), (np.array([0, 1]),
                                               np.array([0, 0]))), shape=(2, 2))
        cases.append((A, A, mask))
        cases.append((A, B, mask))
        cases.append((A2, A2, mask))
        cases.append((A2, B2, mask))

        # 5x5 tests
        A = np.array([[0., 16.9, 6.4, 0.0, 5.8],
                      [16.9, 13.8, 7.2, 0., 9.5],
                      [6.4, 7.2, 12., 6.1, 5.9],
                      [0.0, 0., 6.1, 0., 0.],
                      [5.8, 9.5, 5.9, 0., 13.]])
        C = A.copy()
        C[1, 0] = 3.1
        C[3, 2] = 10.1
        A2 = A.copy()
        A2[1, :] = 0.0
        A3 = A2.copy()
        A3[:, 1] = 0.0
        A = sparse.csr_matrix(A)
        A2 = sparse.csr_matrix(A2)
        A3 = sparse.csr_matrix(A3)
        C = sparse.csr_matrix(C)

        mask = A.copy()
        mask.data[:] = 1.0
        cases.append((A, A, mask))
        cases.append((C, C, mask))
        cases.append((A2, A2, mask))
        cases.append((A3, A3, mask))
        cases.append((A, A2, mask))
        cases.append((A3, A, mask))
        cases.append((A, C, mask))
        cases.append((C, A, mask))

        mask.data[1] = 0.0
        mask.data[5] = 0.0
        mask.data[9] = 0.0
        mask.data[13] = 0.0
        mask.eliminate_zeros()
        cases.append((A, A, mask))
        cases.append((C, C, mask))
        cases.append((A2, A2, mask))
        cases.append((A3, A3, mask))
        cases.append((A, A2, mask))
        cases.append((A3, A, mask))
        cases.append((A, C, mask))
        cases.append((C, A, mask))

        # Laplacian tests
        A = poisson((5, 5), format='csr')
        B = A.copy()
        B.data[1] = 3.5
        B.data[11] = 11.6
        B.data[28] = -3.2
        C = sparse.csr_matrix(np.zeros(A.shape))
        mask = A.copy()
        mask.data[:] = 1.0
        cases.append((A, A, mask))
        cases.append((A, B, mask))
        cases.append((B, A, mask))
        cases.append((C, A, mask))
        cases.append((A, C, mask))
        cases.append((C, C, mask))

        # Imaginary tests
        A = np.array([[0.0 + 0.j, 0.0 + 16.9j, 6.4 + 1.2j, 0.0 + 0.j, 0.0 + 0.j],
                      [16.9 + 0.j, 13.8 + 0.j, 7.2 + 0.j, 0.0 + 0.j, 0.0 + 9.5j],
                      [0.0 + 6.4j, 7.2 - 8.1j, 12.0 + 0.j, 6.1 + 0.j, 5.9 + 0.j],
                      [0.0 + 0.j, 0.0 + 0.j, 6.1 + 0.j, 0.0 + 0.j, 0.0 + 0.j],
                      [5.8 + 0.j, -4.0 + 9.5j, -3.2 - 5.9j, 0.0 + 0.j, 13.0 + 0.j]])
        C = A.copy()
        C[1, 0] = 3.1j - 1.3
        C[3, 2] = -10.1j + 9.7
        A = sparse.csr_matrix(A)
        C = sparse.csr_matrix(C)

        mask = A.copy()
        mask.data[:] = 1.0
        cases.append((A, A, mask))
        cases.append((C, C, mask))
        cases.append((A, C, mask))
        cases.append((C, A, mask))

        for case in cases:
            A = case[0].tocsr()
            B = case[1].tocsc()
            mask = case[2].tocsr()
            A.sort_indices()
            B.sort_indices()
            mask.sort_indices()
            result = mask.copy()
            incomplete_mat_mult_csr(A.indptr, A.indices, A.data, B.indptr,
                                    B.indices, B.data, result.indptr,
                                    result.indices, result.data, A.shape[0])
            result.eliminate_zeros()
            exact = (A*B).multiply(mask)
            exact.sort_indices()
            exact.eliminate_zeros()
            assert_array_almost_equal(exact.data, result.data)
            assert_array_equal(exact.indptr, result.indptr)
            assert_array_equal(exact.indices, result.indices)

    def test_evolution_strength_of_connection(self):
        # Params:  A, B, epsilon=4.0, k=2, proj_type="l2"
        cases = []

        # Ensure that isotropic diffusion results in isotropic strength stencil
        for N in [3, 5, 7, 10]:
            A = poisson((N,), format='csr')
            B = np.ones((A.shape[0], 1))
            cases.append({'A': A.copy(), 'B': B.copy(), 'epsilon': 4.0,
                          'k': 2, 'proj': 'l2'})

        # Ensure that anisotropic diffusion results in an anisotropic
        # strength stencil
        for N in [3, 6, 7]:
            u = np.ones(N*N)
            A = sparse.spdiags([-u, -0.001*u, 2.002*u, -0.001*u, -u],
                               [-N, -1, 0, 1, N], N*N, N*N, format='csr')
            B = np.ones((A.shape[0], 1))
            cases.append({'A': A.copy(), 'B': B.copy(), 'epsilon': 4.0,
                          'k': 2, 'proj': 'l2'})

        # Ensure that isotropic elasticity results in an isotropic stencil
        for N in [3, 6, 7]:
            (A, B) = linear_elasticity((N, N), format='bsr')
            cases.append({'A': A.copy(), 'B': B.copy(), 'epsilon': 32.0,
                          'k': 8, 'proj': 'D_A'})

        # Run an example with a non-uniform stencil
        ex = load_example('airfoil')
        A = ex['A'].tocsr()
        B = np.ones((A.shape[0], 1))
        cases.append({'A': A.copy(), 'B': B.copy(), 'epsilon': 8.0, 'k': 4,
                      'proj': 'D_A'})
        Absr = A.tobsr(blocksize=(5, 5))
        cases.append({'A': Absr.copy(), 'B': B.copy(), 'epsilon': 8.0, 'k': 4,
                      'proj': 'D_A'})
        # Different B
        B = np.arange(1, 2*A.shape[0]+1, dtype=float).reshape(-1, 2)
        cases.append({'A': A.copy(), 'B': B.copy(), 'epsilon': 4.0, 'k': 2,
                      'proj': 'l2'})
        cases.append({'A': Absr.copy(), 'B': B.copy(), 'epsilon': 4.0, 'k': 2,
                      'proj': 'l2'})

        # Zero row and column
        A.data[A.indptr[4]:A.indptr[5]] = 0.0
        A = A.tocsc()
        A.data[A.indptr[4]:A.indptr[5]] = 0.0
        A.eliminate_zeros()
        A = A.tocsr()
        A.sort_indices()
        cases.append({'A': A.copy(), 'B': B.copy(), 'epsilon': 4.0, 'k': 2,
                      'proj': 'l2'})
        Absr = A.tobsr(blocksize=(5, 5))
        cases.append({'A': Absr.copy(), 'B': B.copy(), 'epsilon': 4.0, 'k': 2,
                      'proj': 'l2'})

        for ca in cases:
            np.random.seed(2001321804)  # make results deterministic
            result = evolution_soc(ca['A'], ca['B'], epsilon=ca['epsilon'],
                                   k=ca['k'], proj_type=ca['proj'],
                                   symmetrize_measure=False)
            np.random.seed(2001321804)  # make results deterministic
            expected = reference_evolution_soc(ca['A'], ca['B'],
                                               epsilon=ca['epsilon'],
                                               k=ca['k'], proj_type=ca['proj'])
            assert_array_almost_equal(result.toarray(), expected.toarray(),
                                      decimal=4)

        # Test Scale Invariance for multiple near nullspace candidates
        (A, B) = linear_elasticity((5, 5), format='bsr')
        np.random.seed(4055795935)  # make results deterministic
        result_unscaled = evolution_soc(A, B, epsilon=4.0,
                                        k=2, proj_type='D_A',
                                        symmetrize_measure=False)
        # create scaled A
        D = sparse.spdiags([np.arange(A.shape[0], 2*A.shape[0], dtype=float)],
                           [0], A.shape[0], A.shape[0], format='csr')
        Dinv = sparse.spdiags([1.0/np.arange(A.shape[0], 2*A.shape[0], dtype=float)],
                              [0], A.shape[0], A.shape[0], format='csr')
        np.random.seed(3969802542)  # make results deterministic
        result_scaled = evolution_soc((D*A*D).tobsr(blocksize=(2, 2)),
                                      Dinv*B, epsilon=4.0, k=2,
                                      proj_type='D_A',
                                      symmetrize_measure=False)
        assert_array_almost_equal(result_scaled.toarray(),
                                  result_unscaled.toarray(), decimal=2)


# Define Complex tests
class TestComplexStrengthOfConnection(TestCase):
    def setUp(self):
        self.cases = []

        # random matrices
        np.random.seed(954619597)
        for N in [2, 3, 5]:
            self.cases.append(sparse.csr_matrix(np.random.rand(N, N))
                              + sparse.csr_matrix(1.0j*np.random.rand(N, N)))

        # Poisson problems in 1D and 2D
        for N in [2, 3, 5, 7, 10, 11, 19]:
            A = poisson((N,), format='csr')
            A.data = A.data + 1.0j*A.data
            self.cases.append(A)
        for N in [2, 3, 7, 9]:
            A = poisson((N, N), format='csr')
            A.data = A.data + 1.0j*np.random.rand(A.data.shape[0],)
            self.cases.append(A)

        for name in ['knot', 'airfoil', 'bar']:
            ex = load_example(name)
            A = ex['A'].tocsr()
            A.data = A.data + 0.5j*np.random.rand(A.data.shape[0],)
            self.cases.append(A)

    def test_classical_strength_of_connection(self):
        for A in self.cases:
            for theta in [0.0, 0.05, 0.25, 0.50, 0.90]:
                result = classical_soc(A, theta)
                expected = reference_classical_soc(A, theta)

                assert_equal(result.nnz, expected.nnz)
                assert_array_almost_equal(result.toarray(), expected.toarray())

    def test_symmetric_strength_of_connection(self):
        for A in self.cases:
            for theta in [0.0, 0.1, 0.5, 1.0, 10.0]:
                expected = reference_symmetric_soc(A, theta)
                result = symmetric_soc(A, theta)

                assert_equal(result.nnz, expected.nnz)
                assert_array_almost_equal(result.toarray(), expected.toarray())

    def test_evolution_strength_of_connection(self):
        cases = []

        # Single near nullspace candidate
        stencil = [[0.0, -1.0, 0.0], [-0.001, 2.002, -0.001], [0.0, -1.0, 0.0]]
        A = 1.0j*stencil_grid(stencil, (4, 4), format='csr')
        B = 1.0j*np.ones((A.shape[0], 1))
        B[0] = 1.2 - 12.0j
        B[11] = -14.2
        cases.append({'A': A.copy(), 'B': B.copy(), 'epsilon': 4.0, 'k': 2,
                      'proj': 'l2'})

        # Multiple near nullspace candidate
        B = 1.0j*np.ones((A.shape[0], 2))
        B[0:-1:2, 0] = 0.0
        B[1:-1:2, 1] = 0.0
        B[-1, 0] = 0.0
        B[11, 1] = -14.2
        B[0, 0] = 1.2 - 12.0j
        cases.append({'A': A.copy(), 'B': B.copy(), 'epsilon': 4.0, 'k': 2,
                      'proj': 'l2'})
        Absr = A.tobsr(blocksize=(2, 2))
        cases.append({'A': Absr.copy(), 'B': B.copy(), 'epsilon': 4.0, 'k': 2,
                      'proj': 'l2'})

        for ca in cases:
            np.random.seed(1778200711)  # make results deterministic
            result = evolution_soc(ca['A'], ca['B'], epsilon=ca['epsilon'],
                                   k=ca['k'], proj_type=ca['proj'],
                                   symmetrize_measure=False)
            np.random.seed(1778200711)  # make results deterministic
            expected = reference_evolution_soc(ca['A'], ca['B'],
                                               epsilon=ca['epsilon'],
                                               k=ca['k'], proj_type=ca['proj'])
            assert_array_almost_equal(result.toarray(), expected.toarray())

        # Test Scale Invariance for a single candidate
        A = 1.0j*poisson((5, 5), format='csr')
        B = 1.0j*np.arange(1, A.shape[0]+1, dtype=float).reshape(-1, 1)
        np.random.seed(1335104015)  # make results deterministic
        result_unscaled = evolution_soc(A, B, epsilon=4.0, k=2,
                                        proj_type='D_A',
                                        symmetrize_measure=False)
        # create scaled A
        D = sparse.spdiags([np.arange(A.shape[0], 2*A.shape[0], dtype=float)],
                           [0], A.shape[0], A.shape[0], format='csr')
        Dinv = sparse.spdiags([1.0/np.arange(A.shape[0], 2*A.shape[0], dtype=float)],
                              [0], A.shape[0], A.shape[0], format='csr')
        np.random.seed(743434914)  # make results deterministic
        result_scaled = evolution_soc(D*A*D, Dinv*B, epsilon=4.0, k=2,
                                      proj_type='D_A',
                                      symmetrize_measure=False)
        assert_array_almost_equal(result_scaled.toarray(),
                                  result_unscaled.toarray(), decimal=2)

        # Test that the l2 and D_A are the same for the 1 candidate case
        np.random.seed(2417151167)  # make results deterministic
        resultDA = evolution_soc(D*A*D, Dinv*B, epsilon=4.0,
                                 k=2, proj_type='D_A',
                                 symmetrize_measure=False)
        np.random.seed(2866319482)  # make results deterministic
        resultl2 = evolution_soc(D*A*D, Dinv*B, epsilon=4.0,
                                 k=2, proj_type='l2',
                                 symmetrize_measure=False)
        assert_array_almost_equal(resultDA.toarray(), resultl2.toarray())

        # Test Scale Invariance for multiple candidates
        (A, B) = linear_elasticity((5, 5), format='bsr')
        A = 1.0j*A
        B = 1.0j*B
        np.random.seed(3756761764)  # make results deterministic
        result_unscaled = evolution_soc(A, B, epsilon=4.0, k=2,
                                        proj_type='D_A',
                                        symmetrize_measure=False)
        # create scaled A
        D = sparse.spdiags([np.arange(A.shape[0], 2*A.shape[0], dtype=float)],
                           [0], A.shape[0], A.shape[0], format='csr')
        Dinv = sparse.spdiags([1.0/np.arange(A.shape[0], 2*A.shape[0], dtype=float)],
                              [0], A.shape[0], A.shape[0], format='csr')
        np.random.seed(1944403548)  # make results deterministic
        result_scaled = evolution_soc((D*A*D).tobsr(blocksize=(2, 2)), Dinv*B,
                                      epsilon=4.0, k=2, proj_type='D_A',
                                      symmetrize_measure=False)
        assert_array_almost_equal(result_scaled.toarray(),
                                  result_unscaled.toarray(), decimal=2)


# reference implementations for unittests  #
def reference_classical_soc(A, theta, norm='abs'):
    """Construct reference implementation of classical SOC.

    This complex extension of the classic Ruge-Stuben
    strength-of-connection has some theoretical justification in
    "AMG Solvers for Complex-Valued Matrices", Scott MacClachlan,
    Cornelis Oosterlee

    A connection is strong if,
      | a_ij| >= theta * max_{k != i} |a_ik|
    """
    S = sparse.coo_matrix(A)

    # remove diagonals
    mask = S.row != S.col
    S.row = S.row[mask]
    S.col = S.col[mask]
    S.data = S.data[mask]
    max_offdiag = np.empty(S.shape[0])
    max_offdiag[:] = np.finfo(S.data.dtype).min

    # Note abs(.) takes the complex modulus
    if norm == 'abs':
        for i, v in zip(S.row, S.data):
            max_offdiag[i] = max(max_offdiag[i], abs(v))
    if norm == 'min':
        for i, v in zip(S.row, S.data):
            max_offdiag[i] = max(max_offdiag[i], -v)

    # strong connections
    if norm == 'abs':
        mask = np.abs(S.data) >= (theta * max_offdiag[S.row])
    if norm == 'min':
        mask = -S.data >= (theta * max_offdiag[S.row])

    S.row = S.row[mask]
    S.col = S.col[mask]
    S.data = S.data[mask]

    # Add back diagonal
    D = sparse.eye(S.shape[0], S.shape[0], format='csr', dtype=A.dtype)
    D.data[:] = sparse.csr_matrix(A).diagonal()
    S = S.tocsr() + D

    # Strength represents "distance", so take the magnitude
    S.data = np.abs(S.data)

    # Scale S by the largest magnitude entry in each row
    largest_row_entry = np.zeros((S.shape[0],), dtype=S.dtype)
    for i in range(S.shape[0]):
        for j in range(S.indptr[i], S.indptr[i+1]):
            val = abs(S.data[j])
            if val > largest_row_entry[i]:
                largest_row_entry[i] = val

    largest_row_entry[largest_row_entry != 0] =\
        1.0 / largest_row_entry[largest_row_entry != 0]
    S = S.tocsr()
    S = scale_rows(S, largest_row_entry, copy=True)

    return S


def reference_symmetric_soc(A, theta):
    # This is just a direct complex extension of the classic
    # SA strength-of-connection measure.  The extension continues
    # to compare magnitudes. This should reduce to the classic
    # measure if A is all real.

    # if theta == 0:
    #    return A

    D = np.abs(A.diagonal())

    S = sparse.coo_matrix(A)

    mask = S.row != S.col
    DD = np.array(D[S.row] * D[S.col]).reshape(-1,)
    # Note that abs takes the complex modulus element-wise
    # Note that using the square of the measure is the technique used
    # in the C++ routine, so we use it here.  Doing otherwise causes errors.
    mask &= ((np.real(S.data)**2 + np.imag(S.data)**2) >= theta*theta*DD)

    S.row = S.row[mask]
    S.col = S.col[mask]
    S.data = S.data[mask]

    # Add back diagonal
    D = sparse.eye(S.shape[0], S.shape[0], format='csr', dtype=A.dtype)
    D.data[:] = sparse.csr_matrix(A).diagonal()
    S = S.tocsr() + D

    # Strength represents "distance", so take the magnitude
    S.data = np.abs(S.data)

    # Scale S by the largest magnitude entry in each row
    largest_row_entry = np.zeros((S.shape[0],), dtype=S.dtype)
    for i in range(S.shape[0]):
        for j in range(S.indptr[i], S.indptr[i+1]):
            val = abs(S.data[j])
            if val > largest_row_entry[i]:
                largest_row_entry[i] = val

    largest_row_entry[largest_row_entry != 0] =\
        1.0 / largest_row_entry[largest_row_entry != 0]
    S = S.tocsr()
    S = scale_rows(S, largest_row_entry, copy=True)

    return S


def reference_evolution_soc(A, B, epsilon=4.0, k=2, proj_type='l2'):
    """All python reference implementation for Evolution Strength of Connection.

    --> If doing imaginary test, both A and B should be imaginary type upon entry

    --> This does the "unsymmetrized" version of the ode measure
    """
    # number of PDEs per point is defined implicitly by block size
    csrflag = sparse.isspmatrix_csr(A)
    if csrflag:
        numPDEs = 1
    else:
        numPDEs = A.blocksize[0]
        A = A.tocsr()

    # Preliminaries
    near_zero = np.finfo(float).eps
    sqrt_near_zero = np.sqrt(np.sqrt(near_zero))
    Bmat = np.array(B)
    A.eliminate_zeros()
    A.sort_indices()
    dimen = A.shape[1]
    NullDim = Bmat.shape[1]

    # Get spectral radius of Dinv*A, this is the time step size for the ODE
    D = A.diagonal()
    Dinv = np.zeros_like(D)
    mask = (D != 0.0)
    Dinv[mask] = 1.0 / D[mask]
    Dinv[D == 0] = 1.0
    Dinv_A = scale_rows(A, Dinv, copy=True)
    rho_DinvA = approximate_spectral_radius(Dinv_A)

    # Calculate (Atilde^k) naively
    S = (sparse.eye(dimen, dimen, format='csr') - (1.0/rho_DinvA)*Dinv_A)
    Atilde = sparse.eye(dimen, dimen, format='csr')
    for _i in range(k):
        Atilde = S * Atilde

    # Strength Info should be row-based, so transpose Atilde
    Atilde = Atilde.T.tocsr()

    # Construct and apply a sparsity mask for Atilde that restricts Atilde^T to
    # the nonzero pattern of A, with the added constraint that row i of
    # Atilde^T retains only the nonzeros that are also in the same PDE as i.

    mask = A.copy()

    # Only consider strength at dofs from your PDE.  Use mask to enforce this
    # by zeroing out all entries in Atilde that aren't from your PDE.
    if numPDEs > 1:
        row_length = np.diff(mask.indptr)
        my_pde = np.mod(np.arange(dimen), numPDEs)
        my_pde = np.repeat(my_pde, row_length)
        mask.data[np.mod(mask.indices, numPDEs) != my_pde] = 0.0
        del row_length, my_pde
        mask.eliminate_zeros()

    # Apply mask to Atilde, zeros in mask have already been eliminated at start
    # of routine.
    mask.data[:] = 1.0
    Atilde = Atilde.multiply(mask)
    Atilde.eliminate_zeros()
    Atilde.sort_indices()
    del mask

    # Calculate strength based on constrained min problem of
    LHS = np.zeros((NullDim+1, NullDim+1), dtype=A.dtype)
    RHS = np.zeros((NullDim+1,), dtype=A.dtype)

    # Choose tolerance for dropping "numerically zero" values later
    tol = set_tol(Atilde.dtype)

    for i in range(dimen):

        # Get rowptrs and col indices from Atilde
        rowstart = Atilde.indptr[i]
        rowend = Atilde.indptr[i+1]
        length = rowend - rowstart
        colindx = Atilde.indices[rowstart:rowend]

        # Local diagonal of A is used for scale invariant min problem
        D_A = np.eye(length, dtype=A.dtype)
        if proj_type == 'D_A':
            for j in range(length):
                D_A[j, j] = D[colindx[j]]

        # Find row i's position in colindx, matrix must have sorted column
        # indices.
        iInRow = colindx.searchsorted(i)

        if length <= NullDim:
            # Do nothing, because the number of nullspace vectors will
            # be able to perfectly approximate this row of Atilde.
            Atilde.data[rowstart:rowend] = 1.0
        else:
            # Grab out what we want from Atilde and B.  Put into zi, Bi
            zi = Atilde.data[rowstart:rowend]

            Bi = Bmat[colindx, :]

            # Construct constrained min problem
            # CC = D_A[iInRow, iInRow]
            LHS[0:NullDim, 0:NullDim] = 2.0*Bi.conj().T.dot(D_A.dot(Bi))
            LHS[0:NullDim, NullDim] = D_A[iInRow, iInRow]*(Bi[iInRow, :].conj().T)
            LHS[NullDim, 0:NullDim] = Bi[iInRow, :]
            RHS[0:NullDim] = 2.0*Bi.conj().T.dot(D_A.dot(zi))
            RHS[NullDim] = zi[iInRow]

            # Calc Soln to Min Problem
            x = sla.pinv(LHS).dot(RHS)

            # Calculate best constrained approximation to zi with span(Bi), and
            # filter out "numerically" zero values.  This is important because
            # we look only at the sign of values below when calculating angle.
            zihat = Bi.dot(x[:-1])
            tol_i = np.max(np.abs(zihat))*tol
            zihat.real[np.abs(zihat.real) < tol_i] = 0.0
            if np.iscomplexobj(zihat):
                zihat.imag[np.abs(zihat.imag) < tol_i] = 0.0

            # if angle in the complex plane between individual entries is
            # greater than 90 degrees, then weak.  We can just look at the dot
            # product to determine if angle is greater than 90 degrees.
            angle = np.real(np.ravel(zihat))*np.real(np.ravel(zi)) +\
                np.imag(np.ravel(zihat))*np.imag(np.ravel(zi))
            angle = angle < 0.0
            angle = np.array(angle, dtype=bool)

            # Calculate approximation ratio
            zi = zihat/zi

            # If the ratio is small, then weak connection
            zi[np.abs(zi) <= 1e-4] = 1e100

            # If angle is greater than 90 degrees, then weak connection
            zi[angle] = 1e100

            # Calculate Relative Approximation Error
            zi = np.abs(1.0 - zi)

            # important to make "perfect" connections explicitly nonzero
            zi[zi < sqrt_near_zero] = 1e-4

            # Calculate and apply drop-tol.  Ignore diagonal by making it very
            # large
            zi[iInRow] = 1e5
            drop_tol = np.min(zi)*epsilon
            zi[zi > drop_tol] = 0.0
            Atilde.data[rowstart:rowend] = np.ravel(zi)

    # Clean up, and return Atilde
    Atilde.eliminate_zeros()
    Atilde.data = np.array(np.real(Atilde.data), dtype=float)

    # Set diagonal to 1.0, as each point is strongly connected to itself.
    Id = sparse.eye(dimen, dimen, format='csr')
    Id.data -= Atilde.diagonal()
    Atilde = Atilde + Id

    # If converted BSR to CSR we return amalgamated matrix with the minimum
    # nonzero for each block making up the nonzeros of Atilde
    if not csrflag:
        Atilde = Atilde.tobsr(blocksize=(numPDEs, numPDEs))

        # Atilde = sparse.csr_matrix((data, row, col), shape=(*,*))
        At = []
        for i in range(Atilde.indices.shape[0]):
            Atmin = Atilde.data[i, :, :][Atilde.data[i, :, :].nonzero()]
            At.append(Atmin.min())

        Atilde = sparse.csr_matrix((np.array(At), Atilde.indices, Atilde.indptr),
                                   shape=(int(Atilde.shape[0]/numPDEs),
                                          int(Atilde.shape[1]/numPDEs)))

    # Standardized strength values require small values be weak and large
    # values be strong.  So, we invert the algebraic distances computed here
    Atilde.data = 1.0/Atilde.data

    # Scale Atilde by the largest magnitude entry in each row
    largest_row_entry = np.zeros((Atilde.shape[0],), dtype=Atilde.dtype)
    for i in range(Atilde.shape[0]):
        for j in range(Atilde.indptr[i], Atilde.indptr[i+1]):
            val = abs(Atilde.data[j])
            if val > largest_row_entry[i]:
                largest_row_entry[i] = val

    largest_row_entry[largest_row_entry != 0] =\
        1.0 / largest_row_entry[largest_row_entry != 0]
    Atilde = Atilde.tocsr()
    Atilde = scale_rows(Atilde, largest_row_entry, copy=True)

    return Atilde


def reference_distance_soc(A, V, theta=2.0, relative_drop=True):
    """Construct reference distance based strength of connection."""
    # deal with the supernode case
    if sparse.isspmatrix_bsr(A):
        dimen = int(A.shape[0]/A.blocksize[0])
        C = sparse.csr_matrix((np.ones((A.data.shape[0],)), A.indices, A.indptr),
                              shape=(dimen, dimen))
    else:
        A = A.tocsr()
        dimen = A.shape[0]
        C = A.copy()
        C.data = np.real(C.data)

    if V.shape[1] == 2:
        three_d = False
    elif V.shape[1] == 3:
        three_d = True

    for i in range(dimen):
        rowstart = C.indptr[i]
        rowend = C.indptr[i+1]
        pt_i = V[i, :]
        for j in range(rowstart, rowend):
            if C.indices[j] == i:
                # ignore the diagonal entry by making it large
                C.data[j] = np.finfo(np.float64).max
            else:
                # distance between entry j and i
                pt_j = V[C.indices[j], :]
                dist = (pt_i[0] - pt_j[0])**2
                dist += (pt_i[1] - pt_j[1])**2
                if three_d:
                    dist += (pt_i[2] - pt_j[2])**2
                C.data[j] = np.sqrt(dist)

        # apply drop tolerance
        this_row = C.data[rowstart:rowend]
        if relative_drop:
            tol_i = theta*this_row.min()
            this_row[this_row > tol_i] = 0.0
        else:
            this_row[this_row > theta] = 0.0

        C.data[rowstart:rowend] = this_row

    C.eliminate_zeros()
    C = C + 2.0*sparse.eye(C.shape[0], C.shape[1], format='csr')

    # Standardized strength values require small values be weak and large
    # values be strong.  So, we invert the distances.
    C.data = 1.0/C.data

    # Scale C by the largest magnitude entry in each row
    largest_row_entry = np.zeros((C.shape[0],), dtype=C.dtype)
    for i in range(C.shape[0]):
        for j in range(C.indptr[i], C.indptr[i+1]):
            val = abs(C.data[j])
            if val > largest_row_entry[i]:
                largest_row_entry[i] = val

    largest_row_entry[largest_row_entry != 0] =\
        1.0 / largest_row_entry[largest_row_entry != 0]
    C = C.tocsr()
    C = scale_rows(C, largest_row_entry, copy=True)

    return C
