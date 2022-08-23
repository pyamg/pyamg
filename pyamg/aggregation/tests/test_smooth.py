"""Test prolongation smoothing."""
import warnings
import numpy as np
from numpy.testing import TestCase, assert_array_almost_equal,\
    assert_equal, assert_almost_equal
from scipy import sparse

from pyamg.gallery import poisson, linear_elasticity, load_example,\
    gauge_laplacian
from pyamg.aggregation import smoothed_aggregation_solver, rootnode_solver
from pyamg.amg_core import incomplete_mat_mult_bsr


class TestEnergyMin(TestCase):
    def test_incomplete_mat_mult_bsr(self):

        # Define incomplete mat mult bsr gold
        def incomplete_mat_mult_bsr_gold(A, B, S):
            """Compute A*B --> S.

            Compute only at the existing sparsity structure of S
            A,B and S are assumed BSR
            """
            # Ablocksize = A.blocksize
            # Bblocksize = B.blocksize
            S = S.copy()   # don't overwrite the original S

            S.data[:] = 1.0
            SS = A * B
            SS = SS.multiply(S)

            # Union the sparsity patterns of SS and S, storing
            # explicit zeros, where S has an entry, but SS does not
            S = S.tocoo()
            SS = SS.tocoo()
            S.data[:] = 0.0
            SS = sparse.coo_matrix((np.hstack((S.data, SS.data)),
                                    (np.hstack((S.row, SS.row)),
                                     np.hstack((S.col, SS.col)))),
                                   shape=S.shape)
            # Convert back to BSR
            SS = SS.tobsr((A.blocksize[0], B.blocksize[1]))
            SS.sort_indices()
            return SS

        # Test a critical helper routine for energy_min_prolongation(...)
        # We test that (A*B).multiply(mask) = incomplete_mat_mult_bsr(A,B,mask)
        cases = []

        # 1x1 tests
        A = sparse.csr_matrix(np.array([[1.1]])).tobsr(blocksize=(1, 1))
        B = sparse.csr_matrix(np.array([[1.0]])).tobsr(blocksize=(1, 1))
        A2 = sparse.csr_matrix(np.array([[0.]])).tobsr(blocksize=(1, 1))
        mask = sparse.csr_matrix(np.array([[1.]])).tobsr(blocksize=(1, 1))
        mask2 = sparse.csr_matrix(np.array([[0.]])).tobsr(blocksize=(1, 1))
        cases.append((A, A, mask))                                  # 1x1,  1x1
        cases.append((A, A, mask2))                                 # 1x1,  1x1
        cases.append((A, B, mask))                                  # 1x1,  1x1
        cases.append((A, A2, mask))                                 # 1x1,  1x1
        cases.append((A, A2, mask2))                                # 1x1,  1x1
        cases.append((A2, A2, mask))                                # 1x1,  1x1

        # 1x2 and 2x1 tests
        A = sparse.csr_matrix(np.array([[1.1]])).tobsr(blocksize=(1, 1))
        B = sparse.csr_matrix(np.array([[1.0, 2.3]])).tobsr(blocksize=(1, 1))
        A2 = sparse.csr_matrix(np.array([[0.]])).tobsr(blocksize=(1, 1))
        mask = sparse.csr_matrix(np.array([[1., 1.]])).tobsr(blocksize=(1, 1))
        mask2 = sparse.csr_matrix(np.array([[0., 1.]])).tobsr(blocksize=(1, 1))
        cases.append((A, B, mask))                                  # 1x1,  1x2
        cases.append((A2, B, mask))                                 # 1x1,  1x2
        cases.append((A, B, mask2))                                 # 1x1,  1x2
        cases.append((A2, B, mask2))                                # 1x1,  1x2

        B2 = sparse.csr_matrix(np.array([[0.0, 2.3]])).tobsr(blocksize=(1, 1))
        B3 = sparse.csr_matrix(np.array([[3.0, 0.0]])).tobsr(blocksize=(1, 1))
        mask = sparse.csr_matrix(np.array([[1., 1.],
                                           [1., 1.]])).tobsr(blocksize=(1, 1))
        mask3 = sparse.csr_matrix(np.array([[0., 1.],
                                            [1., 0.]])).tobsr(blocksize=(1, 1))
        cases.append((B.T.copy(), B2, mask))                      # 2x1,  1x2
        cases.append((B.T.copy(), B2, mask3))                     # 2x1,  1x2
        cases.append((B.T.copy(), B3, mask))                      # 2x1,  1x2
        cases.append((B.T.copy(), B3, mask3))                     # 2x1,  1x2

        B = B.tobsr(blocksize=(1, 2))
        B2 = B2.tobsr(blocksize=(1, 2))
        B3 = B3.tobsr(blocksize=(1, 2))
        mask = sparse.csr_matrix(np.array([[1., 1.],
                                           [1., 1.]])).tobsr(blocksize=(2, 2))
        mask2 = sparse.csr_matrix(np.array([[0., 0.],
                                            [0., 0.]])).tobsr(blocksize=(2, 2))
        cases.append((B.T.copy(), B2, mask))                      # 2x1,  1x2
        cases.append((B.T.copy(), B2, mask2))                     # 2x1,  1x2
        cases.append((B.T.copy(), B3, mask))                      # 2x1,  1x2
        cases.append((B.T.copy(), B3, mask2))                     # 2x1,  1x2

        B = B.tobsr(blocksize=(1, 1))
        B2 = B2.tobsr(blocksize=(1, 1))
        B3 = B3.tobsr(blocksize=(1, 1))
        mask = sparse.csr_matrix(np.array([[1.]])).tobsr(blocksize=(1, 1))
        mask2 = sparse.csr_matrix(np.array([[0.]])).tobsr(blocksize=(1, 1))
        cases.append((B, B2.T.copy(), mask))                      # 1x2,  2x1
        cases.append((B, B2.T.copy(), mask2))                     # 1x2,  2x1
        cases.append((B, B3.T.copy(), mask))                      # 1x2,  2x1
        cases.append((B, B3.T.copy(), mask2))                     # 1x2,  2x1

        B = B.tobsr(blocksize=(1, 2))
        B2 = B2.tobsr(blocksize=(1, 2))
        B3 = B3.tobsr(blocksize=(1, 2))
        cases.append((B, B2.T.copy(), mask))                      # 1x2,  2x1
        cases.append((B, B2.T.copy(), mask2))                     # 1x2,  2x1
        cases.append((B, B3.T.copy(), mask))                      # 1x2,  2x1
        cases.append((B, B3.T.copy(), mask2))                     # 1x2,  2x1

        # 2x2 tests
        A = sparse.csr_matrix(np.array([[1., 2.], [2., 4.]])).tobsr(blocksize=(2, 2))
        B = sparse.csr_matrix(np.array([[1.3, 2.], [2.8, 4.]])).tobsr(blocksize=(2, 2))
        A2 = sparse.csr_matrix(np.array([[1.3, 0.], [0., 4.]])).tobsr(blocksize=(2, 2))
        B2 = sparse.csr_matrix(np.array([[1.3, 0.], [2., 4.]])).tobsr(blocksize=(2, 1))
        mask = sparse.csr_matrix((np.array([1., 1., 1., 1.]),
                                 (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))),
                                 shape=(2, 2)).tobsr(blocksize=(2, 2))
        mask2 = sparse.csr_matrix((np.array([1., 0., 1., 0.]),
                                  (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))),
                                  shape=(2, 2)).tobsr(blocksize=(2, 1))
        mask2.eliminate_zeros()
        cases.append((A, A, mask))                                  # 2x2,  2x2
        cases.append((A, B, mask))                                  # 2x2,  2x2
        cases.append((A2, A2, mask))                                # 2x2,  2x2
        cases.append((A2, B2, mask2))                               # 2x2,  2x2
        cases.append((A, B2, mask2))                                # 2x2,  2x2

        A = A.tobsr(blocksize=(1, 1))
        B = B.tobsr(blocksize=(1, 1))
        A2 = A2.tobsr(blocksize=(1, 1))
        B2 = B2.tobsr(blocksize=(1, 1))
        mask = A.copy()
        mask2 = A2.copy()
        mask3 = sparse.csr_matrix((np.ones(2, dtype=float),
                                  (np.array([0, 1]), np.array([0, 0]))),
                                  shape=(2, 2)).tobsr(blocksize=(1, 1))
        cases.append((A, B, mask))                                  # 2x2,  2x2
        cases.append((A2, B2, mask2))                               # 2x2,  2x2
        cases.append((A, B, mask3))                                 # 2x2,  2x2
        cases.append((A2, B2, mask3))                               # 2x2,  2x2

        B = sparse.csr_matrix(np.array([[1.0], [2.3]])).tobsr(blocksize=(1, 1))
        B2 = sparse.csr_matrix(np.array([[1.1], [0.0]])).tobsr(blocksize=(1, 1))
        mask = sparse.csr_matrix(np.array([[1.], [1.1]])).tobsr(blocksize=(1, 1))
        mask2 = sparse.csr_matrix(np.array([[0.], [1.1]])).tobsr(blocksize=(1, 1))
        cases.append((A, B, mask))                                 # 2x2,  2x1
        cases.append((A, B2, mask2))                                # 2x2,  2x1
        cases.append((A2, B, mask))                                 # 2x2,  2x1
        cases.append((A2, B2, mask2))                               # 2x2,  2x1

        A = A.tobsr(blocksize=(1, 1))
        B = B.tobsr(blocksize=(1, 1))
        A2 = A2.tobsr(blocksize=(2, 2))
        B2 = B2.tobsr(blocksize=(2, 1))
        mask = mask.tobsr(blocksize=(1, 1))
        mask2 = mask.tobsr(blocksize=(2, 1))
        cases.append((A, B, mask))                                 # 2x2,  2x1
        cases.append((A2, B2, mask2))                              # 2x2,  2x1

        B = sparse.csr_matrix(np.array([[1.3, 2., 1.0], [2.8, 4., 0.]]))
        B = B.tobsr(blocksize=(1, 1))
        B2 = sparse.csr_matrix(np.array([[1.3, 2., 1.0], [2.8, 4., 0.]]))
        B2 = B2.tobsr(blocksize=(2, 3))
        mask = sparse.csr_matrix(np.array([[1., 2., 1.], [1., 2., 0.]]))
        mask = mask.tobsr(blocksize=(1, 1))
        mask2 = mask.tobsr((2, 3))
        mask3 = sparse.csr_matrix(np.array([[0., 0., 0.], [0., 0., 0.]]))
        mask3 = mask3.tobsr(blocksize=(2, 3))
        cases.append((A, B, mask))                                 # 2x2,  2x3
        cases.append((A2, B2, mask2))                              # 2x2,  2x3
        cases.append((A2, B2, mask3))                              # 2x2,  2x3
        cases.append((A, B, mask))                                 # 2x2,  2x3
        cases.append((A2, B2, mask2))                              # 2x2,  2x3
        cases.append((A2, B2, mask3))                              # 2x2,  2x3

        # 4x4 tests
        A = np.array([[0., 16.9, 6.4, 0.0],
                      [16.9, 13.8, 7.2, 0.],
                      [6.4, 7.2, 12., 6.1],
                      [0.0, 0., 6.1, 0.]])
        B = A.copy()
        B = A[:, 0:2]
        B[1, 0] = 3.1
        B[2, 1] = 10.1
        A2 = A.copy()
        A2[1, :] = 0.0
        A3 = A2.copy()
        A3[:, 1] = 0.0
        A = sparse.csr_matrix(A).tobsr(blocksize=(2, 2))
        A2 = sparse.csr_matrix(A2).tobsr(blocksize=(2, 2))
        A3 = sparse.csr_matrix(A3).tobsr(blocksize=(2, 2))
        B = sparse.csr_matrix(B).tobsr(blocksize=(2, 2))
        B2 = sparse.csr_matrix(B).tobsr(blocksize=(2, 1))

        mask = A.copy()
        mask2 = B.copy()
        mask3 = B2.copy()
        cases.append((A, B, mask2))                                 # 4x4,  4x2
        cases.append((A2, B, mask2))                                # 4x4,  4x2
        cases.append((A3, B, mask2))                                # 4x4,  4x2
        cases.append((A3, A3, mask))                                # 4x4,  4x4
        cases.append((A, A, mask))                                  # 4x4,  4x4
        cases.append((A2, A2, mask))                                # 4x4,  4x4
        cases.append((A, A2, mask))                                 # 4x4,  4x4
        cases.append((A3, A, mask))                                 # 4x4,  4x4
        cases.append((A, B2, mask3))                               # 4x4,  4x2
        cases.append((A2, B2, mask3))                              # 4x4,  4x2
        cases.append((A3, B2, mask3))                              # 4x4,  4x2

        # Laplacian tests, (tests zero rows, too)
        A = poisson((5, 5), format='csr')
        A = A.tobsr(blocksize=(5, 5))
        Ai = 1.0j * A
        B = A.toarray()
        B = sparse.csr_matrix(B[:, 0:8])
        B = B.tobsr(blocksize=(5, 8))
        Bi = 1.0j * B
        B2 = B.tobsr(blocksize=(5, 2))
        B2.eliminate_zeros()
        B2i = 1.0j * B2
        B3 = B.tobsr(blocksize=(5, 1))
        B3.eliminate_zeros()
        B3i = 1.0j * B3
        mask = B.copy()
        mask2 = B2.copy()
        mask3 = B3.copy()
        cases.append((A, A, A.copy()))                         # 25x25,  25x25
        cases.append((A, B, mask))                             # 25x25,  25x8
        cases.append((A, B2, mask2))                           # 25x25,  25x8
        cases.append((A, B3, mask3))                           # 25x25,  25x8
        cases.append((Ai, Bi, mask))                           # 25x25,  25x8
        cases.append((Ai, B2i, mask2))                         # 25x25,  25x8
        cases.append((Ai, B3i, mask3))                         # 25x25,  25x8
        cases.append((Ai, Ai, Ai.copy()))                      # 25x25,  25x25

        for case in cases:
            A = case[0].tobsr()
            B = case[1].tobsr()
            mask = case[2].tobsr()

            # Test A*B --> result
            result = mask.copy()
            result.data = np.array(result.data, dtype=A.dtype)
            result.data[:] = 0.0
            incomplete_mat_mult_bsr(A.indptr, A.indices, np.ravel(A.data),
                                    B.indptr, B.indices, np.ravel(B.data),
                                    result.indptr, result.indices,
                                    np.ravel(result.data),
                                    int(A.shape[0] / A.blocksize[0]),
                                    int(result.shape[1] / result.blocksize[1]),
                                    A.blocksize[0], A.blocksize[1],
                                    B.blocksize[1])
            exact = incomplete_mat_mult_bsr_gold(A, B, mask)
            assert_array_almost_equal(result.data, exact.data)
            assert_array_almost_equal(result.indices, exact.indices)
            assert_array_almost_equal(result.indptr, exact.indptr)

            # Test B.T*A.T --> result.T
            A = A.T.copy()
            B = B.T.copy()
            mask = mask.T.copy()
            result = mask.copy()
            result.data = np.array(result.data, dtype=A.dtype)
            result.data[:] = 0.0
            incomplete_mat_mult_bsr(B.indptr, B.indices, np.ravel(B.data),
                                    A.indptr, A.indices, np.ravel(A.data),
                                    result.indptr, result.indices,
                                    np.ravel(result.data),
                                    int(B.shape[0] / B.blocksize[0]),
                                    int(result.shape[1] / result.blocksize[1]),
                                    B.blocksize[0], B.blocksize[1],
                                    A.blocksize[1])
            exact = incomplete_mat_mult_bsr_gold(B, A, mask)
            assert_array_almost_equal(result.data, exact.data)
            assert_array_almost_equal(result.indices, exact.indices)
            assert_array_almost_equal(result.indptr, exact.indptr)

    def test_range(self):
        """Check that P*R=B."""
        warnings.filterwarnings('ignore', category=UserWarning,
                                message='Having less target vectors')
        np.random.seed(18410243)  # make tests repeatable

        cases = []

        # Simple, real-valued diffusion problems
        name = 'airfoil'
        X = load_example('airfoil')
        A = X['A'].tocsr()
        B = X['B']
        cases.append((A, B, ('jacobi',
                             {'filter_entries': True, 'weighting': 'local'}), name))
        cases.append((A, B, ('jacobi',
                             {'filter_entries': True, 'weighting': 'block'}), name))

        cases.append((A, B, ('energy', {'maxiter': 3}), name))
        cases.append((A, B, ('energy', {'krylov': 'cgnr', 'weighting': 'diagonal'}), name))
        cases.append((A, B, ('energy', {'krylov': 'gmres', 'degree': 2}), name))

        name = 'poisson'
        A = poisson((10, 10), format='csr')
        B = np.ones((A.shape[0], 1))
        cases.append((A, B, ('jacobi',
                             {'filter_entries': True, 'weighting': 'diagonal'}), name))
        cases.append((A, B, ('jacobi',
                             {'filter_entries': True, 'weighting': 'local'}), name))

        cases.append((A, B, 'energy', name))
        cases.append((A, B, ('energy', {'degree': 2}), name))
        cases.append((A, B, ('energy', {'krylov': 'cgnr', 'degree': 2,
                                        'weighting': 'diagonal'}), name))
        cases.append((A, B, ('energy', {'krylov': 'gmres'}), name))

        # Simple, imaginary-valued problems
        name = 'random imaginary'
        iA = 1.0j * A
        iB = 1.0 + np.random.rand(iA.shape[0], 2)\
                 + 1.0j * (1.0 + np.random.rand(iA.shape[0], 2))

        cases.append((iA, B, ('jacobi',
                              {'filter_entries': True, 'weighting': 'diagonal'}), name))
        cases.append((iA, B, ('jacobi',
                              {'filter_entries': True, 'weighting': 'block'}), name))
        cases.append((iA, iB, ('jacobi',
                               {'filter_entries': True, 'weighting': 'local'}), name))
        cases.append((iA, iB, ('jacobi',
                               {'filter_entries': True, 'weighting': 'block'}), name))

        cases.append((iA.tobsr(blocksize=(5, 5)), B,
                      ('jacobi', {'filter_entries': True, 'weighting': 'block'}), name))
        cases.append((iA.tobsr(blocksize=(5, 5)), iB,
                      ('jacobi', {'filter_entries': True, 'weighting': 'block'}), name))

        cases.append((iA, B, ('energy', {'krylov': 'cgnr', 'degree': 2,
                                         'weighting': 'diagonal'}), name))
        cases.append((iA, iB, ('energy', {'krylov': 'cgnr',
                                          'weighting': 'diagonal'}), name))
        cases.append((iA.tobsr(blocksize=(5, 5)), B,
                      ('energy',
                       {'krylov': 'cgnr', 'degree': 2, 'maxiter': 3,
                        'weighting': 'diagonal', 'postfilter': {'theta': 0.05}}), name))
        cases.append((iA.tobsr(blocksize=(5, 5)), B,
                      ('energy',
                       {'krylov': 'cgnr', 'degree': 2, 'maxiter': 3,
                        'weighting': 'diagonal',
                        'prefilter': {'theta': 0.05}}), name))
        cases.append((iA.tobsr(blocksize=(5, 5)), B,
                      ('energy',
                       {'krylov': 'cgnr', 'degree': 2,
                        'weighting': 'diagonal', 'maxiter': 3}), name))
        cases.append((iA.tobsr(blocksize=(5, 5)), iB,
                      ('energy', {'krylov': 'cgnr', 'weighting': 'diagonal'}), name))

        cases.append((iA, B, ('energy', {'krylov': 'gmres'}), name))
        cases.append((iA, iB, ('energy', {'krylov': 'gmres', 'degree': 2}), name))
        cases.append((iA.tobsr(blocksize=(5, 5)), B,
                      ('energy',
                       {'krylov': 'gmres', 'degree': 2, 'maxiter': 3}), name))
        cases.append((iA.tobsr(blocksize=(5, 5)), iB,
                      ('energy', {'krylov': 'gmres'}), name))

        # Simple, imaginary-valued problems
        name = 'random imaginary + I'
        iA = A + 1.0j * sparse.eye(A.shape[0], A.shape[1])

        cases.append((iA, B, ('jacobi',
                              {'filter_entries': True, 'weighting': 'local'}), name))
        cases.append((iA, B, ('jacobi',
                              {'filter_entries': True, 'weighting': 'block'}), name))
        cases.append((iA, iB, ('jacobi',
                               {'filter_entries': True, 'weighting': 'diagonal'}), name))
        cases.append((iA, iB, ('jacobi',
                               {'filter_entries': True, 'weighting': 'block'}), name))
        cases.append((iA.tobsr(blocksize=(4, 4)), iB,
                      ('jacobi', {'filter_entries': True, 'weighting': 'block'}), name))

        cases.append((iA, B, ('energy', {'krylov': 'cgnr', 'weighting': 'diagonal'}), name))
        cases.append((iA.tobsr(blocksize=(4, 4)), iB,
                      ('energy', {'krylov': 'cgnr', 'weighting': 'diagonal'}), name))

        cases.append((iA, B, ('energy', {'krylov': 'gmres'}), name))
        cases.append((iA.tobsr(blocksize=(4, 4)), iB,
                      ('energy',
                       {'krylov': 'gmres', 'degree': 2, 'maxiter': 3}), name))

        cases.append((iA.tobsr(blocksize=(4, 4)), iB,
                      ('energy',
                       {'krylov': 'gmres', 'degree': 2, 'maxiter': 3,
                        'postfilter': {'theta': 0.05}}), name))
        cases.append((iA.tobsr(blocksize=(4, 4)), iB,
                      ('energy',
                       {'krylov': 'gmres', 'degree': 2, 'maxiter': 3,
                        'prefilter': {'theta': 0.05}}), name))

        name = 'gauge laplacian'
        A = gauge_laplacian(10, spacing=1.0, beta=0.21)
        B = np.ones((A.shape[0], 1))
        cases.append((A, iB, ('jacobi',
                              {'filter_entries': True, 'weighting': 'diagonal'}), name))
        cases.append((A, iB, ('jacobi',
                              {'filter_entries': True, 'weighting': 'local'}), name))

        cases.append((A, B, ('energy', {'krylov': 'cg'}), name))
        cases.append((A, iB, ('energy', {'krylov': 'cgnr', 'weighting': 'diagonal'}), name))
        cases.append((A, iB, ('energy', {'krylov': 'gmres'}), name))

        name = 'gauge laplacian bsr'
        cases.append((A.tobsr(blocksize=(2, 2)), B,
                      ('energy', {'krylov': 'cgnr', 'degree': 2, 'weighting': 'diagonal',
                                  'maxiter': 3, 'postfilter': {'theta': 0.05}}),
                     name))
        cases.append((A.tobsr(blocksize=(2, 2)), B,
                      ('energy', {'krylov': 'cgnr', 'degree': 2, 'weighting': 'diagonal',
                       'maxiter': 3, 'prefilter': {'theta': 0.05}}),
                     name))
        cases.append((A.tobsr(blocksize=(2, 2)), B,
                      ('energy', {'krylov': 'cgnr', 'degree': 2, 'maxiter': 3,
                                  'weighting': 'diagonal'}),
                      name))
        cases.append((A.tobsr(blocksize=(2, 2)), iB,
                     ('energy', {'krylov': 'cg'}), name))
        cases.append((A.tobsr(blocksize=(2, 2)), B,
                      ('energy', {'krylov': 'gmres', 'degree': 2, 'maxiter': 3}),
                      name))
        cases.append((A.tobsr(blocksize=(2, 2)), B,
                     ('energy', {'krylov': 'gmres', 'degree': 2,
                      'maxiter': 3, 'postfilter': {'theta': 0.05}}),
                     name))
        cases.append((A.tobsr(blocksize=(2, 2)), B,
                     ('energy', {'krylov': 'gmres', 'degree': 2,
                      'maxiter': 3, 'prefilter': {'theta': 0.05}}),
                     name))

        #
        name = 'linear elasticity'
        A, B = linear_elasticity((10, 10))
        cases.append((A, B, ('jacobi',
                             {'filter_entries': True, 'weighting': 'diagonal'}), name))
        cases.append((A, B, ('jacobi',
                             {'filter_entries': True, 'weighting': 'local'}), name))
        cases.append((A, B, ('jacobi',
                             {'filter_entries': True, 'weighting': 'block'}), name))
        cases.append((A, B, ('energy', {'degree': 2}), name))
        cases.append((A, B, ('energy', {'degree': 3, 'postfilter': {'theta': 0.05}}), name))
        cases.append((A, B, ('energy', {'degree': 3, 'prefilter': {'theta': 0.05}}), name))
        cases.append((A, B, ('energy', {'krylov': 'cgnr', 'weighting': 'diagonal'}), name))
        cases.append((A, B, ('energy', {'krylov': 'gmres', 'degree': 2}), name))

        # Classic SA cases
        for A, B, smooth, _name in cases:
            ml = smoothed_aggregation_solver(A, B=B, max_coarse=1,
                                             max_levels=2, smooth=smooth)
            P = ml.levels[0].P
            B = ml.levels[0].B
            R = ml.levels[1].B
            assert_almost_equal(P * R, B)

        def _get_blocksize(A):
            # Helper Function: return the blocksize of a matrix
            if sparse.isspmatrix_bsr(A):
                return A.blocksize[0]

            return 1

        # Root-node cases
        counter = 0
        for A, B, smooth, _name in cases:
            counter += 1

            if isinstance(smooth, tuple):
                smoother = smooth[0]
            else:
                smoother = smooth

            if smoother == 'energy' and (B.shape[1] >= _get_blocksize(A)):
                ic = [('gauss_seidel_nr',
                       {'sweep': 'symmetric', 'iterations': 4}), None]
                ml = rootnode_solver(A, B=B, max_coarse=1, max_levels=2,
                                     smooth=smooth,
                                     improve_candidates=ic,
                                     keep=True, symmetry='nonsymmetric')
                T = ml.levels[0].T.tocsr()
                Cpts = ml.levels[0].Cpts
                Bf = ml.levels[0].B
                Bf_H = ml.levels[0].BH
                Bc = ml.levels[1].B
                P = ml.levels[0].P.tocsr()

                T.eliminate_zeros()
                P.eliminate_zeros()

                # P should preserve B in its range, wherever P
                # has enough nonzeros
                mask = ((P.indptr[1:] - P.indptr[:-1]) >= B.shape[1])
                assert_almost_equal((P*Bc)[mask, :], Bf[mask, :])
                assert_almost_equal((P*Bc)[mask, :], Bf_H[mask, :])

                # P should be the identity at Cpts
                I1 = sparse.eye(T.shape[1], T.shape[1], format='csr', dtype=T.dtype)
                I2 = P[Cpts, :]
                assert_almost_equal(I1.data, I2.data)
                assert_equal(I1.indptr, I2.indptr)
                assert_equal(I1.indices, I2.indices)

                # T should be the identity at Cpts
                I2 = T[Cpts, :]
                assert_almost_equal(I1.data, I2.data)
                assert_equal(I1.indptr, I2.indptr)
                assert_equal(I1.indices, I2.indices)

    def test_postfilter(self):
        """Check that using postfilter reduces NNZ in P."""
        np.random.seed(3198379291)  # make tests repeatable
        cases = []

        # Simple, real-valued diffusion problems
        X = load_example('airfoil')
        A = X['A'].tocsr()
        B = X['B']

        cases.append((A, B,
                      ('energy', {'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
                      {'theta': 0.05}))

        cases.append((A, B,
                      ('energy', {'krylov': 'gmres', 'degree': 2,
                                  'maxiter': 3}),
                      {'k': 3}))

        cases.append((A.tobsr(blocksize=(2, 2)),
                      np.hstack((B, np.random.rand(B.shape[0], 1))),
                      ('energy', {'krylov': 'cg', 'degree': 2,
                                  'maxiter': 3}),
                      {'theta': 0.1}))

        # Simple, imaginary-valued problems
        iA = 1.0j * A
        iB = 1.0 + np.random.rand(iA.shape[0], 2)\
                 + 1.0j * (1.0 + np.random.rand(iA.shape[0], 2))

        cases.append((iA, B,
                      ('energy', {'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
                      {'theta': 0.05}))

        cases.append((iA, iB,
                      ('energy', {'krylov': 'gmres', 'degree': 2,
                                  'maxiter': 3}),
                      {'k': 3}))

        cases.append((A.tobsr(blocksize=(2, 2)),
                      np.hstack((B, np.random.rand(B.shape[0], 1))),
                      ('energy', {'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
                      {'theta': 0.1}))

        for A, B, smooth, postfilter in cases:
            ml_nofilter = rootnode_solver(A, B=B, max_coarse=1, max_levels=2,
                                          smooth=smooth, keep=True)
            smooth[1]['postfilter'] = postfilter
            ml_filter = rootnode_solver(A, B=B, max_coarse=1, max_levels=2,
                                        smooth=smooth, keep=True)
            assert_equal(ml_nofilter.levels[0].P.nnz
                         > ml_filter.levels[0].P.nnz, True)

    def test_prefilter(self):
        """Check that using prefilter reduces NNZ in P."""
        np.random.seed(483333169)  # make tests repeatable
        cases = []

        # Simple, real-valued diffusion problems
        X = load_example('airfoil')
        A = X['A'].tocsr()
        B = X['B']

        cases.append((A, B,
                      ('energy', {'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
                      {'theta': 0.05}))

        cases.append((A, B,
                      ('energy', {'krylov': 'gmres', 'degree': 2,
                                  'maxiter': 3}),
                      {'k': 3}))

        cases.append((A.tobsr(blocksize=(2, 2)),
                      np.hstack((B, np.random.rand(B.shape[0], 1))),
                      ('energy', {'krylov': 'cg', 'degree': 2,
                                  'maxiter': 3}),
                      {'theta': 0.1}))

        # Simple, imaginary-valued problems
        iA = 1.0j * A
        iB = 1.0 + np.random.rand(iA.shape[0], 2)\
                 + 1.0j * (1.0 + np.random.rand(iA.shape[0], 2))

        cases.append((iA, B,
                      ('energy', {'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
                      {'theta': 0.05}))

        cases.append((iA, iB,
                      ('energy', {'krylov': 'gmres', 'degree': 2,
                                  'maxiter': 3}),
                      {'k': 3}))

        cases.append((A.tobsr(blocksize=(2, 2)),
                      np.hstack((B, np.random.rand(B.shape[0], 1))),
                      ('energy', {'krylov': 'cg', 'degree': 2, 'maxiter': 3}),
                      {'theta': 0.1}))

        for A, B, smooth, prefilter in cases:
            ml_nofilter = rootnode_solver(A, B=B, max_coarse=1, max_levels=2,
                                          smooth=smooth, keep=True)
            smooth[1]['prefilter'] = prefilter
            ml_filter = rootnode_solver(A, B=B, max_coarse=1, max_levels=2,
                                        smooth=smooth, keep=True)
            assert_equal(ml_nofilter.levels[0].P.nnz
                         > ml_filter.levels[0].P.nnz, True)

# class TestSatisfyConstaints(TestCase):
#    def test_scalar(self):
#
#        U = bsr_matrix([[1,2],[2,1]], blocksize=(1,1))
#        pattern = bsr_matrix([[1,1],[1,1]],blocksize=(1,1))
#        B = np.array(np.array([[1],[1]]))
#        BtBinv = [ np.array([[0.5]]), np.array([[0.5]]) ]
#        colindices = [ np.array([0,1]), np.array([0,1]) ]
#
#        U = satisfy_constraints(U, pattern, B, BtBinv, colindices)
#
#
#        assert_equal(U.toarray(), np.array([[0,1],[1,0]]))
#        assert_almost_equal(U*B, 0*U*B)
#
#    def test_block(self):
#        pattern = np.array([[1,1,0,0],
#                            [1,1,0,0],
#                            [1,1,1,1],
#                            [0,0,1,1],
#                            [0,0,1,1]])
#        U = np.array([[1,2,0,0],
#                   [4,3,0,0],
#                   [5,6,8,7],
#                   [0,0,4,1],
#                   [0,0,2,3]])
#
#        pattern = bsr_matrix(pattern, blocksize=(1,2))
#        U = bsr_matrix(U, blocksize=(1,2))
#        B = np.array([[1,1],
#                   [1,2],
#                   [1,3],
#                   [1,4]])
#        colindices = [np.array([0,1]), np.array([0,1]), np.array([0,1,2,3]),
#                      np.array([0,1]), np.array([0,1])]
#
#        BtBinv = zeros((5,2,2))
#        for i in range(5):
#            colindx = colindices[i]
#            if len(colindx) > 0:
#                Bi = np.array(B)[colindx,:]
#                BtBinv[i] = pinv(Bi.T.dot(Bi))
#
#        U = satisfy_constraints(U, pattern, B, BtBinv, colindices)
#        assert_almost_equal(U*B, 0*U*B)
