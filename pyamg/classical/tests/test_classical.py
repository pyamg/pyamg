"""Test classical AMG."""
import warnings

import numpy as np

from numpy.testing import TestCase, assert_equal, assert_almost_equal, \
    assert_array_almost_equal

from scipy.sparse import csr_matrix, coo_matrix, SparseEfficiencyWarning

from pyamg.gallery import poisson, load_example
from pyamg.strength import classical_strength_of_connection

from pyamg.classical import split
from pyamg.classical.classical import ruge_stuben_solver
from pyamg.classical.interpolate import direct_interpolation, \
    classical_interpolation


class TestRugeStubenFunctions(TestCase):
    def setUp(self):
        self.cases = []

        # random matrices
        np.random.seed(0)
        for N in [2, 3, 5]:
            self.cases.append(csr_matrix(np.random.rand(N, N)))

        # Poisson problems in 1D and 2D
        for N in [2, 3, 5, 7, 10, 11, 19]:
            self.cases.append(poisson((N,), format='csr'))
        for N in [2, 3, 5, 7, 10, 11]:
            self.cases.append(poisson((N, N), format='csr'))

        for name in ['knot', 'airfoil', 'bar']:
            ex = load_example(name)
            self.cases.append(ex['A'].tocsr())

    def test_RS_splitting(self):
        for A in self.cases:
            S = classical_strength_of_connection(A, 0.0)

            splitting = split.RS(S)

            assert splitting.min() >= 0     # could be all 1s
            assert_equal(splitting.max(), 1)

            S.data[:] = 1

            # check that all F-nodes are strongly connected to a C-node
            assert (splitting + S*splitting).min() > 0

            # THIS IS NOT STRICTLY ENFORCED!
            # check that all strong connections S[i, j] satisfy either:
            # (0) i is a C-node
            # (1) j is a C-node
            # (2) k is a C-node and both i and j are strongly connected to k
            #
            # X = S.tocoo()

            # remove C->F edges (i.e. S[i, j] where (0) holds)
            # mask = splitting[X.row] == 0
            # X.row  = X.row[mask]
            # X.col  = X.col[mask]
            # X.data = X.data[mask]

            # remove F->C edges (i.e. S[i, j] where (1) holds)
            # mask = splitting[X.col] == 0
            # X.row  = X.row[mask]
            # X.col  = X.col[mask]
            # X.data = X.data[mask]

            # X now consists of strong F->F edges only
            #
            # (S * S.T)[i, j] is the # of C nodes on which both i and j
            # strongly depend (i.e. the number of k's where (2) holds)
            # Y = (S*S.T) - X
            # assert(Y.nnz == 0 or Y.data.min() > 0)

    def test_cljp_splitting(self):
        for A in self.cases:
            S = classical_strength_of_connection(A, 0.0)

            splitting = split.CLJP(S)

            assert splitting.min() >= 0     # could be all 1s
            assert_equal(splitting.max(), 1)

            S.data[:] = 1

            # check that all F-nodes are strongly connected to a C-node
            assert (splitting + S*splitting).min() > 0

    def test_cljpc_splitting(self):
        for A in self.cases:
            S = classical_strength_of_connection(A, 0.0)

            splitting = split.CLJPc(S)

            assert splitting.min() >= 0     # could be all 1s
            assert_equal(splitting.max(), 1)

            S.data[:] = 1

            # check that all F-nodes are strongly connected to a C-node
            assert (splitting + S*splitting).min() > 0

    def test_direct_interpolation(self):
        for A in self.cases:

            S = classical_strength_of_connection(A, 0.0)
            splitting = split.RS(S)

            result = direct_interpolation(A, S, splitting)
            expected = reference_direct_interpolation(A, S, splitting)

            assert_almost_equal(result.toarray(), expected.toarray())

    def test_classical_interpolation(self):
        for A in self.cases:
            # the reference code is very slow, so just take a small block of A
            mini = min(100, A.shape[0])
            A = ((A.tocsr()[0:mini, :])[:, 0:mini]).tocsr()

            S = classical_strength_of_connection(A, 0.0)
            splitting = split.RS(S, second_pass=True)

            result = classical_interpolation(A, S, splitting, modified=False)
            expected = reference_classical_interpolation(A, S, splitting)

            # elasticity produces large entries, so normalize
            Diff = result - expected
            Diff.data = abs(Diff.data)
            expected.data = 1./abs(expected.data)
            Diff = Diff.multiply(expected)
            Diff.data[Diff.data < 1e-7] = 0.0
            Diff.eliminate_zeros()
            assert (Diff.nnz == 0)

    def test_remove_strong_FF_connections(self):
        from pyamg import amg_core
        # test removing an F-F connection without any strong C in between (4--2),
        # while keeping the F-f connection (4--6) which does have a strong C in between
        C = csr_matrix(np.array([[2., 1., 0., 0., 0., 0.],
                                 [1., 2., 1., 1., 0., 0.],
                                 [0., 1., 2., 0., 0., 0.],
                                 [0., 1., 0., 2., 1., 1.],
                                 [0., 0., 0., 1., 2., 1.],
                                 [0., 0., 0., 1., 1., 2.]]))
        splitting = np.array([1, 0, 1, 0, 1, 0], dtype=C.indices.dtype)
        amg_core.remove_strong_FF_connections(6, C.indptr, C.indices,
                                              C.data, splitting)

        exact = np.array([[2., 1., 0., 0., 0., 0.],
                          [1., 2., 1., 0., 0., 0.],
                          [0., 1., 2., 0., 0., 0.],
                          [0., 0., 0., 2., 1., 1.],
                          [0., 0., 0., 1., 2., 1.],
                          [0., 0., 0., 1., 1., 2.]])
        assert_array_almost_equal(C.toarray(), exact)


class TestSolverPerformance(TestCase):
    def test_poisson(self):
        cases = []

        cases.append((500,))
        cases.append((250, 250))
        cases.append((25, 25, 25))

        for case in cases:
            A = poisson(case, format='csr')

            for interp in ['direct',
                           ('classical', {'modified': False}),
                           ('classical', {'modified': True})]:

                np.random.seed(0)  # make tests repeatable
                x = np.random.rand(A.shape[0])
                b = A*np.random.rand(A.shape[0])  # zeros_like(x)

                ml = ruge_stuben_solver(A, interpolation=interp, max_coarse=50)

                res = []
                x_sol = ml.solve(b, x0=x, maxiter=20, tol=1e-12,
                                 residuals=res)
                del x_sol

                avg_convergence_ratio = (res[-1]/res[0])**(1.0/len(res))
                assert (avg_convergence_ratio < 0.20)

    def test_matrix_formats(self):
        warnings.simplefilter('ignore', SparseEfficiencyWarning)

        # Do dense, csr, bsr and csc versions of A all yield the same solver
        A = poisson((7, 7), format='csr')
        cases = [A.tobsr(blocksize=(1, 1))]
        cases.append(A.tocsc())
        cases.append(A.toarray())

        rs_old = ruge_stuben_solver(A, max_coarse=10)
        for AA in cases:
            rs_new = ruge_stuben_solver(AA, max_coarse=10)
            Ac_old = rs_old.levels[-1].A.toarray()
            Ac_new = rs_new.levels[-1].A.toarray()
            assert np.abs(np.ravel(Ac_old - Ac_new)).max() < 0.01
            rs_old = rs_new


#   reference implementations for unittests  #
def reference_direct_interpolation(A, S, splitting):

    # Interpolation weights are computed based on entries in A, but subject to
    # the sparsity pattern of C.  So, we copy the entries of A into the
    # sparsity pattern of C.
    S = S.copy()
    S.data[:] = 1.0
    S = S.multiply(A)

    A = coo_matrix(A)
    S = coo_matrix(S)

    # remove diagonals
    mask = S.row != S.col
    S.row = S.row[mask]
    S.col = S.col[mask]
    S.data = S.data[mask]

    # strong C points
    c_mask = splitting[S.col] == 1
    C_s = coo_matrix((S.data[c_mask], (S.row[c_mask], S.col[c_mask])),
                     shape=S.shape)

    # strong F points
    # f_mask = ~c_mask
    # F_s = coo_matrix((S.data[f_mask], (S.row[f_mask], S.col[f_mask])),
    #                shape=S.shape)

    # split A in to + and -
    mask = (A.data > 0) & (A.row != A.col)
    A_pos = coo_matrix((A.data[mask], (A.row[mask], A.col[mask])),
                       shape=A.shape)
    mask = (A.data < 0) & (A.row != A.col)
    A_neg = coo_matrix((A.data[mask], (A.row[mask], A.col[mask])),
                       shape=A.shape)

    # split C_S in to + and -
    mask = C_s.data > 0
    C_s_pos = coo_matrix((C_s.data[mask], (C_s.row[mask], C_s.col[mask])),
                         shape=A.shape)
    mask = ~mask
    C_s_neg = coo_matrix((C_s.data[mask], (C_s.row[mask], C_s.col[mask])),
                         shape=A.shape)

    sum_strong_pos = np.ravel(C_s_pos.sum(axis=1))
    sum_strong_neg = np.ravel(C_s_neg.sum(axis=1))

    sum_all_pos = np.ravel(A_pos.sum(axis=1))
    sum_all_neg = np.ravel(A_neg.sum(axis=1))

    diag = A.diagonal()

    mask = (sum_strong_neg != 0.0)
    alpha = np.zeros_like(sum_all_neg)
    alpha[mask] = sum_all_neg[mask] / sum_strong_neg[mask]

    mask = (sum_strong_pos != 0.0)
    beta = np.zeros_like(sum_all_pos)
    beta[mask] = sum_all_pos[mask] / sum_strong_pos[mask]

    mask = sum_strong_pos == 0
    diag[mask] += sum_all_pos[mask]
    beta[mask] = 0

    C_s_neg.data *= -alpha[C_s_neg.row]/diag[C_s_neg.row]
    C_s_pos.data *= -beta[C_s_pos.row]/diag[C_s_pos.row]

    C_rows = splitting.nonzero()[0]
    C_inject = coo_matrix((np.ones(sum(splitting)), (C_rows, C_rows)),
                          shape=A.shape)

    P = C_s_neg.tocsr() + C_s_pos.tocsr() + C_inject.tocsr()

    splitting_map = np.concatenate(([0], np.cumsum(splitting)))
    P = csr_matrix((P.data, splitting_map[P.indices], P.indptr),
                   shape=(P.shape[0], splitting_map[-1]))

    return P


# strength has zero diagonal...?
def reference_classical_interpolation(A, S, splitting):

    # this routine only tests the computation of the "weights" the computation
    # of the sparsity pattern is the same as for direct interpolation, and is
    # tested through the reference_direct_interpolation routine.
    from pyamg import amg_core
    S = S.copy()
    S.data[:] = 1.0
    S = S.multiply(A)
    Pp = np.empty_like(A.indptr)
    amg_core.rs_direct_interpolation_pass1(A.shape[0], S.indptr, S.indices,
                                           splitting, Pp)
    nnz = Pp[-1]
    Pj = np.empty(nnz, dtype=Pp.dtype)
    Px = np.empty(nnz, dtype=A.dtype)
    SD = S.diagonal()
    F_NODE = 0
    C_NODE = 1

    # Now, we implement the second pass in Python to double check the C++ code
    for i in range(A.shape[0]):
        # If node is is a C-point, do injection
        if (splitting[i] == C_NODE):
            Pj[Pp[i]] = i
            Px[Pp[i]] = 1

        # Else compute classical interpolation weight
        else:
            rowstartA = A.indptr[i]
            rowendA = A.indptr[i+1]
            rowstartS = S.indptr[i]
            rowendS = S.indptr[i+1]

            # Denominator = a_ii + sum_{m in weak connections} a_im
            denominator = sum(A.data[rowstartA:rowendA])
            denominator -= sum(S.data[rowstartS:rowendS])
            denominator += SD[i]

            # Compute interpolation weights from strongly connected C-points
            nnz = Pp[i]
            for jj in range(rowstartS, rowendS):
                Sj = S.indices[jj]
                if (splitting[Sj] == C_NODE) and (Sj != i):
                    Pj[nnz] = Sj
                    numerator = S.data[jj]
                    for kk in range(rowstartS, rowendS):
                        Sk = S.indices[kk]
                        if (splitting[Sk] == F_NODE) and (Sk != i):
                            inner_denominator = 0.0
                            for ll in range(rowstartS, rowendS):
                                Sl = S.indices[ll]
                                if (splitting[Sl] == C_NODE) and (Sl != i):
                                    for search_ind in range(A.indptr[Sk], A.indptr[Sk+1]):
                                        if (A.indices[search_ind] == Sl):
                                            inner_denominator += A.data[search_ind]

                            for search_ind in range(A.indptr[Sk], A.indptr[Sk+1]):
                                if (A.indices[search_ind] == Sj) and \
                                   (inner_denominator != 0.0):
                                    numerator += \
                                        (S.data[kk]*A.data[search_ind]/inner_denominator)

                    Px[nnz] = -numerator/denominator
                    nnz += 1

    reorder = np.zeros((A.shape[0],))
    cumulative = 0
    for i in range(A.shape[0]):
        reorder[i] = cumulative
        cumulative += splitting[i]

    for i in range(Pp[A.shape[0]]):
        Pj[i] = reorder[Pj[i]]

    return csr_matrix((Px, Pj, Pp))
