import numpy as np
import scipy as sp

from scipy.sparse import csr_matrix, coo_matrix

from pyamg.gallery import poisson, load_example
from pyamg.strength import classical_strength_of_connection

from pyamg.classical import split
from pyamg.classical.classical import ruge_stuben_solver
from pyamg.classical.interpolate import direct_interpolation

from numpy.testing import TestCase, assert_equal, assert_almost_equal


class TestRugeStubenFunctions(TestCase):
    def setUp(self):
        self.cases = []

        # random matrices
        np.random.seed(0)
        for N in [2, 3, 5]:
            self.cases.append(csr_matrix(sp.rand(N, N)))

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

            assert(splitting.min() >= 0)     # could be all 1s
            assert_equal(splitting.max(), 1)

            S.data[:] = 1

            # check that all F-nodes are strongly connected to a C-node
            assert((splitting + S*splitting).min() > 0)

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

            assert(splitting.min() >= 0)     # could be all 1s
            assert_equal(splitting.max(), 1)

            S.data[:] = 1

            # check that all F-nodes are strongly connected to a C-node
            assert((splitting + S*splitting).min() > 0)

    def test_cljpc_splitting(self):
        for A in self.cases:
            S = classical_strength_of_connection(A, 0.0)

            splitting = split.CLJPc(S)

            assert(splitting.min() >= 0)     # could be all 1s
            assert_equal(splitting.max(), 1)

            S.data[:] = 1

            # check that all F-nodes are strongly connected to a C-node
            assert((splitting + S*splitting).min() > 0)

    def test_direct_interpolation(self):
        for A in self.cases:

            S = classical_strength_of_connection(A, 0.0)
            splitting = split.RS(S)

            result = direct_interpolation(A, S, splitting)
            expected = reference_direct_interpolation(A, S, splitting)

            assert_almost_equal(result.todense(), expected.todense())


class TestSolverPerformance(TestCase):
    def test_poisson(self):
        cases = []

        cases.append((500,))
        cases.append((250, 250))
        cases.append((25, 25, 25))

        for case in cases:
            A = poisson(case, format='csr')

            np.random.seed(0)  # make tests repeatable

            x = sp.rand(A.shape[0])
            b = A*sp.rand(A.shape[0])  # zeros_like(x)

            ml = ruge_stuben_solver(A, max_coarse=50)

            res = []
            x_sol = ml.solve(b, x0=x, maxiter=20, tol=1e-12,
                             residuals=res)
            del x_sol

            avg_convergence_ratio = (res[-1]/res[0])**(1.0/len(res))

            assert(avg_convergence_ratio < 0.20)

    def test_matrix_formats(self):

        # Do dense, csr, bsr and csc versions of A all yield the same solver
        A = poisson((7, 7), format='csr')
        cases = [A.tobsr(blocksize=(1, 1))]
        cases.append(A.tocsc())
        cases.append(A.todense())

        rs_old = ruge_stuben_solver(A, max_coarse=10)
        for AA in cases:
            rs_new = ruge_stuben_solver(AA, max_coarse=10)
            assert(np.abs(np.ravel(rs_old.levels[-1].A.todense() -
                          rs_new.levels[-1].A.todense())).max() < 0.01)
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

    map = np.concatenate(([0], np.cumsum(splitting)))
    P = csr_matrix((P.data, map[P.indices], P.indptr),
                   shape=(P.shape[0], map[-1]))

    return P
