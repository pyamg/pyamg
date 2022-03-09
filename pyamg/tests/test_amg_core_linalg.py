"""Test amg_core.linalg functions."""
import numpy as np
from scipy import sparse
from pyamg import amg_core


def test_real():
    A = np.array([[1.,  2.,  0.5,  0.0],
                  [1.,  2.,  1.5,  3.0],
                  [0.,  4.,  4.0,  2.1],
                  [-5.,  2.,  4.0, 10.0]])
    A0 = A.copy()

    # no lumping, threhold 0.0
    A = sparse.csr_matrix(A0.copy())
    amg_core.linalg.filter_matrix_rows(4, 0.0, A.indptr, A.indices, A.data, 0)
    np.testing.assert_array_equal(A.toarray(), A0)

    # no lumping, threhold 1.0
    A = sparse.csr_matrix(A0.copy())
    amg_core.linalg.filter_matrix_rows(4, 1.0, A.indptr, A.indices, A.data, 0)
    B = np.array([[1.,  2.,  0.0,  0.],
                  [0.,  2.,  0.0,  3.],
                  [0.,  4.,  4.0,  0.],
                  [0.,  0.,  0.0, 10.]])
    np.testing.assert_array_equal(A.toarray(), B)

    # lumping, threhold 1.0
    A = sparse.csr_matrix(A0.copy())
    amg_core.linalg.filter_matrix_rows(4, 1.0, A.indptr, A.indices, A.data, True)
    B = np.array([[1.5,  2.0,  0.0,  0.0],
                  [0.0,  4.5,  0.0,  3.0],
                  [0.0,  4.0,  6.1,  0.0],
                  [0.0,  0.0,  0.0, 11.0]])
    np.testing.assert_array_equal(A.toarray(), B)


def test_imag():
    A = np.array([[1.,  2.,  0.5,  0.0],
                  [1.,  2.,  1.5,  3.0],
                  [0.,  4.,  4.0,  2.1],
                  [-5.,  2.,  4.0, 10.0]]) * 1j
    A0 = A.copy()

    # no lumping, threhold 0.0
    A = sparse.csr_matrix(A0.copy())
    amg_core.linalg.filter_matrix_rows(4, 0.0, A.indptr, A.indices, A.data, 0)
    np.testing.assert_array_equal(A.toarray(), A0)

    # no lumping, threhold 1.0
    A = sparse.csr_matrix(A0.copy())
    amg_core.linalg.filter_matrix_rows(4, 1.0, A.indptr, A.indices, A.data, 0)
    B = np.array([[1.,  2.,  0.0,  0.0],
                  [0.,  2.,  0.0,  3.0],
                  [0.,  4.,  4.0,  0.0],
                  [0.,  0.,  0.0, 10.0]]) * 1j
    np.testing.assert_array_equal(A.toarray(), B)

    # lumping, threhold 1.0
    A = sparse.csr_matrix(A0.copy())
    amg_core.linalg.filter_matrix_rows(4, 1.0, A.indptr, A.indices, A.data, True)
    B = np.array([[1.5,  2.0,  0.0,  0.0],
                  [0.0,  4.5,  0.0,  3.0],
                  [0.0,  4.0,  6.1,  0.0],
                  [0.0,  0.0,  0.0, 11.0]]) * 1j
    np.testing.assert_array_equal(A.toarray(), B)


def test_complex():
    A = np.array([[1. + 3.0j,  2. + 2.0j,  0.5 + 1.0j,  4.1 + 0.0j],
                  [1. + 0.0j,  2. + 0.0j,  1.5 + 0.0j,  3.0 + 0.0j],
                  [0. + 0.0j,  4. + 0.0j,  4.0 + 0.0j,  2.1 + 2.0j],
                  [-5. + 2.0j,  2. + 0.0j,  5.0 + 9.0j, 10.0 + 0.0j]])
    A0 = A.copy()

    # no lumping, threhold 0.0
    A = sparse.csr_matrix(A0.copy())
    amg_core.linalg.filter_matrix_rows(4, 0.0, A.indptr, A.indices, A.data, 0)
    np.testing.assert_array_equal(A.toarray(), A0)

    # no lumping, threhold 1.0
    A = sparse.csr_matrix(A0.copy())
    amg_core.linalg.filter_matrix_rows(4, 1.0, A.indptr, A.indices, A.data, 0)
    B = np.array([[1. + 3.0j,  0. + 0.0j,  0.0 + 0.0j,  4.1 + 0.0j],
                  [0. + 0.0j,  2. + 0.0j,  0.0 + 0.0j,  3.0 + 0.0j],
                  [0. + 0.0j,  4. + 0.0j,  4.0 + 0.0j,  0.0 + 0.0j],
                  [0. + 0.0j,  0. + 0.0j,  5.0 + 9.0j, 10.0 + 0.0j]])
    np.testing.assert_array_equal(A.toarray(), B)

    # lumping, threhold 1.0
    A = sparse.csr_matrix(A0.copy())
    amg_core.linalg.filter_matrix_rows(4, 1.0, A.indptr, A.indices, A.data, True)
    B = np.array([[3.5 + 6.0j,  0.0 + 0.0j,  0.0 + 0.0j,  4.1 + 0.0j],
                  [0.0 + 0.0j,  4.5 + 0.0j,  0.0 + 0.0j,  3.0 + 0.0j],
                  [0.0 + 0.0j,  4.0 + 0.0j,  6.1 + 2.0j,  0.0 + 0.0j],
                  [0.0 + 0.0j,  0.0 + 0.0j,  5.0 + 9.0j,  7.0 + 2.0j]])
    np.testing.assert_array_equal(A.toarray(), B)
