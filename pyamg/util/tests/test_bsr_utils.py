"""Test BSR functions."""
import numpy as np
from scipy.sparse import bsr_matrix
from pyamg.util.bsr_utils import bsr_getrow, bsr_row_setscalar, bsr_row_setvector

from numpy.testing import TestCase, assert_equal


class TestBSRUtils(TestCase):
    def test_bsr_getrow(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
        B = bsr_matrix((data, indices, indptr), shape=(6, 6))
        r, i = bsr_getrow(B, 2)
        assert_equal(r, np.array([[3], [3]]))
        assert_equal(i, np.array([4, 5]))

    def test_bsr_row_setscalar(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)

        indptr2 = np.array([0, 2, 3, 6])
        indices2 = np.array([0, 2, 2, 0, 1, 2])
        data2 = np.array([[[1, 1], [1, 1]],
                          [[2, 2], [2, 2]],
                          [[3, 3], [3, 3]],
                          [[4, 4], [22, 22]],
                          [[5, 5], [22, 22]],
                          [[6, 6], [22, 22]]])

        B2 = bsr_matrix((data2, indices2, indptr2), shape=(6, 6))
        B = bsr_matrix((data, indices, indptr), shape=(6, 6))
        bsr_row_setscalar(B, 5, 22)
        diff = np.ravel((B - B2).data)
        assert_equal(diff.shape[0], 0)

    def test_bsr_row_setvector(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)

        indptr2 = np.array([0, 2, 3, 6])
        indices2 = np.array([0, 2, 2, 0, 1, 2])
        data2 = np.array([[[1, 1], [1, 1]],
                          [[2, 2], [2, 2]],
                          [[3, 3], [3, 3]],
                          [[4, 4], [11, 22]],
                          [[5, 5], [33, 44]],
                          [[6, 6], [55, 66]]])

        B2 = bsr_matrix((data2, indices2, indptr2), shape=(6, 6))
        B = bsr_matrix((data, indices, indptr), shape=(6, 6))
        bsr_row_setvector(B, 5, np.array([11, 22, 33, 44, 55, 66]))
        diff = np.ravel((B - B2).data)
        assert_equal(diff.shape[0], 0)
