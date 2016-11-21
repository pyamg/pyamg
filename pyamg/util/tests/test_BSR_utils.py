from numpy import array, ravel
from scipy import mat
from scipy.sparse import bsr_matrix
from pyamg.util.BSR_utils import BSR_Get_Row, BSR_Row_WriteScalar,\
    BSR_Row_WriteVect

from numpy.testing import TestCase, assert_equal


class TestBSRUtils(TestCase):
    def test_BSR_Get_Row(self):
        indptr = array([0, 2, 3, 6])
        indices = array([0, 2, 2, 0, 1, 2])
        data = array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
        B = bsr_matrix((data, indices, indptr), shape=(6, 6))
        r, i = BSR_Get_Row(B, 2)
        assert_equal(r, mat(array([[3], [3]])))
        assert_equal(i, array([4, 5]))

    def test_BSR_Row_WriteScalar(self):
        indptr = array([0, 2, 3, 6])
        indices = array([0, 2, 2, 0, 1, 2])
        data = array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)

        indptr2 = array([0, 2, 3, 6])
        indices2 = array([0, 2, 2, 0, 1, 2])
        data2 = array([[[1, 1], [1, 1]],
                       [[2, 2], [2, 2]],
                       [[3, 3], [3, 3]],
                       [[4, 4], [22, 22]],
                       [[5, 5], [22, 22]],
                       [[6, 6], [22, 22]]])

        B2 = bsr_matrix((data2, indices2, indptr2), shape=(6, 6))
        B = bsr_matrix((data, indices, indptr), shape=(6, 6))
        BSR_Row_WriteScalar(B, 5, 22)
        diff = ravel((B - B2).data)
        assert_equal(diff.shape[0], 0)

    def test_BSR_Row_WriteVect(self):
        indptr = array([0, 2, 3, 6])
        indices = array([0, 2, 2, 0, 1, 2])
        data = array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)

        indptr2 = array([0, 2, 3, 6])
        indices2 = array([0, 2, 2, 0, 1, 2])
        data2 = array([[[1, 1], [1, 1]],
                       [[2, 2], [2, 2]],
                       [[3, 3], [3, 3]],
                       [[4, 4], [11, 22]],
                       [[5, 5], [33, 44]],
                       [[6, 6], [55, 66]]])

        B2 = bsr_matrix((data2, indices2, indptr2), shape=(6, 6))
        B = bsr_matrix((data, indices, indptr), shape=(6, 6))
        BSR_Row_WriteVect(B, 5, array([11, 22, 33, 44, 55, 66]))
        diff = ravel((B - B2).data)
        assert_equal(diff.shape[0], 0)
