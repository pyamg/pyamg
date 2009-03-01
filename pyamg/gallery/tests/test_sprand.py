from pyamg.testing import *

from pyamg.gallery.sprand import *

class TestSprand(TestCase):
    def test_10x20(self):
        A = sprand(10, 20, 1.0)
        assert_equal(A.shape, (10,20))
        assert(A.data.max() <= 1.0)
        assert(A.data.min() >= 0.0)

        A = sprand(10, 20, 0.0)
        assert_equal(A.shape, (10,20))
        assert_equal(A.nnz, 0)


#class TestSprandSPD(TestCase):
#    def test_20x20(self):
#        from scipy.linalg import eigvals
#
#        A = sprand_spd(20, 1.0)
#        assert_equal(A.shape, (20, 20))
#        assert_equal((A - A.T).nnz, 0)
#        assert(eigvals(A.todense()).min() > 0)
#
#        A = sprand_spd(20, 0.0)
#        assert_equal(A.shape, (20, 20))
#        assert_equal(A.nnz, 0)

