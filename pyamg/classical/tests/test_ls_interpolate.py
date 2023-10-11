import numpy as np
from scipy.sparse import csr_matrix
from pyamg.gallery import poisson, load_example
from pyamg.strength import classical_strength_of_connection 
from pyamg.classical.split import RS 

from numpy.testing import TestCase

############## UPDATE TO BE TEST CASE FOR LS INTERPOLATION #######################
'''
from pyamg.classical.interpolate import ls_interpolate
class TestLSInterpolate(TestCase):
    def setUp(self):
        self.cases = []
        #

        # Random matrices, cases 0-2
        np.random.seed(0)
        for N in [2, 3, 5]:
            self.cases.append(csr_matrix(np.random.rand(N, N)))

        # Poisson problems in 1D, cases 3-9
        for N in [2, 3, 5, 7, 10, 11, 19]:
            self.cases.append(poisson((N,), format='csr'))

        # Poisson problems in 2D, cases 10-15
        for N in [2, 3, 5, 7, 10, 11]:
            self.cases.append(poisson((N, N), format='csr'))

        for name in ['knot', 'airfoil', 'bar']:
            ex = load_example(name)
            self.cases.append(ex['A'].tocsr())

    def test_ls_interpolate(self):
        A = self.cases[6]
        num_tvs = 5 

        # 1d-tests, should be alternating aggregates
        #       (n-1)/2 < = sum <= (n+1)/2.
        # Test auto thetacs and set thetacs values
        for i in range(3, 10):
            A = self.cases[i]
            V = np.random.rand(A.shape[0], num_tvs)

        # 2d-tests. CR is a little more picky with parameters and relaxation
        # type in 2d. Can still bound above by (n+1)/2.
        # Need looser lower bound,
        # say (n+1)/4.
        for i in range(10, 15):
            A = self.cases[i]
            V = np.random.rand(A.shape[0], num_tvs)
'''
