import numpy as np
from scipy import rand, real, imag, arange
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr, spdiags,\
    coo_matrix
import scipy.sparse
from scipy.linalg import pinv

from pyamg.gallery import poisson, linear_elasticity, load_example,\
    stencil_grid
from pyamg.strength import classical_strength_of_connection,\
    symmetric_strength_of_connection, evolution_strength_of_connection,\
    distance_strength_of_connection, algebraic_distance, affinity_distance


from numpy.testing import TestCase, assert_equal, assert_array_almost_equal,\
    assert_array_equal, assert_almost_equal



class TestStrengthComplexity(TestCase):

    self.cases = []

    # Poisson problems in 1D and 2D
    for N in [2, 3, 5, 7, 10, 11, 19]:
        self.cases.append(poisson((N,), format='csr'))
    for N in [2, 3, 7, 9]:
        self.cases.append(poisson((N, N), format='csr'))

    for name in ['knot', 'airfoil', 'bar']:
        ex = load_example(name)
        self.cases.append(ex['A'].tocsr())

    def test_classical(self):

        for A in self.cases:

            # theta = 0, no entries dropped
            theta = 0.0
            cost = [0]
            classical_strength_of_connection(A, theta, cost)
            assert_almost_equal(cost[0], 3.0)

            for theta in [0.1,0.25,0.5,0.75,0.9]:
                cost = [0]
                classical_strength_of_connection(A, theta, cost)
                assert(cost[0] <= 3.0)
                assert(cost[0] > 0.0)

            # theta = 1, only largest entries in each row remain
            theta = 1.0
            cost = [0]
            classical_strength_of_connection(A, theta, cost)
            est = 2.0 + float(A.shape[0]) / A.nnz
            assert_almost_equal(cost[0],est)

    def test_symmetric(self):

        for A in self.cases:

            # theta = 0, no entries dropped
            theta = 0.0
            cost = [0]
            symmetric_strength_of_connection(A, theta, cost)
            assert_almost_equal(cost[0], 3.5)

            for theta in [0.1,0.25,0.5,0.75,0.9]:
                cost = [0]
                symmetric_strength_of_connection(A, theta, cost)
                assert(cost[0] <= 3.5)
                assert(cost[0] > 0.0)

            # theta = 1, only largest entries in each row remain
            theta = 1.0
            cost = [0]
            symmetric_strength_of_connection(A, theta, cost)
            est = 1.5 + 2.0*float(A.shape[0]) / A.nnz
            assert_almost_equal(cost[0],est)

    def test_evolution(self):


    def test_distance(self):
        data = load_example('airfoil')
        cases = []
        cases.append((data['A'].tocsr(), data['vertices']))
       
        for (A, V) in cases:
            dim = V.shape[1]
            for theta in [1.5, 2.0, 2.5]:
                cost = [0]
                lower_bound = 3*dim + float(A.shape[0]) / A.nnz
                upper_bound = 3*dim + 3
                distance_soc(A, V, theta=theta, relative_drop=True, cost=cost)
                assert(cost[0] >= lower_bound)
                assert(cost[0] <= upper_bound)

        for (A, V) in cases:
            for theta in [0.5, 1.0, 1.5]:
                cost = [0]
                lower_bound = 3*dim + float(A.shape[0]) / A.nnz
                upper_bound = 3*dim + 3
                distance_soc(A, V, theta=theta, relative_drop=False, cost=cost)
                assert(cost[0] >= lower_bound)
                assert(cost[0] <= upper_bound)

    def test_affinity(self):


    def test_algebraic(self):



class TestSmoothComplexity(TestCase):

    self.cases = []

    # Poisson problems in 1D and 2D
    for N in [10, 11, 19, 26]:
        A = poisson((N,), format='csr')
        tempAgg = standard_aggregation(A)
        T, B = fit_candidates(tempAgg, np.ones((A.shape[0],)) )
        self.cases.append( {'A': A, 'T': T, 'B': B} )
    for N in [5, 7, 9]:
        A = poisson((N,N), format='csr')
        tempAgg = standard_aggregation(A)
        T, B = fit_candidates(tempAgg, np.ones((A.shape[0],)) )
        self.cases.append( {'A': A, 'T': T, 'B': B} )

    for name in ['knot', 'airfoil', 'bar']:
        A = ex['A'].tocsr()
        tempAgg = standard_aggregation(A)
        T, B = fit_candidates(tempAgg, np.ones((A.shape[0],)) )
        self.cases.append( {'A': A, 'T': T, 'B': B} )

    def test_richardson(self):

        for mats in self.cases:
            for degree in [0,1,2]:
                cost = [0]
                P = richardson_prolongation_smoother(mats['A'], mats['T'],
                                                    degree=degree, cost=cost)
                assert(cost[0] >= 15)    # 15 WUs to find spectral radius
                assert(cost[0] <= (15 + degree*float(P.nnz)/A.nnz) )

    def test_jacobi(self):

        for mats in self.cases:
            for degree in [0,1,2]:

                # diagonal weighting
                cost = [0]
                P = jacobi_prolongation_smoothing(mats['A'], mats['T'], mats['A'],
                                                    mats['B'], weighting='diagonal',
                                                    degree=degree, cost=cost)
                assert(cost[0] >= 17)    # 17 WUs to find spectral radius and scale
                assert(cost[0] <= (17 + degree*float(P.nnz)/A.nnz) )

                # local weighting
                cost = [0]
                P = jacobi_prolongation_smoothing(mats['A'], mats['T'], mats['A'],
                                                    mats['B'], weighting='local',
                                                    degree=degree, cost=cost)
                assert(cost[0] >= 3)    # 3 WUs to get Gershgorin radius and scale
                assert(cost[0] <= (3 + degree*float(P.nnz)/A.nnz) )

                # local weighting and filtering
                cost = [0]
                P = jacobi_prolongation_smoothing(mats['A'], mats['T'], mats['A'],
                                                    mats['B'], weighting='local',
                                                    filter=True, degree=degree,
                                                    cost=cost)
                assert(cost[0] >= 3)    # 3 WUs to get Gershgorin radius and scale
                # Approximate upper bound on WUs per smoothing, with filtering
                assert(cost[0] <= (3 + 7*degree*float(P.nnz)/A.nnz) )


    def test_energy(self):



class TestMethodsComplexity(TestCase):

    def test_sa(self):


    def test_classical(self):


    def test_rootnode(self):


    