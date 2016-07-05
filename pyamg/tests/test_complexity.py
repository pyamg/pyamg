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
    assert_array_equal



class TestStrengthComplexity(TestCase):


    def test_classical(self):


    def test_symmetric(self):


    def test_evolution(self):


    def test_distance(self):


    def test_affinity(self):


    def test_algebraic(self):




class TestSmoothComplexity(TestCase):

    def test_richardson(self):


    def test_jacobi(self):


    def test_energy(self):



class TestMethodsComplexity(TestCase):

    def test_sa(self):


    def test_classical(self):


    def test_rootnode(self):


    