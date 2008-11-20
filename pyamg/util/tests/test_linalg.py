from pyamg.testing import *

from numpy import matrix, array, diag, zeros, sqrt, abs, ravel
from scipy import rand, linalg, real, imag, mat, diag, random
from scipy.sparse import csr_matrix
from scipy.linalg import svd, eigvals

from pyamg.util.linalg import *

class TestLinalg(TestCase):
    def test_norm(self):
        cases = []

        cases.append(  4 )
        cases.append( -1 )
        cases.append( 2.5 )
        cases.append( 3 + 5j )
        cases.append( 7 - 2j )
        cases.append( [1 + 3j, 6] )
        cases.append( [1 + 3j, 6 - 2j] )

        for A in cases:
            assert_almost_equal( norm(A), linalg.norm(A) )

    def test_approximate_spectral_radius(self):
        cases = []

        cases.append( matrix([[-4]]) )
        cases.append( array([[-4]]) )
        
        cases.append( array([[2,0],[0,1]]) )
        cases.append( array([[-2,0],[0,1]]) )
      
        cases.append( array([[100,0,0],[0,101,0],[0,0,99]]) )
        
        for i in range(1,5):
            cases.append( rand(i,i) )
       
        # method should be almost exact for small matrices
        for A in cases:
            A = A.astype(float)
            Asp = csr_matrix(A)

            expected = max([linalg.norm(x) for x in linalg.eigvals(A)])
            assert_almost_equal( approximate_spectral_radius(A),   expected )
            assert_almost_equal( approximate_spectral_radius(Asp), expected )
        
        # try symmetric matrices
        for A in cases:
            A = A + A.transpose()
            A = A.astype(float) 
            Asp = csr_matrix(A)

            expected = max([linalg.norm(x) for x in linalg.eigvals(A)])
            assert_almost_equal( approximate_spectral_radius(A,symmetric=True),   expected )
            assert_almost_equal( approximate_spectral_radius(Asp,symmetric=True), expected )
      
        #TODO test larger matrices
    
    def test_infinity_norm(self):
        A = matrix([[-4]])
        assert_equal(infinity_norm(csr_matrix(A)),4)

        A = matrix([[1,0,-5],[-2,5,0]])
        assert_equal(infinity_norm(csr_matrix(A)),7)

        A = matrix([[0,1],[0,-5]])
        assert_equal(infinity_norm(csr_matrix(A)),5)

        A = matrix([[1.3,-4.7,0],[-2.23,5.5,0],[9,0,-2]])
        assert_equal(infinity_norm(csr_matrix(A)),11)


class TestComplexLinalg(TestCase):    
    def test_approximate_spectral_radius(self):
        cases = []

        cases.append( matrix([[-4-4.0j]]) )
        cases.append( array([[-4+8.2j]]) )
        
        cases.append( array([[2.0-2.9j,0],[0,1.5]]) )
        cases.append( array([[-2.0-2.4j,0],[0,1.21]]) )
      
        cases.append( array([[100+1.0j,0,0],[0,101-1.0j,0],[0,0,99+9.9j]]) )
        
        for i in range(1,6):
            cases.append( rand(i,i)+1.0j*rand(i,i) )
       
        # method should be almost exact for small matrices
        for A in cases:
            Asp = csr_matrix(A)
            e = linalg.eigvals(A)
            expected = max(abs(e))
            assert_almost_equal( approximate_spectral_radius(A),   expected )
            assert_almost_equal( approximate_spectral_radius(Asp), expected )
            
            AA = mat(A).H*mat(A)
            AAsp = csr_matrix(AA)
            e = linalg.eigvals(AA)
            expected = max(abs(e))
            assert_almost_equal( approximate_spectral_radius(AA, symmetric=True),   expected )
            assert_almost_equal( approximate_spectral_radius(AAsp, symmetric=True), expected )
 
    def test_infinity_norm(self):
        A = matrix([[-4-3.0j]])
        assert_equal(infinity_norm(csr_matrix(A)),5.0)

        A = matrix([[1,0,4.0-3.0j],[-2,5,0]])
        assert_equal(infinity_norm(csr_matrix(A)),7)

        A = matrix([[0,1],[0,-4.0+3.0j]])
        assert_equal(infinity_norm(csr_matrix(A)),5.0)


    def test_cond(self):
        # make tests repeatable
        random.seed(0)

        # Should be exact
        cases = []
        A = mat(array([2.14]))
        cases.append(A)
        A = mat(array([2.14j]))
        cases.append(A)
        A = mat(array([-1.2 + 2.14j]))
        cases.append(A)
        for i in range(1,6):
            A = mat(rand(i,i))
            cases.append(A)
            cases.append(1.0j*A)
            A = mat(A + 1.0j*rand(i,i))
            cases.append(A)

        for A in cases:
            U, Sigma, Vh = svd(A)
            exact = max(Sigma)/min(Sigma)
            c = cond(A)
            assert_almost_equal(exact, c)

    def test_condest(self):
        # make tests repeatable
        random.seed(0)
        
        # Should be exact for small matrices
        cases = []
        A = mat(array([2.14]))
        cases.append(A)
        A = mat(array([2.14j]))
        cases.append(A)
        A = mat(array([-1.2 + 2.14j]))
        cases.append(A)
        for i in range(1,6):
            A = mat(rand(i,i))
            cases.append(A)
            cases.append(1.0j*A)
            A = mat(A + 1.0j*rand(i,i))
            cases.append(A)

        for A in cases:
            eigs = eigvals(A)
            exact = max(abs(eigs))/min(abs(eigs))
            c = condest(A)
            assert_almost_equal(exact, c)

    def test_issymm(self):
        # make tests repeatable
        random.seed(0)
        
        # 1x1
        A = mat(rand(1,1))
        assert_equal(issymm(A), 0.0)
        A = mat(1.0j*rand(1,1))
        assert_equal(issymm(A), max(abs(ravel(A - A.H))))
        
        # 2x2
        A = mat(array([[1.0, 0.0], [2.0, 1.0]]))
        assert_equal(issymm(A), max(abs(ravel(A-A.H))) )
        Ai = 1.0j*A
        assert_equal(issymm(Ai), max(abs(ravel(Ai-Ai.H))))
        A = A + Ai
        assert_equal(issymm(A), max(abs(ravel(A-A.H))))
        assert_equal(issymm(A+A.H), 0.0)
        
        # 3x3
        A = mat(rand(3,3))
        assert_equal(issymm(A), max(abs(ravel(A-A.H))) )
        Ai = mat(1.0j*rand(3,3))
        assert_equal(issymm(Ai), max(abs(ravel(Ai-Ai.H))))
        A = A + Ai
        assert_equal(issymm(A), max(abs(ravel(A-A.H))))
        assert_equal(issymm(A+A.H), 0.0)


