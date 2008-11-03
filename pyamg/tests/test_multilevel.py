from pyamg.testing import *

from numpy import matrix, array, diag, arange
from scipy import rand
from scipy.linalg import norm
from scipy.sparse import csr_matrix

from pyamg.gallery import poisson
from pyamg.multilevel import *

class TestMultilevel(TestCase):
    def test_coarse_grid_solver(self):
        cases = []

        cases.append( csr_matrix(diag(arange(1,5))) )
        cases.append( poisson( (4,),  format='csr') )
        cases.append( poisson( (4,4), format='csr') )
       
        # method should be almost exact for small matrices
        for A in cases:
            for solver in ['splu','pinv','pinv2','lu','cholesky','cg']:
                s = coarse_grid_solver(solver)

                b = arange(A.shape[0],dtype=A.dtype)

                x = s(A,b)
                assert_almost_equal( A*x, b)

                # subsequent calls use cached data
                x = s(A,b)  
                assert_almost_equal( A*x, b)

    def test_aspreconditioner(self):
        from pyamg import smoothed_aggregation_solver
        from scipy.sparse.linalg import cg

        A = poisson((100,100), format='csr')
        b = rand(A.shape[0])

        ml = smoothed_aggregation_solver(A)

        for cycle in ['V','W','F']:
            M = ml.aspreconditioner(cycle='V')
            x,info = cg(A, b, tol=1e-8, maxiter=30, M=M)
            assert( norm(b - A*x) < 1e-8*norm(b) )
