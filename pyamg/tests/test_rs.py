from scipy.testing import *

import numpy
from scipy import rand
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix

from pyamg.gallery import poisson

from pyamg.rs import * 


class TestRugeStubenFunctions(TestCase):
    def setUp(self):
        self.cases = []

        # random matrices
        numpy.random.seed(0)
        for N in [2,3,5]:
            self.cases.append( csr_matrix(rand(N,N)) )

        # poisson problems in 1D and 2D
        for N in [2,3,5,7,10,11,19]:
            self.cases.append( poisson( (N,), format='csr') )
        for N in [2,3,5,7,10,11]:
            self.cases.append( poisson( (N,N), format='csr') )
    
    def test_rs_strong_connections(self):
        for theta in [ 0.0, 0.05, 0.25, 0.50, 0.90 ]:
            for A in self.cases:
                result   = rs_strong_connections( A, theta )
                expected = reference_rs_strong_connections( A, theta )
                assert_equal( result.nnz, expected.nnz )
                assert_equal( result.todense(), expected.todense() )

    def test_rs_cf_splitting(self):
        
        for A in self.cases:
            S = rs_strong_connections( A, 0.0 )

            splitting = rs_cf_splitting( S )

            assert( splitting.min() >= 0 )     #could be all 1s
            assert_equal( splitting.max(), 1 ) 

            S.data[:] = 1

            # check that all F-nodes are strongly connected to a C-node
            assert( (splitting + S*splitting).min() > 0 )

            # check that all strong connections S[i,j] satisfy either:
            # (0) i is a C-node
            # (1) j is a C-node
            # (2) k is a C-node and both i and j are are strongly connected to k
            
            X = S.tocoo()

            # remove C->F edges (i.e. S[i,j] where (0) holds )
            mask = splitting[X.row] == 0
            X.row  = X.row[mask]
            X.col  = X.col[mask]
            X.data = X.data[mask]

            # remove F->C edges (i.e. S[i,j] where (1) holds )
            mask = splitting[X.col] == 0 
            X.row  = X.row[mask]
            X.col  = X.col[mask] 
            X.data = X.data[mask]

            # X now consists of strong F->F edges only
            
            # (S * S.T)[i,j] is the # of C nodes on which both i and j 
            # strongly depend (i.e. the number of k's where (2) holds)
            Y = (S*S.T) - X
            assert( Y.nnz == 0 or Y.data.min() > 0 )
   
    def test_direct_prolongator(self):
        for A in self.cases:
            S = rs_strong_connections( A, 0.0 )

            splitting = rs_cf_splitting( S )

            P = rs_direct_prolongator(A,S,splitting)


class TestRugeStubenSolver(TestCase):
    def test_poisson(self):
        cases = []
        
        cases.append( (500,) )
        cases.append( (250,250) )
        cases.append( (25,25,25) )

        for case in cases:
            A = poisson( case, format='csr' )

            numpy.random.seed(0) #make tests repeatable

            x = rand(A.shape[0])
            b = A*rand(A.shape[0]) #zeros_like(x)

            ml = ruge_stuben_solver(A, max_coarse=50)

            x_sol,residuals = ml.solve(b, x0=x, maxiter=20, tol=1e-12, return_residuals=True)

            avg_convergence_ratio = (residuals[-1]/residuals[0])**(1.0/len(residuals))
            
            assert(avg_convergence_ratio < 0.20)



################################################
##   reference implementations for unittests  ##
################################################
def reference_rs_strong_connections(A,theta):
    S = coo_matrix(A)

    # remove diagonals
    mask = S.row != S.col

    S.row  = S.row[mask]
    S.col  = S.col[mask]
    S.data = S.data[mask]
  
    S = lil_matrix(S)

    I = []
    J = []
    V = [] 
    
    for i,row in enumerate(S):
        threshold = theta * min(row.data[0])

        for j,v in zip(row.rows[0],row.data[0]):
            if v <= threshold:
                I.append(i)
                J.append(j)
                V.append(v)
   
    S = coo_matrix( (V,(I,J)), shape=A.shape).tocsr()
    return S


if __name__ == '__main__':
    nose.run(argv=['', __file__])
