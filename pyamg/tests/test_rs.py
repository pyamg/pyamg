from scipy.testing import *

import numpy
from scipy import rand
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix

from pyamg.gallery import poisson

from pyamg.rs import * 


class TestStrengthOfConnection(TestCase):
    def test_rs_strong_connections(self):
        cases = []

        # random matrices
        numpy.random.seed(0)
        for N in [2,3,5]:
            cases.append( csr_matrix(-rand(N,N)) )

        # poisson problems in 1D and 2D
        for N in [2,3,5,7,10,11,19]:
            cases.append( poisson( (N,), format='csr') )
        for N in [2,3,5,7,10,11]:
            cases.append( poisson( (N,N), format='csr') )
        
        for theta in [ 0.0, 0.05, 0.25, 0.50, 0.90 ]:
            for A in cases:
                result   = rs_strong_connections( A, theta )
                expected = reference_rs_strong_connections( A, theta )
                assert_equal( result.nnz, expected.nnz )
                assert_equal( result.todense(), expected.todense() )
    

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

            ml = ruge_stuben_solver(A)

            x_sol,residuals = ml.solve(b,x0=x,maxiter=20,tol=1e-12,return_residuals=True)

            avg_convergence_ratio = (residuals[-1]/residuals[0])**(1.0/len(residuals))
            
            assert(avg_convergence_ratio < 0.10)



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
