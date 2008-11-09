from pyamg.testing import *

import numpy
from numpy import ravel, ones, concatenate, cumsum, zeros
from scipy import rand
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix

from pyamg.gallery import poisson, load_example
from pyamg.strength import classical_strength_of_connection

from pyamg.classical import split
from pyamg.classical.classical import ruge_stuben_solver
from pyamg.classical.interpolate import direct_interpolation

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

        for name in ['knot','airfoil','bar']:
            ex = load_example(name)
            self.cases.append( ex['A'].tocsr() )

    def test_RS_splitting(self):
        for A in self.cases:
            S = classical_strength_of_connection(A, 0.0)

            splitting = split.RS( S )

            assert( splitting.min() >= 0 )     #could be all 1s
            assert_equal( splitting.max(), 1 ) 

            S.data[:] = 1

            # check that all F-nodes are strongly connected to a C-node
            assert( (splitting + S*splitting).min() > 0 )

            ### THIS IS NOT STRICTLY ENFORCED!
            ## check that all strong connections S[i,j] satisfy either:
            ## (0) i is a C-node
            ## (1) j is a C-node
            ## (2) k is a C-node and both i and j are are strongly connected to k
            #
            #X = S.tocoo()

            ## remove C->F edges (i.e. S[i,j] where (0) holds )
            #mask = splitting[X.row] == 0
            #X.row  = X.row[mask]
            #X.col  = X.col[mask]
            #X.data = X.data[mask]

            ## remove F->C edges (i.e. S[i,j] where (1) holds )
            #mask = splitting[X.col] == 0 
            #X.row  = X.row[mask]
            #X.col  = X.col[mask] 
            #X.data = X.data[mask]

            ## X now consists of strong F->F edges only
            #
            ## (S * S.T)[i,j] is the # of C nodes on which both i and j 
            ## strongly depend (i.e. the number of k's where (2) holds)
            #Y = (S*S.T) - X
            #assert( Y.nnz == 0 or Y.data.min() > 0 )
   
    def test_direct_interpolation(self):
        for A in self.cases:

            S = classical_strength_of_connection(A, 0.0)
            splitting = split.RS( S )

            result   = direct_interpolation(A,S,splitting)                
            expected = reference_direct_interpolation( A, S, splitting )

            assert_almost_equal( result.todense(), expected.todense() )

            

class TestSolverPerformance(TestCase):
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

            residuals = []
            x_sol = ml.solve(b, x0=x, maxiter=20, tol=1e-12, residuals=residuals)

            avg_convergence_ratio = (residuals[-1]/residuals[0])**(1.0/len(residuals))
            
            assert(avg_convergence_ratio < 0.20)



################################################
##   reference implementations for unittests  ##
################################################


def reference_direct_interpolation(A,S,splitting):

    A = coo_matrix(A)
    S = coo_matrix(S)  

    #strong C points
    c_mask = splitting[S.col] == 1
    C_s = coo_matrix( (S.data[c_mask],(S.row[c_mask],S.col[c_mask])), shape=S.shape)

    #strong F points
    f_mask = ~c_mask
    F_s = coo_matrix( (S.data[f_mask],(S.row[f_mask],S.col[f_mask])), shape=S.shape)

    # split A in to + and -
    mask = (A.data > 0) & (A.row != A.col)
    A_pos = coo_matrix( (A.data[mask],(A.row[mask],A.col[mask])), shape=A.shape)
    mask = (A.data < 0) & (A.row != A.col)
    A_neg = coo_matrix( (A.data[mask],(A.row[mask],A.col[mask])), shape=A.shape)

    # split C_S in to + and -
    mask = C_s.data > 0
    C_s_pos = coo_matrix( (C_s.data[mask],(C_s.row[mask],C_s.col[mask])), shape=A.shape)
    mask = ~mask
    C_s_neg = coo_matrix( (C_s.data[mask],(C_s.row[mask],C_s.col[mask])), shape=A.shape)

    sum_strong_pos = ravel(C_s_pos.sum(axis=1))
    sum_strong_neg = ravel(C_s_neg.sum(axis=1))

    sum_all_pos = ravel(A_pos.sum(axis=1))
    sum_all_neg = ravel(A_neg.sum(axis=1))

    diag = A.diagonal()

    alpha = sum_all_neg / sum_strong_neg
    beta  = sum_all_pos / sum_strong_pos

    mask = sum_strong_pos == 0
    diag[mask] += sum_all_pos[mask]
    beta[mask] = 0

    C_s_neg.data *= -alpha[C_s_neg.row]/diag[C_s_neg.row]
    C_s_pos.data *= -beta[C_s_pos.row]/diag[C_s_pos.row]
    
    C_rows = splitting.nonzero()[0] 
    C_inject = coo_matrix( (ones(sum(splitting)),(C_rows,C_rows)), shape=A.shape)
    
    P = C_s_neg.tocsr() + C_s_pos.tocsr() + C_inject.tocsr()

    map = concatenate(([0],cumsum(splitting)))
    P = csr_matrix( (P.data,map[P.indices],P.indptr), shape=(P.shape[0],map[-1]))

    return P

