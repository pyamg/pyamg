from pyamg.testing import *
import scipy.sparse
import numpy

from numpy import array, zeros, ones, mat, ravel
from scipy import rand, mat
from scipy.linalg import pinv2
from scipy.sparse import bsr_matrix, csr_matrix, eye, isspmatrix_bsr

from pyamg.aggregation.smooth import Satisfy_Constraints
from pyamg.gallery import poisson, linear_elasticity, load_example, gauge_laplacian
from pyamg.aggregation import smoothed_aggregation_solver, rootnode_solver
from pyamg.amg_core import incomplete_BSRmatmat

class TestEnergyMin(TestCase):
    def test_incomplete_BSRmatmat(self):
        # Test a critical helper routine for energy_min_prolongation(...)
        # We test that (A*B).multiply(mask) = incomplete_BSRmatmat(A,B,mask)
        cases = []

        # 1x1 tests
        A = csr_matrix(mat([[1.1]])).tobsr(blocksize=(1,1))
        B = csr_matrix(mat([[1.0]])).tobsr(blocksize=(1,1))
        A2 = csr_matrix(mat([[0.]])).tobsr(blocksize=(1,1))
        mask = csr_matrix(mat([[1.]])).tobsr(blocksize=(1,1))
        mask2 = csr_matrix(mat([[0.]])).tobsr(blocksize=(1,1))
        cases.append( (A,A,mask) ) 
        cases.append( (A,A,mask2) ) 
        cases.append( (A,B,mask) ) 
        cases.append( (A,A2,mask) ) 
        cases.append( (A,A2,mask2) ) 
        cases.append( (A2,A2,mask) ) 

        # 2x2 tests
        A = csr_matrix(mat([[1.,2.],[2.,4.]])).tobsr(blocksize=(2,2))
        B = csr_matrix(mat([[1.3,2.],[2.8,4.]])).tobsr(blocksize=(2,2))
        A2 = csr_matrix(mat([[1.3,0.],[0.,4.]])).tobsr(blocksize=(2,2))
        B2 = csr_matrix(mat([[1.3,0.],[2.,4.]])).tobsr(blocksize=(2,1))
        mask = csr_matrix( (ones(4, dtype=float),(array([0,0,1,1]),array([0,1,0,1]))), shape=(2,2) ).tobsr(blocksize=(2,2))
        mask2 = csr_matrix( (array([1.,0.,1.,0.]),(array([0,0,1,1]),array([0,1,0,1]))), shape=(2,2) ).tobsr(blocksize=(2,1))
        mask2.eliminate_zeros()
        cases.append( (A,A,mask) ) 
        cases.append( (A,B,mask) ) 
        cases.append( (A2,A2,mask) ) 
        cases.append( (A2,B2,mask2) ) 
        cases.append( (A,B2,mask2) ) 

        A = A.tobsr(blocksize=(1,1))
        B = B.tobsr(blocksize=(1,1))
        A2 = A2.tobsr(blocksize=(1,1))
        B2 = B2.tobsr(blocksize=(1,1))
        mask = A.copy()
        mask.data[:] = 1.0
        mask2 = A2.copy()
        mask2.data[:] = 1.0
        mask3 = csr_matrix( (ones(2, dtype=float),(array([0,1]),array([0,0]))), shape=(2,2) ).tobsr(blocksize=(1,1))
        cases.append( (A,B,mask) )
        cases.append( (A2,B2,mask2) )
        cases.append( (A,B,mask3) )
        cases.append( (A2,B2,mask3) )

        # 4x4 tests
        A = mat([[  0. ,  16.9,   6.4,   0.0],
                 [ 16.9,  13.8,   7.2,   0. ],
                 [  6.4,   7.2,  12. ,   6.1],
                 [  0.0,   0. ,   6.1,   0. ]])
        B = A.copy()
        B = A[:,0:2]
        B[1,0] = 3.1
        B[2,1] = 10.1
        A2 = A.copy()
        A2[1,:] = 0.0
        A3 = A2.copy()
        A3[:,1] = 0.0
        A = csr_matrix(A).tobsr(blocksize=(2,2))
        A2 = csr_matrix(A2).tobsr(blocksize=(2,2))
        A3 = csr_matrix(A3).tobsr(blocksize=(2,2))
        B = csr_matrix(B).tobsr(blocksize=(2,2))
        B2 = csr_matrix(B).tobsr(blocksize=(2,1))

        mask = A.copy()
        mask.data[:] = 1.0
        mask2 = B.copy()
        mask2.data[:] = 1.0
        mask3 = B2.copy()
        mask3.data[:] = 1.0
        cases.append( (A,B,mask2) )
        cases.append( (A2,B,mask2) )
        cases.append( (A3,B,mask2) )
        cases.append( (A3,A3,mask) )
        cases.append( (A,A,mask) )
        cases.append( (A2,A2,mask) )
        cases.append( (A,A2,mask) )
        cases.append( (A3,A,mask) )
        cases.append( (A, B2,mask3) )
        cases.append( (A2, B2,mask3) )
        cases.append( (A3, B2,mask3) )

        # Laplacian tests
        A = poisson((5,5),format='csr')
        A = A.tobsr(blocksize=(5,5))
        Ai = 1.0j*A
        B = A.todense()
        B = csr_matrix(B[:,0:8])
        B = B.tobsr(blocksize=(5,8))
        Bi = 1.0j*B
        B2 = B.tobsr(blocksize=(5,2))
        B2i = 1.0j*B2
        B3 = B.tobsr(blocksize=(5,1))
        B3i = 1.0j*B3
        mask=B.copy()
        mask.data[:]=1.0
        mask2=B2.copy()
        mask2.data[:]=1.0
        mask3=B3.copy()
        mask3.data[:]=1.0
        cases.append( (A,B,mask) )
        cases.append( (A,B2,mask2) )
        cases.append( (A,B3,mask3) )
        cases.append( (Ai,Bi,mask) )
        cases.append( (Ai,B2i,mask2) )
        cases.append( (Ai,B3i,mask3) )

        for case in cases:
            A = case[0].tobsr()
            B = case[1].tobsr()
            B2 = case[1].T.tobsr()
            mask = case[2].tobsr()
            
            A.sort_indices()
            B.sort_indices()
            B2.sort_indices()
            mask.sort_indices()
            
            result = mask.copy()
            result.data = array(result.data, dtype=A.dtype)
            incomplete_BSRmatmat(A.indptr,A.indices,ravel(A.data), B2.indptr,B2.indices,ravel(B2.data), 
                                 result.indptr,result.indices,ravel(result.data), 
                                 mask.shape[0], mask.blocksize[0], mask.blocksize[1])

            exact = (A*B).multiply(mask)
            differ = max(abs(ravel((result-exact).todense())))
            assert_almost_equal(differ, 0.0)


    def test_range(self):
        """Check that P*R=B"""
        numpy.random.seed(0) #make tests repeatable
        
        cases = []

        ##
        # Simple, real-valued diffusion problems
        X = load_example('airfoil')
        A = X['A'].tocsr(); B = X['B']
        cases.append((A,B,('jacobi', {'filter' : True, 'weighting' : 'local'}) ))
        cases.append((A,B,('jacobi', {'filter' : True, 'weighting' : 'block'}) ))

        cases.append((A,B,('energy', {'maxiter' : 3}) ))
        cases.append((A,B,('energy', {'krylov' : 'cgnr'}) ))
        cases.append((A,B,('energy', {'krylov' : 'gmres', 'degree' : 2}) ))
        
        A = poisson((10,10), format='csr')
        B = ones((A.shape[0],1))
        cases.append((A,B,('jacobi', {'filter' : True, 'weighting' : 'diagonal'}) ))
        cases.append((A,B,('jacobi', {'filter' : True, 'weighting' : 'local'}) ))

        cases.append((A,B,'energy'))
        cases.append((A,B,('energy', {'degree' : 2}) ))
        cases.append((A,B,('energy', {'krylov' : 'cgnr', 'degree' : 2}) ))
        cases.append((A,B,('energy', {'krylov' : 'gmres'}) ))

        ##
        # Simple, imaginary-valued problems
        iA = 1.0j*A
        iB = 1.0 + rand(iA.shape[0],2) + 1.0j*(1.0 + rand(iA.shape[0],2))
        
        cases.append((iA, B,('jacobi', {'filter' : True, 'weighting' : 'diagonal'}) ))
        cases.append((iA, B,('jacobi', {'filter' : True, 'weighting' : 'block'}) ))
        cases.append((iA,iB,('jacobi', {'filter' : True, 'weighting' : 'local'}) ))
        cases.append((iA,iB,('jacobi', {'filter' : True, 'weighting' : 'block'}) ))
        
        cases.append((iA.tobsr(blocksize=(5,5)),  B, ('jacobi', {'filter' : True, 'weighting' : 'block'}) ))
        cases.append((iA.tobsr(blocksize=(5,5)), iB, ('jacobi', {'filter' : True, 'weighting' : 'block'}) ))
        
        cases.append((iA,B, ('energy', {'krylov' : 'cgnr', 'degree' : 2}) ))
        cases.append((iA,iB,('energy', {'krylov' : 'cgnr'}) ))
        cases.append((iA.tobsr(blocksize=(5,5)),B, ('energy', {'krylov' : 'cgnr', 'degree' : 2, 'maxiter' : 3}) ))
        cases.append((iA.tobsr(blocksize=(5,5)),iB,('energy', {'krylov' : 'cgnr'}) ))
 
        cases.append((iA,B, ('energy', {'krylov' : 'gmres'}) ))
        cases.append((iA,iB,('energy', {'krylov' : 'gmres', 'degree' : 2}) ))
        cases.append((iA.tobsr(blocksize=(5,5)),B, ('energy', {'krylov' : 'gmres', 'degree' : 2, 'maxiter' : 3}) ))
        cases.append((iA.tobsr(blocksize=(5,5)),iB,('energy', {'krylov' : 'gmres'}) ))

        ##
        #
        # Simple, imaginary-valued problems
        iA = A + 1.0j*scipy.sparse.eye(A.shape[0], A.shape[1])

        cases.append((iA,B, ('jacobi', {'filter' : True, 'weighting' : 'local'}) ))
        cases.append((iA,B, ('jacobi', {'filter' : True, 'weighting' : 'block'}) ))
        cases.append((iA,iB,('jacobi', {'filter' : True, 'weighting' : 'diagonal'}) ))
        cases.append((iA,iB,('jacobi', {'filter' : True, 'weighting' : 'block'}) ))
        cases.append((iA.tobsr(blocksize=(4,4)), iB, ('jacobi', {'filter' : True, 'weighting' : 'block'}) ))
        
        cases.append((iA,B,  ('energy', {'krylov' : 'cgnr'}) ))
        cases.append((iA.tobsr(blocksize=(4,4)),iB,('energy', {'krylov' : 'cgnr'}) ))

        cases.append((iA,B,                         ('energy', {'krylov' : 'gmres'}) ))
        cases.append((iA.tobsr(blocksize=(4,4)),iB, ('energy', {'krylov' : 'gmres', 'degree' : 2, 'maxiter' : 3}) ))

        ##
        #
        A = gauge_laplacian(10, spacing=1.0, beta=0.21)
        B = ones((A.shape[0],1))
        cases.append((A,iB,('jacobi', {'filter' : True, 'weighting' : 'diagonal'}) ))
        cases.append((A,iB,('jacobi', {'filter' : True, 'weighting' : 'local'}) ))

        cases.append((A,B,                        ('energy', {'krylov' : 'cg'}) ))
        cases.append((A,iB,                       ('energy', {'krylov' : 'cgnr'}) ))
        cases.append((A,iB,                       ('energy', {'krylov' : 'gmres'}) ))
        
        cases.append((A.tobsr(blocksize=(2,2)),B, ('energy', {'krylov' : 'cgnr', 'degree' : 2, 'maxiter' : 3}) ))
        cases.append((A.tobsr(blocksize=(2,2)),iB,('energy', {'krylov' : 'cg'}) ))
        cases.append((A.tobsr(blocksize=(2,2)),B, ('energy', {'krylov' : 'gmres', 'degree' : 2, 'maxiter' : 3}) ))

        ##
        #
        A,B = linear_elasticity((10,10))
        cases.append((A,B,('jacobi', {'filter' : True, 'weighting' : 'diagonal'}) ))
        cases.append((A,B,('jacobi', {'filter' : True, 'weighting' : 'local'}) ))
        cases.append((A,B,('jacobi', {'filter' : True, 'weighting' : 'block'}) ))

        cases.append((A,B,('energy', {'degree' : 2}) ))
        cases.append((A,B,('energy', {'krylov' : 'cgnr'}) ))
        cases.append((A,B,('energy', {'krylov' : 'gmres', 'degree' : 2}) ))
               

        ##
        # Classic SA cases
        for A,B,smooth in cases:
            ml = smoothed_aggregation_solver(A, B=B, max_coarse=1, max_levels=2, smooth=smooth )
            P = ml.levels[0].P
            B = ml.levels[0].B
            R = ml.levels[1].B
            assert_almost_equal(P*R, B)
        
        def blocksize(A):
            # Helper Function: return the blocksize of a matrix 
            if isspmatrix_bsr(A):
                return A.blocksize[0]
            else:
                return 1

        ##
        # Root-node cases
        counter = 0
        for A,B,smooth in cases:
            counter += 1
            
            if isinstance( smooth, tuple):
                smoother = smooth[0]
            else:
                smoother = smooth
            
            if smoother == 'energy' and (B.shape[1] >= blocksize(A)):
                 ml = rootnode_solver(A, B=B, max_coarse=1, max_levels=2, smooth=smooth, keep=True, symmetry='nonsymmetric')
                 T = ml.levels[0].T.tocsr()
                 Cpts = ml.levels[0].Cpts
                 Bf = ml.levels[0].B 
                 Bf_H = ml.levels[0].BH 
                 Bc = ml.levels[1].B 
                 P = ml.levels[0].P.tocsr()
                 ##
                 # P should preserve B in its range, wherever P 
                 # has enough nonzeros
                 mask = ((P.indptr[1:] - P.indptr[:-1]) >= B.shape[1])
                 assert_almost_equal( (P*Bc)[mask,:], Bf[mask,:])
                 assert_almost_equal( (P*Bc)[mask,:], Bf_H[mask,:])
                 ##
                 # P should be the identity at Cpts
                 I = eye(T.shape[1], T.shape[1], format='csr', dtype=T.dtype)
                 I2 = P[Cpts,:]
                 assert_almost_equal(I.data, I2.data)
                 assert_equal(I.indptr, I2.indptr)
                 assert_equal(I.indices, I2.indices)
                 ##
                 # T should be the identity at Cpts
                 I2 = T[Cpts,:]
                 assert_almost_equal(I.data, I2.data)
                 assert_equal(I.indptr, I2.indptr)
                 assert_equal(I.indices, I2.indices)


#class TestSatisfyConstaints(TestCase):
#    def test_scalar(self):
#
#        U = bsr_matrix([[1,2],[2,1]], blocksize=(1,1))
#        Sparsity_Pattern = bsr_matrix([[1,1],[1,1]],blocksize=(1,1))
#        B = mat(array([[1],[1]]))
#        BtBinv = [ array([[0.5]]), array([[0.5]]) ]
#        colindices = [ array([0,1]), array([0,1]) ]
#
#        U = Satisfy_Constraints(U, Sparsity_Pattern, B, BtBinv, colindices)
#
#
#        assert_equal(U.todense(), array([[0,1],[1,0]]))
#        assert_almost_equal(U*B, 0*U*B)
#
#    def test_block(self):
#        SparsityPattern = array([[1,1,0,0],
#                                 [1,1,0,0],
#                                 [1,1,1,1],
#                                 [0,0,1,1],
#                                 [0,0,1,1]])
#        U = array([[1,2,0,0],
#                   [4,3,0,0],
#                   [5,6,8,7],
#                   [0,0,4,1],
#                   [0,0,2,3]])
#        
#        Sparsity_Pattern = bsr_matrix(SparsityPattern, blocksize=(1,2))
#        U = bsr_matrix(U, blocksize=(1,2))
#        B = array([[1,1],
#                   [1,2],
#                   [1,3],
#                   [1,4]])
#        colindices = [ array([0,1]), array([0,1]), array([0,1,2,3]), array([0,1]), array([0,1]) ]
#        
#        BtBinv = zeros((5,2,2)) 
#        for i in range(5):
#            colindx = colindices[i]
#            if len(colindx) > 0:
#                Bi = mat(B)[colindx,:]
#                BtBinv[i] = pinv2(Bi.T*Bi)
#
#        U = Satisfy_Constraints(U, Sparsity_Pattern, B, BtBinv, colindices)
#        assert_almost_equal(U*B, 0*U*B)

