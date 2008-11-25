''' Helper Functions for Testing Krylov Solvers'''

import pyamg
from pyamg.testing import *
from pyamg.util.linalg import norm
from numpy import array, zeros, ones, ravel, max, dot, eye
from scipy.sparse import csr_matrix, isspmatrix
from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg.isolve.utils import make_system
from scipy import mat, ceil, rand, diag, float32, float64, real, imag, complex64, isscalar
from warnings import warn
import scipy.sparse

__all__ = ['real_runs', 'real_runs_restrt', 'complex_runs', 'complex_runs_restrt', 'fgmres_runs']

# Simple test if krylov method converges correctly
def run_krylov(krylov, A ,b, x0=None, tol=1e-6, restrt=None, maxiter=None, M=None, callback=None, picky=False, check=True, left_precon=True, Symmetric=False):
    if Symmetric:
        try: A.psolve
        except AttributeError:
            A = A.H*A    
        else:
            psolve = A.psolve
            A = A.H*A    
            A.psolve = psolve

    if restrt != None:
        (x,flag) = krylov(A, b, x0=x0, tol=tol, restrt=restrt, maxiter=maxiter, M=M, callback=callback)
    else:
        (x,flag) = krylov(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, callback=callback)

    # Check type of x, but have to allow for a real A and a complex x, or vice versa
    if ((A.dtype == float) and (x.dtype == float)) or ((A.dtype == complex) and (x.dtype == complex)):
        #print "A and x are of different dtypes"
        assert_equal(x.dtype, A.dtype)
    
    A,M,x0,b,postprocess = make_system(A,M,x0,b,None)
    b = b.reshape(-1,1);
    x = x.reshape(-1,1);
    
    # For left preconditioning, krylov returns the preconditioned residual, not the actual residual
    # Skip for right preconditioning methods, like fgmres    
    if left_precon:
        normr = norm(ravel(M*( b - A*x.reshape(-1,1) )))
    else:     
        normr = norm(ravel(b - A*x.reshape(-1,1) ))

    if picky:
        #print "Iterated past tol = %e, || r ||_2 = %e" % (normr, 0.05*norm(ravel(b))*tol)
        assert_equal( normr > 0.01*norm(ravel(b))*tol, True)    
    
    normb = norm(ravel(b))
    if normb == 0.0:
        normb = 1.0
    if Symmetric:
        # Relax tol.  The normal equation methods just aren't numerically stable
        tol = 15*tol    

    if (flag == 0) and (normr > normb*tol) and check:
        #print "Krylov method returned with falsely reported convergence, || r ||_2 = %e" % normr
        assert(0)
    elif (normr > normb*tol) and check:
        #print "Krylov method did not converge in the maximum allowed number of iterations.  || r ||_2 = %e, flag = %d" % (normr, flag)
        assert(0)        

    return x

# Use this function to define the mat-vec routine preconditioners tested below
def matvec_precon(x, M=None):
    if isspmatrix(M):
        return M*x
    else:
        return dot(M,x)


def real_runs(krylov, n_max = 12, Weak=False, Symmetric=False):   
    '''
    Test method defined by function ptr, krylov, on a number of test cases in Real Arithmetic
    Max system size to be tested is n_max
    Weak is used to be somewhat less rigorous for methods like cgne or cgnr
    Symmetric allows this routine to be run with a symmetric solver like CG
    '''
   
    from numpy.random import seed
    seed(0)

    A = zeros((10,10))
    M = zeros((10,9))
    v1 = zeros((10,1))
    v2 = zeros((11,1))
    
    # Call krylov with all zero matrices 
    (x,flag) = krylov(A,v1)
    assert_equal(x.any(), False)             #print 'krylov failed with all zero input'
    assert_equal(flag, 0)    
    
    (x,flag) = krylov(A,v1, x0=v1)
    assert_equal(x.any(), False)             #print 'krylov failed with all zero input'
    assert_equal(flag, 0)
    
    (x,flag) = krylov(A,v1, M=A)
    assert_equal(x.any(), False)             #print 'krylov failed with all zero input'
    assert_equal(flag, 0)
    
    (x,flag) = krylov(A,v1, M=A, x0=v1)
    assert_equal(x.any(), False)             #print 'krylov failed with all zero input'
    assert_equal(flag, 0)
    
    # Build and test list of random dense matrices
    cases = []
    for i in range(1,n_max):
        cases.append( (mat(rand(i,i)), rand(i,1)) )
        cases.append( (mat(rand(i,i)), zeros((i,))) )

    for i in range(len(cases)):
        run_krylov(krylov, cases[i][0], cases[i][1], Symmetric=Symmetric)
        run_krylov(krylov, cases[i][0], ravel(cases[i][1]), Symmetric=Symmetric)
        if not Weak:
            run_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, Symmetric=Symmetric)
        run_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],1), Symmetric=Symmetric)
        run_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],), Symmetric=Symmetric)
    
    # Build and test list of random dense matrices that are single precision
    cases = []
    for i in range(1,n_max):
        cases.append( (mat(rand(i,i), dtype=float32), array(rand(i,1),dtype=float32)) )

    for i in range(len(cases)):
        run_krylov(krylov, cases[i][0], cases[i][1], tol=5e-5, Symmetric=Symmetric)
    
   # Build and test list of random sparse matrices
    cases = []
    for i in range(1,n_max):
        A = rand(i,i)
        A[ abs(A) < 0.5 ] = 0.0
        A = csr_matrix(A) + scipy.sparse.eye(i,i)
        cases.append( (A, rand(i,1)) )

    for i in range(len(cases)):
        run_krylov(krylov, cases[i][0], cases[i][1], Symmetric=Symmetric)
        run_krylov(krylov, cases[i][0], ravel(cases[i][1]), Symmetric=Symmetric)
        if not Weak:
            run_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, Symmetric=Symmetric)
        run_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],1), Symmetric=Symmetric)
        run_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],), Symmetric=Symmetric)
 
    # Build and test list of random sparse matrices that are single precision
    cases = []
    for i in range(1,n_max):
        A = rand(i,i)
        A[ abs(A) < 0.5 ] = 0.0
        A = csr_matrix(A, dtype=float32) + scipy.sparse.eye(i,i,dtype=float32)
        cases.append( (A, array(rand(i,1),dtype=float32)) )

    for i in range(len(cases)):
        run_krylov(krylov, cases[i][0], cases[i][1], tol=5e-5, Symmetric=Symmetric)


    # Build and test list of random diagonally dominant dense matrices, with diagonal preconditioner
    cases = []
    for i in range(1,n_max):
        A = mat(rand(i,i)+i*eye(i,i))
        cases.append(( A, rand(i,), mat(diag(1.0/diag(A))) ))
        A = mat(rand(i,i)+i*eye(i,i))
        cases.append(( A, zeros((i,1)), mat(diag(1.0/diag(A))) ))

    # test with matrix and mat-vec routine as the preconditioner
    for i in range(len(cases)):
        x0 = rand(cases[i][1].shape[0],1)
        a = run_krylov(krylov, cases[i][0], cases[i][1], M=cases[i][2], Symmetric=Symmetric)
        if not Weak:
            b = run_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, M=cases[i][2], Symmetric=Symmetric)
        c = run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, M=cases[i][2], Symmetric=Symmetric)
        
        cases[i][0].psolve = lambda x:matvec_precon(x, M=cases[i][2])
        d = run_krylov(krylov, cases[i][0], cases[i][1], Symmetric=Symmetric)
        if not Weak:
            e = run_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, Symmetric=Symmetric)
        f = run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, Symmetric=Symmetric)
        
        #print "matrix and mat-vec routine preconditioning yielded different results"
        assert_equal( norm(ravel(a-d)) > 1e-10, False ) 
        if not Weak:
            assert_equal( norm(ravel(b-e)) > 1e-10, False ) 
        assert_equal( norm(ravel(c-f)) > 1e-10, False )

    # Test: (1) Precon speeds up convergence 
    #       (2) Krylov halts when tol is satisfied, and iterations go no further
    #       (3) Callback_fcn works properly
    cases = []
    for i in range(8, 10):
        A = pyamg.poisson( (i,i), format='csr' )
        SA =  pyamg.smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2')
        A.psolve = SA.psolve
        cases.append(( A, rand(A.shape[0],1)    ))

    for i in range(len(cases)):
        A = cases[i][0]
        b = cases[i][1]
        x0=rand(A.shape[0],1)
        
        # Test with preconditioning
        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        #run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, picky=True, Symmetric=False)
        run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, picky=False, Symmetric=False)
        iters1 = len(residuals)
        
        # Test w/o preconditioning
        del A.psolve
        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, Symmetric=Symmetric)
        iters2 = len(residuals)

        #print "Preconditioning not speeding up convergence: " + str(iters2 - iters1) +\
        #      " speedup\nIters are " + str(iters2) + "  " + str(iters1)
        if not Weak:
            assert_equal(0.5*iters2 > iters1, True )
        else:
            assert_equal(0.9*iters2 > iters1, True )

    # test krylov with different numbers of iterations.  
    # use a callback function to make sure that the correct number of iterations are taken.
    cases=[]
    for i in range(n_max, n_max+2):
        A = pyamg.poisson( (i,i), format='csr' )
        cases.append(( A, rand(A.shape[0],1)    ))
    
    for i in range(len(cases)):
        A = cases[i][0]
        b=cases[i][1]
        x0=rand(A.shape[0],1)
        maxiter=int(ceil(A.shape[0]/8))

        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, maxiter=maxiter, tol=1e-12, check=False, Symmetric=False)
        #print "callback used an incorrect number of times"
        assert_equal( len(residuals), maxiter)


def real_runs_restrt(krylov, n_max=12):
    '''
    Test the restrt option for krylov (e.g. gmres, fgmres)
    All test problems are Poisson with n_max^2 as the matrix size 
    '''

    # test krylov with different numbers of iterations and restarts.  
    # use a callback function to make sure that the correct number of iterations are taken.
    cases=[]
    for i in range(n_max, n_max+2):
        A = pyamg.poisson( (i,i), format='csr' )
        cases.append(( A, rand(A.shape[0],1)    ))

    for i in range(len(cases)):
        A = cases[i][0]
        b=cases[i][1]
        x0=rand(A.shape[0],1)
        
        maxiter=1 
        restrt=int(ceil(A.shape[0]/8))
        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, restrt=restrt, maxiter=maxiter, tol=1e-12, check=False)
        #print "callback used an incorrect number of times"
        assert_equal( len(residuals), restrt*maxiter)

        maxiter=2
        restrt=int(ceil(A.shape[0]/8))
        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, restrt=restrt, maxiter=maxiter, tol=1e-12, check=False)
        #print "callback used an incorrect number of times"
        assert_equal( len(residuals), restrt*maxiter)
        
        maxiter=3 
        restrt=int(ceil(A.shape[0]/8))
        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, restrt=restrt, maxiter=maxiter, tol=1e-12, check=False)
        #print "callback used an incorrect number of times"
        assert_equal( len(residuals), restrt*maxiter)


def complex_runs(krylov, n_max = 12, Weak=False, Symmetric=False):
    '''
    Test method, krylov, on a number of test cases in complex arithmetic
    Largest System size is defined by n_max
    Weak is used to be somewhat less rigorous for methods like cgne or cgnr
    Symmetric allows this routine to be run with a symmetric solver like CG
    '''
    # The Real Test routine tests for invalid and all zero inputs
    
    # Build and test list of random dense matrices
    cases = []
    for i in range(1,n_max):
        cases.append( (mat(rand(i,i) + 1.0j*rand(i,i)), (rand(i,1)+1.0j*rand(i,1)) ) )
        cases.append( (mat(rand(i,i) + 1.0j*rand(i,i)), 1.0j*rand(i,1) ) )
        cases.append( (mat(rand(i,i) + 1.0j*rand(i,i)), rand(i,1) ) )

        cases.append( (mat(1.0j*rand(i,i)), (rand(i,1) + 1.0j*rand(i,1)) ) )
        cases.append( (mat(1.0j*rand(i,i)), 1.0j*rand(i,1) ) )
        cases.append( (mat(1.0j*rand(i,i)), rand(i,1) ) )
        
        cases.append( (mat(rand(i,i)), (rand(i,1) + 1.0j*rand(i,1)) ) )
        cases.append( (mat(rand(i,i)), 1.0j*rand(i,1) ) )
        cases.append( (mat(rand(i,i)), rand(i,1) ) )
        
        cases.append( (mat(rand(i,i) + 1.0j*rand(i,i)), zeros((i,))) )

    for i in range(len(cases)):
        run_krylov(krylov, cases[i][0], cases[i][1], Symmetric=Symmetric)
        run_krylov(krylov, cases[i][0], ravel(cases[i][1]), Symmetric=Symmetric)
        if not Weak:
            run_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, Symmetric=Symmetric)
        run_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],1), Symmetric=Symmetric)
        run_krylov(krylov, cases[i][0], cases[i][1], x0=(rand(cases[i][1].shape[0],)+1.0j*rand(cases[i][1].shape[0],)), Symmetric=Symmetric)
    
    # Build and test list of random dense matrices that are single precision
    cases = []
    for i in range(1,n_max):
        cases.append( (mat(rand(i,i), dtype=float32), array(rand(i,1),dtype=complex64)) )

    for i in range(len(cases)):
        run_krylov(krylov, cases[i][0], cases[i][1], tol=1e-3, Symmetric=Symmetric)
    
   # Build and test list of random sparse matrices
    cases = []
    for i in range(1,n_max):
        Areal = rand(i,i)
        Areal[ abs(Areal) < 0.5 ] = 0.0
        Areal = csr_matrix(Areal) + scipy.sparse.eye(i,i)
 
        Aimag = rand(i,i)
        Aimag[ abs(Aimag) < 0.5 ] = 0.0
        Aimag = csr_matrix(Aimag) + scipy.sparse.eye(i,i)
        Aimag.data = Aimag.data + 1.0j*rand(Aimag.data.shape[0],)
        
        Aimag2 = rand(i,i)
        Aimag2[ abs(Aimag2) < 0.5 ] = 0.0
        Aimag2 = csr_matrix(Aimag2)  + scipy.sparse.eye(i,i)
        Aimag2.data = 1.0j*Aimag2.data
        
        cases.append( (Aimag, (rand(i,1)+1.0j*rand(i,1)) ) )
        cases.append( (Aimag, 1.0j*rand(i,1) ) )
        cases.append( (Aimag, rand(i,1) ) )

        cases.append( (Aimag2, (rand(i,1) + 1.0j*rand(i,1)) ) )
        cases.append( (Aimag2, 1.0j*rand(i,1) ) )
        cases.append( (Aimag2, rand(i,1) ) )
        
        cases.append( (Areal, (rand(i,1) + 1.0j*rand(i,1)) ) )
        cases.append( (Areal, 1.0j*rand(i,1) ) )
        cases.append( (Areal, rand(i,1) ) )
        
        cases.append( (Aimag, zeros((i,))) )

    for i in range(len(cases)):
        run_krylov(krylov, cases[i][0], cases[i][1], Symmetric=Symmetric)
        run_krylov(krylov, cases[i][0], ravel(cases[i][1]), Symmetric=Symmetric)
        if not Weak:
            run_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, Symmetric=Symmetric)
        run_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],1), Symmetric=Symmetric)
        run_krylov(krylov, cases[i][0], cases[i][1], x0=(rand(cases[i][1].shape[0],)+1.0j*rand(cases[i][1].shape[0],)), Symmetric=Symmetric)
 
    # Build and test list of random sparse matrices that are single precision
    cases = []
    for i in range(1,n_max):
        A = rand(i,i)
        A[ abs(A) < 0.5 ] = 0.0
        A = csr_matrix(A, dtype=complex64) + scipy.sparse.eye(i,i,dtype=complex64);
        A.data = A.data + 1.0j*rand(A.data.shape[0],)
        cases.append( (A, array(rand(i,1),dtype=complex64)) )

    for i in range(len(cases)):
        run_krylov(krylov, cases[i][0], cases[i][1], tol=5e-5, Symmetric=Symmetric)


    # Build and test list of random diagonally dominant dense matrices, with diagonal preconditioner
    cases = []
    for i in range(1,n_max):
        A = mat(rand(i,i) + 2*i*eye(i,i) + 1.0j*rand(i,i))
        cases.append(( A, rand(i,), mat(diag(1.0/diag(A))) ))
        A = mat(rand(i,i) + 2*i*eye(i,i) + 1.0j*rand(i,i))
        cases.append(( A, rand(i,) + 1.0j*rand(i,), mat(diag(1.0/diag(A))) ))

    # test with matrix and mat-vec routine as the preconditioner
    for i in range(len(cases)):
        x0 = rand(cases[i][1].shape[0],1) + 1.0j*rand(cases[i][1].shape[0],1)
        a = run_krylov(krylov, cases[i][0], cases[i][1], M=cases[i][2], check=False, Symmetric=Symmetric)
        if not Weak:
            b = run_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, M=cases[i][2], x0=real(x0), check=False, Symmetric=Symmetric)
        c = run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, M=cases[i][2], check=False, Symmetric=Symmetric)
        
        cases[i][0].psolve = lambda x:matvec_precon(x, M=cases[i][2])
        d = run_krylov(krylov, cases[i][0], cases[i][1], check=False, Symmetric=Symmetric)
        if not Weak:
            e = run_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, x0=real(x0), check=False, Symmetric=Symmetric)
        f = run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, check=False, Symmetric=Symmetric)
        
        #print "matrix and mat-vec routine preconditioning yielded different results"
        assert_equal( norm(ravel(a-d)) > 1e-10, False ) 
        if not Weak:
            assert_equal( norm(ravel(b-e)) > 1e-10, False ) 
        assert_equal( norm(ravel(c-f)) > 1e-10, False )

    # Test: (1) Precon speeds up convergence 
    #       (2) Krylov halts when tol is satisfied, and iterations go no further
    #       (3) Callback_fcn works properly
    cases = []
    for i in range(8,10):
        A = i*pyamg.poisson( (i,i), format='csr' )
        SA = pyamg.smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2',
                                        mat_flag='symmetric') 
        A.psolve = SA.psolve
        cases.append(( A, rand(A.shape[0],1)    ))

    for i in range(len(cases)):
        A = cases[i][0]
        b = cases[i][1]
        x0=rand(A.shape[0],1)
        
        # Test with preconditioning
        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        #run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, picky=True, Symmetric=Symmetric, check=False)
        run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, picky=False, Symmetric=Symmetric, check=False)
        iters1 = len(residuals)
        
        # Test w/o preconditioning
        del A.psolve
        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, Symmetric=Symmetric, check=False)
        iters2 = len(residuals)

        #print "Preconditioning not speeding up convergence: " + str(iters2 - iters1) +\
        #      " speedup\nIters are " + str(iters2) + "  " + str(iters1)
        if not Weak:
            assert_equal(0.5*iters2 > iters1, True )
        else:
            assert_equal(0.9*iters2 > iters1, True )

    # test krylov with different numbers of iterations  
    # use a callback function to make sure that the correct number of iterations are taken.
    cases=[]
    for i in range(n_max, n_max+2):
        A = pyamg.poisson( (i,i), format='csr' )
        A.data = 10*A.data + 0.1j*rand(A.data.shape[0],)
        cases.append(( A, rand(A.shape[0],1)    ))

    for i in range(len(cases)):
        A = cases[i][0]
        b=cases[i][1]
        x0=rand(A.shape[0],1)
        
        maxiter=int(ceil(A.shape[0]/8))
        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, maxiter=maxiter, tol=1e-12, check=False, Symmetric=False)
        #print "callback used an incorrect number of times"
        assert_equal( len(residuals), maxiter)


def complex_runs_restrt(krylov, n_max=12):
    '''
    Test the restrt option for krylov (e.g. gmres, fgmres)
    All test problems are Poisson with n_max^2 as the matrix size 
    '''

    cases=[]
    for i in range(n_max, n_max+2):
        A = pyamg.poisson( (i,i), format='csr' )
        A.data = 10*A.data + 0.1j*rand(A.data.shape[0],)
        cases.append(( A, rand(A.shape[0],1)    ))

    for i in range(len(cases)):
        A = cases[i][0]
        b=cases[i][1]
        x0=rand(A.shape[0],1)
        
        maxiter=1 
        restrt=int(ceil(A.shape[0]/8))
        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, restrt=restrt, maxiter=maxiter, tol=1e-12, check=False)
        #print "callback used an incorrect number of times"
        assert_equal( len(residuals), restrt*maxiter)

        maxiter=2 
        restrt=int(ceil(A.shape[0]/8))
        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, restrt=restrt, maxiter=maxiter, tol=1e-12, check=False)
        #print "callback used an incorrect number of times"
        assert_equal( len(residuals), restrt*maxiter)

        maxiter=3
        restrt=int(ceil(A.shape[0]/8))
        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        run_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, restrt=restrt, maxiter=maxiter, tol=1e-12, check=False)
        #print "callback used an incorrect number of times"
        assert_equal( len(residuals), restrt*maxiter)


def fgmres_runs(krylov):
    ''' 
    Test FGMRES by using a variable preconditioner.  
    We use gmres here as the variable preconditioner. 
    ''' 
    from pyamg.krylov import gmres 

    # Real and Imaginary Tests of Poisson Type Matrices
    cases = []
    for i in [7, 8]:
        A = pyamg.poisson( (i,i,i), format='csr' )
        A1 = A.copy()
        A1.data = A1.data + 1.0j*1e-5*rand(A1.data.shape[0],)
        A2 = 1.0j*A.copy()
        A3 = A.copy()

        cases.append( {'A' : A1,  'b' : 1.0j*rand(A.shape[0],1),  'x0' : zeros((A.shape[0],1)), 'tol' : 1e-5} ) 
        cases.append( {'A' : A2,  'b' : rand(A.shape[0],1) + 1.0j*rand(A.shape[0],1),  'x0' : rand(A.shape[0],1)+1.0j*rand(A.shape[0],1),  'tol' : 1e-8} ) 
        cases.append( {'A' : A3,  'b' : rand(A.shape[0],1),  'x0' : rand(A.shape[0],1),  'tol' : 1e-5} ) 
        cases.append( {'A' : A3,  'b' : rand(A.shape[0],1),  'x0' : zeros((A.shape[0],1)),  'tol' : 1e-8} ) 
    
    # Run Tests
    maxiter=5
    for i in range(len(cases)):
        A = cases[i]['A']
        b = cases[i]['b']
        tol = cases[i]['tol']
        x0 = cases[i]['x0']
        precon = lambda guess:gmres(A, b=guess, x0=zeros(b.shape), tol=tol, maxiter=maxiter)[0]
        M = LinearOperator(A.shape, matvec=precon, dtype=A.dtype)
        
        # With preconditioning
        residuals = []
        def callback_fcn(x):
            if isscalar(x): residuals.append(x)
            else:           residuals.append( norm(ravel(b)-ravel(A*x)) )
        x = run_krylov(krylov, A, b, x0=x0, tol=tol, M=M, callback=callback_fcn, left_precon=False)
        iters1 = len(residuals)
    
        # Without preconditioning
        residuals2 = []
        def callback_fcn(x):
            if isscalar(x): residuals2.append(x)
            else:           residuals2.append( norm(ravel(b)-ravel(A*x)) )
        x2 = run_krylov(krylov, A, b, x0=x0, tol=tol, callback=callback_fcn, left_precon=False)
        iters2 = len(residuals2)
        
        #print "%d iter speed up, iters are %d and %d , system size: %d x %d\n" % (speed_up, iters2, iters1, A.shape[0], A.shape[1])
        assert_equal( 0.5*iters2 > iters1, True)


