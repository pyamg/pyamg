"""Solve an arbitrary system"""

__docformat__ = "restructuredtext en"

import numpy
import scipy
from scipy.sparse import isspmatrix_csr, isspmatrix_bsr, csr_matrix
from pyamg import smoothed_aggregation_solver
from pyamg.util.linalg import ishermitian, norm

__all__ = ['solveit']

def solveit(A, b, x0=None, tol=1e-5, maxiter=400, return_solver=False, solver=None, verb=True):
    """
    Solve the arbitrary system Ax=b, The matrix A can be non-Hermitian,
    indefinite, Hermitian positive-definite, etc...  Generic and robust 
    settings for smoothed_aggregation_solver(..) are used to invert A.


    Parameters
    ----------
    A : {array, matrix, csr_matrix, bsr_matrix}
        Matrix to invert, CSR or BSR format preferred for efficiency 
    b : {array}
        Right hand side.
    x0 : {array} : default random vector
        Initial guess.
    tol : {float} : default 1e-5
        Stopping criteria: relative residual r[k]/r[0] tolerance.
    maxiter : {int} : default 400
        Stopping criteria: maximum number of allowable iterations.
    return_solver : {bool} : default False
        True: return the solver generated for future use
    solver : {smoothed_aggregation_solver} : default None
        If instance of a multilevel solver, then solver is used 
        to invert A, thus saving time on setup cost.
    verb : {bool}
        If True, verbose output during runtime information

    Returns
    -------
    x : {array}
        Solution to Ax = b   

    Notes
    -----
    If calling solveit(...) multiple times for the same matrix, A, solver reuse is
    easy and efficient.  Set "return_solver=True", and the return value will be
    a tuple, (x,ml), where ml is the solver used to invert A, and x is the solution 
    to Ax=b.  Then, the next time solveit(...) is called, set "solver=ml". 

    Examples
    --------
    >>> from numpy import arange, array             
    >>> from pyamg import solveit                   
    >>> from pyamg.gallery import poisson           
    >>> from pyamg.util.linalg import norm          
    >>> A = poisson((40,40),format='csr')           
    >>> b = array(arange(A.shape[0]), dtype=float) 
    >>> x = solveit(A,b,verb=False)                 
    >>> print norm(b - A*x)/norm(b)                 
    6.27746173733e-06

    """
    
    ##
    # Convert to CSR or BSR if necessary
    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        try:
            A = csr_matrix(A)
            print 'Implicit conversion of A to CSR in pyamg.smoothed_aggregation_solver'
        except:
            raise TypeError('Argument A must have type csr_matrix or bsr_matrix,\
                             or be convertible to csr_matrix')
    #
    A = A.asfptype()
    
    ##
    # Detect symmetry
    if ishermitian(A, fast_check=True):
        symmetry = 'hermitian'
    else:
        symmetry = 'nonsymmetric'
    #
    if verb:
        print "  Detected a " + symmetry + " matrix"

    ##
    # Symmetry dependent parameters
    if symmetry == 'hermitian':
        smooth = ('energy', {'krylov':'cg', 'maxiter':3, 'degree':2, 'weighting':'local'})
        accel = 'cg'
        prepost = ('block_gauss_seidel', {'sweep':'symmetric', 'iterations':1})
    else:
        smooth = ('energy', {'krylov':'gmres','maxiter':3,'degree':2,'weighting':'local'})
        accel = 'gmres'
        prepost = ('gauss_seidel_nr', {'sweep':'symmetric', 'iterations':2})

    ##
    # Generate solver if necessary
    if solver == None:
       
        ##
        # B is the constant for each variable in a node
        if isspmatrix_bsr(A) and A.blocksize[0] > 1:
            bsize = A.blocksize[0]
            B = numpy.kron(numpy.ones((A.shape[0]/bsize,1), dtype=A.dtype), numpy.eye(bsize))
        else:
            B = numpy.ones((A.shape[0],1), dtype=A.dtype)
        
        if symmetry == 'hermitian':
            BH = None
        else:
            BH = B.copy()
        
        solver = smoothed_aggregation_solver(A, B=B, BH=BH, smooth=smooth,
                 strength=('evolution', {'k':2, 'proj_type':'l2', 'epsilon':3.0}),
                 max_levels=15, max_coarse=500, coarse_solver='pinv', 
                 symmetry=symmetry, aggregate='standard', presmoother=prepost,
                 postsmoother=prepost, keep=False)
    
    else:
        if solver.levels[0].A.shape[0] != A.shape[0]:
            raise TypeError('Argument solver must have level 0 matrix of same size as A') 
        
    ##
    # Initial guess
    if x0 == None:
        x0 = scipy.rand(A.shape[0],)
    
    ##
    # Callback function to print iteration number
    if verb:
        iteration = numpy.zeros((1,))
        def callback(x,iteration):
            iteration[0] = iteration[0] + 1
            print "    iteration %d, maxiter = %d"%(iteration[0],maxiter)
        #
        callback2 = lambda x : callback(x,iteration)
    else:
        callback2 = None
    
    ##
    # Solve with accelerated Krylov method
    x = solver.solve(b, x0=x0, accel=accel, tol=tol, maxiter=maxiter, callback=callback2) 
    if verb:
        r0 = norm( numpy.ravel(b) - numpy.ravel(A*x0) )
        rk = norm( numpy.ravel(b) - numpy.ravel(A*x) )
        if rk != 0.0:
            print "  Residual reduction ||r_k||/||r_0|| = %1.2e"%(rk/r0)
        else: 
            print "  Residuals ||r_k||, ||r_0|| = %1.2e, %1.2e"%(rk,r0)
    
    if return_solver:
        return (x.reshape(b.shape), solver)
    else:
        return x.reshape(b.shape)

