from numpy import array, zeros, ones, sqrt, ravel, abs, max, dot, arange, conjugate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.sputils import upcast
from scipy import hstack, ceil, isnan, isinf
from scipy.linalg import lu_solve
from warnings import warn
from pyamg.util.linalg import norm
import scipy.sparse

__docformat__ = "restructuredtext en"

__all__ = ['fgmres']

def mysign(x):
    if x == 0.0:
        return 1.0
    else:
        # return the complex "sign"
        return x/abs(x)

def fgmres(A, b, x0=None, tol=1e-5, restrt=None, maxiter=None, xtype=None, M=None, callback=None, residuals=None):
    '''
    Flexible Generalized Minimum Residual Method (fGMRES)
        fGMRES iteratively refines the initial solution guess to the system Ax = b
    For robustness, Householder reflections are used to orthonormalize the Krylov Space
    Givens Rotations are used to provide the residual norm each iteration
    Flexibility implies that the right preconditioner, M or A.psolve, can vary from 
    iteration to iteration

    Parameters
    ----------
    A : array, matrix or sparse matrix
        n x n, linear system to solve
    b : array
        n x 1, right hand side
    x0 : array-like
        n x 1, initial guess
        default is a vector of zeros
    tol : float
        convergence tolerance
    restrt : int
        number of restarts
        total iterations = restrt*maxiter
    maxiter : int
        maximum number of allowed inner iterations
    xtype : type
        dtype for the solution
    M : matrix-like
        n x n, inverted preconditioner, i.e. solve A M x = b.
        M need not be static for fgmres
        For preconditioning with a mat-vec routine, set
        A.psolve = func, where func := M y
    callback : function
        callback( ||resid||_2 ) is called each iteration, 
    residuals : {None, empty-list}
        If empty-list, residuals holds the residual norm history,
        including the initial residual, upon completion
     
    Returns
    -------    
    (xNew, info)
    xNew -- an updated guess to the solution of Ax = b
    info -- halting status of gmres
            0  : successful exit
            >0 : convergence to tolerance not achieved,
                 return iteration count instead.  This value
                 is precisely the order of the Krylov space.
            <0 : numerical breakdown, or illegal input


    Notes
    -----
    
    Examples
    --------
    >>>from pyamg.krylov import *
    >>>from scipy import rand
    >>>import pyamg
    >>>A = pyamg.poisson((50,50))
    >>>b = rand(A.shape[0],)
    >>>(x,flag) = fgmres(A,b)
    >>>print pyamg.util.linalg.norm(b - A*x)

    References
    ----------
    Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
    Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003

    '''
    
    # Convert inputs to linear system, with error checking  
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype)
    dimen = A.shape[0]

    # Choose type
    xtype = upcast(A.dtype, x.dtype, b.dtype, M.dtype)

    # We assume henceforth that shape=(n,) for all arrays
    b = ravel(array(b,xtype))
    x = ravel(array(x,xtype))
    
    # Should norm(r) be kept
    if residuals == []:
        keep_r = True
    else:
        keep_r = False

    # check number of iterations
    if restrt == None:
        restrt = 1
    elif restrt < 1:
        raise ValueError('Number of restarts must be positive')

    if maxiter == None:
        maxiter = int(max(ceil(dimen/restrt)))
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')
    elif maxiter > dimen:
        warn('maximimum allowed inner iterations (maxiter) are the number of degress of freedom')
        maxiter = dimen

    # Scale tol by normb
    normb = norm(b) 
    if normb == 0:
        pass
    #    if callback != None:
    #        callback(0.0)
    #
    #    return (postprocess(zeros((dimen,), dtype=xtype)), 0)
    else:
        tol = tol*normb
   
    # Is this a one dimensional matrix?
    if dimen == 1:
        entry = ravel(A*array([1.0], dtype=xtype))
        return (postprocess(b/entry), 0)

    # Prep for method
    r = b - ravel(A*x)
    normr = norm(r)
    if keep_r:
        residuals.append(normr)

    # Is initial guess sufficient?
    if normr <= tol:
        if callback != None:    
            callback(norm(r))
        
        return (postprocess(x), 0)

    # Use separate variable to track iterations.  If convergence fails, we cannot 
    # simply report niter = (outer-1)*maxiter + inner.  Numerical error could cause 
    # the inner loop to halt before reaching maxiter while the actual ||r|| > tol.
    niter = 0

    # Begin fGMRES
    for outer in range(restrt):

        # Calculate vector w, which defines the Householder reflector
        #    Take shortcut in calculating, 
        #    w = r + sign(r[1])*||r||_2*e_1
        w = r 
        beta = mysign(w[0])*normr
        w[0] += beta
        w = w / norm(w)
    
        # Preallocate for Krylov vectors, Householder reflectors and Hessenberg matrix
        # Space required is O(dimen*maxiter)
        H = zeros( (maxiter, maxiter), dtype=xtype)         # upper Hessenberg matrix (actually made upper tri with Given's Rotations) 
        W = zeros( (dimen, maxiter), dtype=xtype)           # Householder reflectors
        Z = zeros( (dimen, maxiter), dtype=xtype)           # For fGMRES, preconditioned vectors must be stored
                                                            #     No Horner-like scheme exists that allow us to avoid this
        W[:,0] = w
    
        # Multiply r with (I - 2*w*w.T), i.e. apply the Householder reflector
        # This is the RHS vector for the problem in the Krylov Space
        g = zeros((dimen,), dtype=xtype) 
        g[0] = -beta
    
        for inner in range(maxiter):
            # Calcute Krylov vector in two steps
            # (1) Calculate v = P_j = (I - 2*w*w.T)v, where k = inner
            v = -2.0*conjugate(w[inner])*w
            v[inner] += 1.0
            # (2) Calculate the rest, v = P_1*P_2*P_3...P_{j-1}*ej.
            for j in range(inner-1,-1,-1):
                v = v - 2.0*dot(conjugate(W[:,j]), v)*W[:,j]
            
            #Apply preconditioner
            v = ravel(M*v)
            # Check for nan, inf    
            if any(isnan(v)) or any(isinf(v)):
                warn('inf or nan after application of preconditioner')
                return(postprocess(x), -1)
            Z[:,inner] = v

            # Calculate new search direction
            v = ravel(A*v)

            # Factor in all Householder orthogonal reflections on new search direction
            for j in range(inner+1):
                v = v - 2.0*dot(conjugate(W[:,j]), v)*W[:,j]
                  
            # Calculate next Householder reflector, w
            #  w = v[inner+1:] + sign(v[inner+1])*||v[inner+1:]||_2*e_{inner+1)
            #  Note that if maxiter = dimen, then this is unnecessary for the last inner 
            #     iteration, when inner = dimen-1.  Here we do not need to calculate a
            #     Householder reflector or Given's rotation because nnz(v) is already the
            #     desired length, i.e. we do not need to zero anything out.
            if inner != dimen-1:
                w = zeros((dimen,), dtype=xtype)
                vslice = v[inner+1:]
                alpha = norm(vslice)
                if alpha != 0:
                    alpha = mysign(vslice[0])*alpha
                    # We do not need the final reflector for future calculations
                    if inner < (maxiter-1):
                        w[inner+1:] = vslice
                        w[inner+1] += alpha
                        w = w / norm(w)
                        W[:,inner+1] = w
      
                    # Apply new reflector to v
                    #  v = v - 2.0*w*(w.T*v)
                    v[inner+1] = -alpha
                    v[inner+2:] = 0.0
            
            # Apply all previous Given's Rotations to v
            if inner == 0:
                # Q will store the cumulative effect of all Given's Rotations
                Q = scipy.sparse.eye(dimen, dimen, format='csr', dtype=xtype)

                # Declare initial Qj, which will be the current Given's Rotation
                rowptr  = hstack( (array([0, 2, 4],int), arange(5,dimen+3,dtype=int)) )
                colindices = hstack( (array([0, 1, 0, 1],int), arange(2, dimen,dtype=int)) )
                data = ones((dimen+2,), dtype=xtype)
                Qj = csr_matrix( (data, colindices, rowptr), shape=(dimen,dimen), dtype=xtype)
            else: 
                # Could avoid building a global Given's Rotation, by storing 
                # and applying each 2x2 matrix individually.
                # But that would require looping, the bane of wonderful Python
                Q = Qj*Q
                v = Q*v
      
            # Calculate Qj, the next Given's rotation, where j = inner
            #  Note that if maxiter = dimen, then this is unnecessary for the last inner 
            #     iteration, when inner = dimen-1.  Here we do not need to calculate a
            #     Householder reflector or Given's rotation because nnz(v) is already the
            #     desired length, i.e. we do not need to zero anything out.
            if inner != dimen-1:
                if v[inner+1] != 0:
                    # Calculate terms for complex 2x2 Given's Rotation
                    # Note that abs(x) takes the complex modulus
                    h1 = v[inner]; h2 = v[inner+1];
                    h1_mag = abs(h1); h2_mag = abs(h2);
                    if h1_mag < h2_mag:
                        mu = h1/h2
                        tau = conjugate(mu)/abs(mu)
                    else:    
                        mu = h2/h1
                        tau = mu/abs(mu)

                    denom = sqrt( h1_mag**2 + h2_mag**2 )               
                    c = h1_mag/denom; s = h2_mag*tau/denom; 
                    Qblock = array([[c, conjugate(s)], [-s, c]], dtype=xtype) 
                    
                    # Modify Qj in csr-format so that it represents the current 
                    #   global Given's Rotation equivalent to Qblock
                    if inner != 0:
                        Qj.data[inner-1] = 1.0
                        Qj.indices[inner-1] = inner-1
                        Qj.indptr[inner-1] = inner-1
                    
                    Qj.data[inner:inner+4] = ravel(Qblock)
                    Qj.indices[inner:inner+4] = [inner, inner+1, inner, inner+1]
                    Qj.indptr[inner:inner+3] = [inner, inner+2, inner+4]
                    
                    # Apply Given's Rotation to g, 
                    #   the RHS for the linear system in the Krylov Subspace.
                    #   Note that this dot does a matrix multiply, not an actual
                    #   dot product where a conjugate transpose is taken
                    g[inner:inner+2] = dot(Qblock, g[inner:inner+2])
                    
                    # Apply effect of Given's Rotation to v
                    v[inner] = dot(Qblock[0,:], v[inner:inner+2]) 
                    v[inner+1] = 0.0
            
            # Write to upper Hessenberg Matrix,
            #   the LHS for the linear system in the Krylov Subspace
            H[:,inner] = v[0:maxiter]
      
            # Don't update normr if last inner iteration, because 
            # normr is calculated directly after this loop ends.
            if inner < maxiter-1:
                normr = abs(g[inner+1])
                if normr < tol:
                    break
                
                # Allow user access to residual
                if callback != None:
                    callback( normr )
                if keep_r:
                    residuals.append(normr)

            niter += 1
                    
        # end inner loop, back to outer loop

        # Find best update to x in Krylov Space, V.  Solve inner+1 x inner+1 system.
        #   Apparently this is the best way to solve a triangular
        #   system in the magical world of scipy
        piv = arange(inner+1)
        y = lu_solve((H[0:(inner+1),0:(inner+1)], piv), g[0:(inner+1)], trans=0)
        
        # No Horner like scheme exists because the preconditioner can change each iteration
        # Hence, we must store each preconditioned vector
        update = dot(Z[:,0:inner+1], y)
        x = x + update
        r = b - ravel(A*x)
        normr = norm(r)

        # Allow user access to residual
        if callback != None:
            callback( normr )
        if keep_r:
            residuals.append(normr)
        
        # Has fGMRES stagnated?
        indices = (x != 0)
        if indices.any():
            change = max(abs( update[indices] / x[indices] ))
            if change < 1e-12:
                # No change, halt
                return (postprocess(x), -1)
        
        # test for convergence
        if normr < tol:
            return (postprocess(x),0)
    
    # end outer loop
    
    return (postprocess(x), niter)

