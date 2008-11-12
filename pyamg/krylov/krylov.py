from numpy import array, zeros, ones, sqrt, ravel, inner, abs, max, dot, arange, eye, conjugate
from scipy.sparse import csr_matrix, isspmatrix
from scipy.sparse.linalg.isolve.utils import make_system
from scipy import mat, hstack, cos, sin, ceil, inf, nan, any, sign, rand, diag, isnan, isinf, float32, float64
from scipy.linalg import lu_solve
from Calc_NormResidual import Calc_NormResidual
from warnings import warn
import scipy.sparse

__all__ = ['cgne', 'cgnr', 'gmres', 'fgmres', 'test_solver', 'test_complex_solver', 'van_der_vorst_plots', 'compare_krylov', 'compare_complex_krylov', 'test_fgmres', 'test_sparse_solver']

def mynorm(x):
    '''
    2-norm of x, because Scipy developers never thought that the library norm function should be fast.
    ALERT: x is assumed to be ravel(x)
    '''
    return sqrt( inner(conjugate(x), x) )

def mysign(x):
    if x == 0.0:
        return 1.0
    else:
        # return the complex "sign"
        return x/abs(x)

def cgnr(A, b, x0=None, tol=1e-5, maxiter=None, xtype=None, M=None, callback=None):
    '''
    Conjugate Gradient, Normal Residual algorithm
    Applies CG to the normal equations, A.H A x = b
    Left preconditioning is supported
    Note that if A is not well-conditioned, this algorithm is unadvisable

    Inputs:
    
    A {sparse matrix}
        n x n, linear system to solve
    b {array-like}
        n x 1, right hand side
    x0 {array-like}
        n x 1, initial guess
        default is a vector of zeros
    tol {float}
        convergence tolerance
    maxiter {int}
        maximum number of allowed iterations
        default is 
    xtype {type}
        dtype for the solution
    M {matrix-like}
        n x n, inverted preconditioner, i.e. solve M A.H A x = b.
        For preconditioning with a mat-vec routine, set
        A.psolve = func, where func := M y
    callback {function}
        callback( r ) is called each iteration, 
        where r = b - A*x 
     
    Outputs:
    
    returns (xNew, info)
        xNew -- an updated guess to the solution of Ax = b
        info -- halting status of gmres
                0  : successful exit
                >0 : convergence to tolerance not achieved,
                     return iteration count instead.  
                <0 : numerical breakdown, or illegal input


    References:

    Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
    Second Edition", SIAM, pp. 276-7, 2003

    '''
    # Note that dense systems do not have the rmatvec attribute, so
    # this code is only appropriate for sparse matrices

    # Use this work around, until rmatvec(x, conjugate=True) is fixed.
    # Currently, conjugate=True copies the matrix for every call to rmatvec(..)
    AH = A.tocsr().conjugate().transpose()
    rmatvec = AH.matvec

    # Convert inputs to linear system.  
    # Make_system (1) checks for compatible dimensions, and  
    # (2) Returns an M such that M=(None | array) and 
    # A.psolve = precon.mat-vec is handled correctly 
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype)
    dimen = A.shape[0]
    matvec = A.matvec
    #rmatvec = A.rmatvec
    psolve = M.matvec
    
    # "Intelligent" type checking, where some inputs may be complex and others may not
    if (A.dtype == complex) or (x.dtype == complex) or (b.dtype == complex) or (M.dtype == complex):
        if A.dtype == complex:
            xtype = A.dtype
        elif M.dtype == complex:
            xtype = M.dtype
        elif b.dtype == complex:
            xtype = b.dtype
        else:
            xtype = x.dtype
    else:
        # All components are real
        xtype = A.dtype

    # This function assumes shape=(n,) arrays
    b = ravel(array(b,xtype))
    x = ravel(array(x,xtype))
    
    # Check iteration numbers
    # CGNE suffers from loss of orthogonality quite easily, so we arbitarily let the method go up to 130% over the
    # theoretically necessary limit of maxiter=dimen
    if maxiter == None:
        maxiter = int(ceil(1.3*dimen)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')
    elif maxiter > (1.3*dimen):
        warn('maximimum allowed inner iterations (maxiter) are the 130% times the number of degress of freedom')
        maxiter = int(ceil(1.3*dimen)) + 2

    # Is RHS all zero?
    normb = mynorm(b) 
    if normb == 0:
        if callback != None:
            callback(0.0)
    
        return (zeros((dimen,), dtype=xtype),0)
   
    # Prep for method
    r = b - ravel(matvec(x))
    rhat = rmatvec(r)
    tol = tol*normb

    # Is initial guess sufficient?
    if mynorm(r) <= tol:
        if callback != None:    
            callback(r)
        
        return (x, 0)
   
    # Begin CGNR

    # Apply preconditioner and calculate initial search direction
    z = ravel(psolve(rhat))
    p = z.copy()
    old_zr = dot(conjugate(z), rhat)

    for iter in range(maxiter):

        # w_j = A p_j
        w = ravel(matvec(p))

        # alpha = (z_j, rhat_j) / (w_j, w_j)
        alpha = old_zr/dot(conjugate(w), w)
        
        # x_{j+1} = x_j + alpha*p_j
        x = x + alpha*p

        # r_{j+1} = r_j - alpha*w_j
        r = r - alpha*w

        # rhat_{j+1} = A.H*r_{j+1}
        rhat = ravel(rmatvec(r))

        # z_{j+1} = M*r_{j+1}
        z = ravel(psolve(rhat))

        # beta = (z_{j+1}, rhat_{j+1}) / (z_j, rhat_j)
        new_zr = dot(conjugate(z), rhat)
        beta = new_zr / old_zr
        old_zr = new_zr

        # p_{j+1} = A.H*z_{j+1} + beta*p_j
        p = z + beta*p

        # Allow user access to residual
        if callback != None:
            callback( r )
        
        # test for convergence
        if mynorm(r) < tol:
            return (x,0)

    # end loop

    return (x, iter+1)

def cgne(A, b, x0=None, tol=1e-5, maxiter=None, xtype=None, M=None, callback=None):
    '''
    Conjugate Gradient, Normal Error algorithm
    Applies CG to the normal equations, A A.H x = b
    Left preconditioning is supported
    Note that if A is not well-conditioned, this algorithm is unadvisable

    Inputs:
    
    A {sparse matrix}
        n x n, linear system to solve
    b {array-like}
        n x 1, right hand side
    x0 {array-like}
        n x 1, initial guess
        default is a vector of zeros
    tol {float}
        convergence tolerance
    maxiter {int}
        maximum number of allowed iterations
        default is 
    xtype {type}
        dtype for the solution
    M {matrix-like}
        n x n, inverted preconditioner, i.e. solve M A A.H x = b.
        For preconditioning with a mat-vec routine, set
        A.psolve = func, where func := M y
    callback {function}
        callback( r ) is called each iteration, 
        where r = b - A*x 
     
    Outputs:
    
    returns (xNew, info)
        xNew -- an updated guess to the solution of Ax = b
        info -- halting status of gmres
                0  : successful exit
                >0 : convergence to tolerance not achieved,
                     return iteration count instead.  
                <0 : numerical breakdown, or illegal input


    References:

    Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
    Second Edition", SIAM, pp. 276-7, 2003

    '''
    # Note that dense systems do not have the rmatvec attribute, so
    # this code is only appropriate for sparse matrices

    # Use this work around, until rmatvec(x, conjugate=True) is fixed.
    # Currently, conjugate=True copies the matrix for every call to rmatvec(..)
    AH = A.tocsr().conjugate().transpose()
    rmatvec = AH.matvec

    # Convert inputs to linear system.  
    # Make_system (1) checks for compatible dimensions, and  
    # (2) Returns an M such that M=(None | array) and 
    # A.psolve = precon.mat-vec is handled correctly 
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype)
    dimen = A.shape[0]
    matvec = A.matvec
    #rmatvec = A.rmatvec
    psolve = M.matvec
    
    # "Intelligent" type checking, where some inputs may be complex and others may not
    if (A.dtype == complex) or (x.dtype == complex) or (b.dtype == complex) or (M.dtype == complex):
        if A.dtype == complex:
            xtype = A.dtype
        elif M.dtype == complex:
            xtype = M.dtype
        elif b.dtype == complex:
            xtype = b.dtype
        else:
            xtype = x.dtype
    else:
        # All components are real
        xtype = A.dtype

    # This function assumes shape=(n,) arrays
    b = ravel(array(b,xtype))
    x = ravel(array(x,xtype))
    
    # Check iteration numbers
    # CGNE suffers from loss of orthogonality quite easily, so we arbitarily let the method go up to 130% over the
    # theoretically necessary limit of maxiter=dimen
    if maxiter == None:
        maxiter = int(ceil(1.3*dimen)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')
    elif maxiter > (1.3*dimen):
        warn('maximimum allowed inner iterations (maxiter) are the 130% times the number of degress of freedom')
        maxiter = int(ceil(1.3*dimen)) + 2

    # Is RHS all zero?
    normb = mynorm(b) 
    if normb == 0:
        if callback != None:
            callback(0.0)
    
        return (zeros((dimen,), dtype=xtype),0)
   
    # Prep for method
    r = b - ravel(matvec(x))
    tol = tol*normb

    # Is initial guess sufficient?
    if mynorm(r) <= tol:
        if callback != None:    
            callback(r)
        
        return (x, 0)
   
    # Begin CGNE

    # Apply preconditioner and calculate initial search direction
    z = ravel(psolve(r))
    p = ravel(rmatvec(z))
    old_zr = dot(conjugate(z), r)

    for iter in range(maxiter):

        # alpha = (z_j, r_j) / (p_j, p_j)
        alpha = old_zr/dot(conjugate(p), p)
        
        # x_{j+1} = x_j + alpha*p_j
        x = x + alpha*p

        # r_{j+1} = r_j - alpha*w_j,   where w_j = A*p_j
        r = r - alpha*ravel(matvec(p))

        # z_{j+1} = M*r_{j+1}
        z = ravel(psolve(r))

        # beta = (z_{j+1}, r_{j+1}) / (z_j, r_j)
        new_zr = dot(conjugate(z), r)
        beta = new_zr / old_zr
        old_zr = new_zr

        # p_{j+1} = A.H*z_{j+1} + beta*p_j
        p = ravel(rmatvec(z)) + beta*p

        # Allow user access to residual
        if callback != None:
            callback( r )
        
        # test for convergence
        if mynorm(r) < tol:
            return (x,0)

    # end loop

    return (x, iter+1)


def fgmres(A, b, x0=None, tol=1e-5, restrt=None, maxiter=None, xtype=None, M=None, callback=None):
    '''
    Flexible Generalized Minimum Residual Method (GMRES)
        GMRES iteratively refines the initial solution guess to the system Ax = b
    For robustness, Householder reflections are used to orthonormalize the Krylov Space
    Givens Rotations are used to provide the residual norm each iteration
    Flexibility implies that the right preconditioner, M or A.psolve, can vary from 
    iteration to iteration

    Inputs:
    
    A { matrix-like}
        n x n, linear system to solve
    b {array-like}
        n x 1, right hand side
    x0 {array-like}
        n x 1, initial guess
        default is a vector of zeros
    tol {float}
        convergence tolerance
    restrt {int}
        number of restarts
        total iterations = restrt*maxiter
    maxiter {int}
        maximum number of allowed iterations
        default is 
    xtype {type}
        dtype for the solution
    M {matrix-like}
        n x n, inverted preconditioner, i.e. solve A M x = b.
        M need not be static for fgmres
        For preconditioning with a mat-vec routine, set
        A.psolve = func, where func := M y
    callback {function}
        callback( ||resid||_2 ) is called each iteration, 
     
    Outputs:
    
    returns (xNew, info)
        xNew -- an updated guess to the solution of Ax = b
        info -- halting status of gmres
                0  : successful exit
                >0 : convergence to tolerance not achieved,
                     return iteration count instead.  This value
                     is precisely the order of the Krylov space.
                <0 : numerical breakdown, or illegal input


    References:

    Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
    Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003

    '''
    
    # Convert inputs to linear system.  
    # Make_system (1) checks for compatible dimensions, and  
    # (2) Returns an M such that M=(None | array) and 
    # A.psolve = precon.mat-vec is handled correctly 
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype)
    dimen = A.shape[0]
    matvec = A.matvec
    psolve = M.matvec

    # "Intelligent" type checking, where some inputs may be complex and others may not
    if (A.dtype == complex) or (x.dtype == complex) or (b.dtype == complex) or (M.dtype == complex):
        if A.dtype == complex:
            xtype = A.dtype
        elif M.dtype == complex:
            xtype = M.dtype
        elif b.dtype == complex:
            xtype = b.dtype
        else:
            xtype = x.dtype
    else:
        # All components are real
        xtype = A.dtype

    # This function assumes shape=(n,) arrays
    b = ravel(array(b,xtype))
    x = ravel(array(x,xtype))
    
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

    # Is RHS all zero?
    normb = mynorm(b) 
    if normb == 0:
        if callback != None:
            callback(0.0)
    
        return (zeros((dimen,), dtype=xtype),0)
   
    # Is this a one dimensional matrix?
    if dimen == 1:
        entry = ravel(matvec(array([1.0], dtype=xtype)))
        return (b/entry, 0)

    # Prep for method
    r = b - ravel(matvec(x))
    normr = mynorm(r)
    tol = tol*normb

    # Is initial guess sufficient?
    if normr <= tol:
        if callback != None:    
            callback(mynorm(r))
        
        return (x, 0)

    # # Allow user access to residual.  Note that this is
    # # the preconditioned residual, not || b - Ax ||_2
    # if callback != None:    
    #    callback(normr)

    # Use separate variable to track iterations.  If convergence fails, we cannot 
    # simply report niter = (outer-1)*maxiter + inner.  Numerical error could cause 
    # the inner loop to halt before reaching maxiter while the actual ||r|| > tol.
    niter = 0

    # Begin FGMRES
    for outer in range(restrt):

        # Calculate vector w, which defines the Householder reflector
        #    Take shortcut in calculating, 
        #    w = r + sign(r[1])*||r||_2*e_1
        w = r 
        beta = mysign(w[0])*normr
        w[0] += beta
        w = w / mynorm(w)
    
        # Preallocate for Krylov vectors, Householder reflectors and Hessenberg matrix
        # Space required is O(dimen*maxiter)
        H = zeros( (maxiter, maxiter), dtype=xtype)         # upper Hessenberg matrix (actually made upper tri with Given's Rotations) 
        W = zeros( (dimen, maxiter), dtype=xtype)           # Householder reflectors
        Z = zeros( (dimen, maxiter), dtype=xtype)           # For flexible GMRES, preconditioned vectors must be stored
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
            v = ravel(psolve(v))
            # Check for nan, inf    
            if any(isnan(v)) or any(isinf(v)):
                warn('inf or nan after application of preconditioner')
                return(x, -1)
            Z[:,inner] = v

            # Calculate new search direction
            v = ravel(matvec(v))

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
                alpha = mynorm(vslice)
                if alpha != 0:
                    alpha = mysign(vslice[0])*alpha
                    # We do not need the final reflector for future calculations
                    if inner < (maxiter-1):
                        w[inner+1:] = vslice
                        w[inner+1] += alpha
                        w = w / mynorm(w)
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
        r = b - ravel(matvec(x))
        normr = mynorm(r)

        # test for convergence
        if normr < tol:
            if callback != None:
                callback( normr )
            
            return (x,0)

        # Allow user access to residual
        if callback != None:
            callback( normr )
        
        # Has GMRES stagnated?
        indices = (x != 0)
        if indices.any():
            change = max(abs( update[indices] / x[indices] ))
            if change < 1e-12:
                # No change, halt
                return (x, -1)
    
    # end outer loop
    
    return (x, niter)



def gmres(A, b, x0=None, tol=1e-5, restrt=None, maxiter=None, xtype=None, M=None, callback=None):
    '''
    Generalized Minimum Residual Method (GMRES)
        GMRES iteratively refines the initial solution guess to the system Ax = b
    For robustness, Householder reflections are used to orthonormalize the Krylov Space
    Givens Rotations are used to provide the residual norm each iteration

    Inputs:
    
    A { matrix-like}
        n x n, linear system to solve
    b {array-like}
        n x 1, right hand side
    x0 {array-like}
        n x 1, initial guess
        default is a vector of zeros
    tol {float}
        convergence tolerance
    restrt {int}
        number of restarts
        total iterations = restrt*maxiter
    maxiter {int}
        maximum number of allowed iterations
        default is 
    xtype {type}
        dtype for the solution
    M {matrix-like}
        n x n, inverted preconditioner, i.e. solve M A x = b.
        For preconditioning with a mat-vec routine, set
        A.psolve = func, where func := M y
    callback {function}
        callback( ||resid||_2 ) is called each iteration, 
     
    Outputs:
    
    returns (xNew, info)
        xNew -- an updated guess to the solution of Ax = b
        info -- halting status of gmres
                0  : successful exit
                >0 : convergence to tolerance not achieved,
                     return iteration count instead.  This value
                     is precisely the order of the Krylov space.
                <0 : numerical breakdown, or illegal input


    References:

    Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
    Second Edition", SIAM, pp. 151-172, pp. 272-275, 2003

    '''
    
    # Convert inputs to linear system.  
    # Make_system (1) checks for compatible dimensions, and  
    # (2) Returns an M such that M=(None | array) and 
    # A.psolve = precon.mat-vec is handled correctly 
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype)
    dimen = A.shape[0]
    matvec = A.matvec
    psolve = M.matvec

    # "Intelligent" type checking, where some inputs may be complex and others may not
    if (A.dtype == complex) or (x.dtype == complex) or (b.dtype == complex) or (M.dtype == complex):
        if A.dtype == complex:
            xtype = A.dtype
        elif M.dtype == complex:
            xtype = M.dtype
        elif b.dtype == complex:
            xtype = b.dtype
        else:
            xtype = x.dtype
    else:
        # All components are real
        xtype = A.dtype

    # This function assumes shape=(n,) arrays
    b = ravel(array(b,xtype))
    x = ravel(array(x,xtype))
    
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

    # Is RHS all zero?
    normb = mynorm(b) 
    if normb == 0:
        if callback != None:
            callback(0.0)
    
        return (zeros((dimen,), dtype=xtype),0)
   
    # Is this a one dimensional matrix?
    if dimen == 1:
        entry = ravel(matvec(array([1.0], dtype=xtype)))
        return (b/entry, 0)

    # Prep for method
    r = b - ravel(matvec(x))
    tol = tol*normb
    
    # Is initial guess sufficient?
    if mynorm(r) <= tol:
        if callback != None:    
            callback(mynorm(r))
        
        return (x, 0)

    #Apply preconditioner
    r = ravel(psolve(r))
    normr = mynorm(r)
    # Check for nan, inf    
    if any(isnan(r)) or any(isinf(r)):
        warn('inf or nan after application of preconditioner')
        return(x, -1)
    
    # # Allow user access to residual.  Note that this is
    # # the preconditioned residual, not || b - Ax ||_2
    # if callback != None:    
    #    callback(normr)

    # Use separate variable to track iterations.  If convergence fails, we cannot 
    # simply report niter = (outer-1)*maxiter + inner.  Numerical error could cause 
    # the inner loop to halt before reaching maxiter while the actual ||r|| > tol.
    niter = 0

    # Begin GMRES
    for outer in range(restrt):

        # Calculate vector w, which defines the Householder reflector
        #    Take shortcut in calculating, 
        #    w = r + sign(r[1])*||r||_2*e_1
        w = r 
        beta = mysign(w[0])*normr
        w[0] += beta
        w = w / mynorm(w)
    
        # Preallocate for Krylov vectors, Householder reflectors and Hessenberg matrix
        # Space required is O(dimen*maxiter)
        H = zeros( (maxiter, maxiter), dtype=xtype)         # upper Hessenberg matrix (actually made upper tri with Given's Rotations) 
        W = zeros( (dimen, maxiter), dtype=xtype)           # Householder reflectors
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
            
            # Calculate new search direction
            v = ravel(matvec(v))

            #Apply preconditioner
            v = ravel(psolve(v))
            # Check for nan, inf    
            if any(isnan(v)) or any(isinf(v)):
                warn('inf or nan after application of preconditioner')
                return(x, -1)

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
                alpha = mynorm(vslice)
                if alpha != 0:
                    alpha = mysign(vslice[0])*alpha
                    # We do not need the final reflector for future calculations
                    if inner < (maxiter-1):
                        w[inner+1:] = vslice
                        w[inner+1] += alpha
                        w = w / mynorm(w)
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
            
            niter += 1
        
        # end inner loop, back to outer loop

        # Find best update to x in Krylov Space, V.  Solve inner+1 x inner+1 system.
        #   Apparently this is the best way to solve a triangular
        #   system in the magical world of scipy
        piv = arange(inner+1)
        y = lu_solve((H[0:(inner+1),0:(inner+1)], piv), g[0:(inner+1)], trans=0)
        
        # Use Horner like Scheme to map solution, y, back to original space.
        # Note that we do not use the last reflector.
        update = zeros(x.shape, dtype=xtype)
        for j in range(inner,-1,-1):
            update[j] += y[j]
            # Apply j-th reflector, (I - 2.0*w_j*w_j.T)*upadate
            update = update - 2.0*dot(conjugate(W[:,j]), update)*W[:,j]

        x = x + update
        r = b - ravel(matvec(x))

        #Apply preconditioner
        r = ravel(psolve(r))
        normr = mynorm(r)
        # Check for nan, inf    
        if any(isnan(r)) or any(isinf(r)):
            warn('inf or nan after application of preconditioner')
            return(x, -1)
        
        # Allow user access to residual
        if callback != None:
            callback( normr )
        
        # Has GMRES stagnated?
        indices = (x != 0)
        if indices.any():
            change = max(abs( update[indices] / x[indices] ))
            if change < 1e-12:
                # No change, halt
                return (x, -1)

        # test for convergence
        if normr < tol:
            return (x,0)
    
    # end outer loop
    
    return (x, niter)


#################################################################################################
#                              Begin Helper Routines for Tests                                  #
#################################################################################################

# Simple test if krylov method converges
def test_krylov(krylov, A ,b, x0=None, tol=1e-6, restrt=None, maxiter=None, M=None, callback=None, left_precon=True, use_restrt=True):
    from scipy.sparse.linalg.interface import aslinearoperator
    
    if use_restrt:
        (x,flag) = krylov(A, b, x0=x0, tol=tol, restrt=restrt, maxiter=maxiter, M=M, callback=callback)
    else:
        (x,flag) = krylov(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, callback=callback)

    # Generalize A and M to just matvec's
    A = aslinearoperator(A) 
    if M != None:
        M = aslinearoperator(M)
    else:
        M = aslinearoperator(scipy.sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype))
    
    matvec = A.matvec
    psolve = M.matvec
    
    b = b.reshape(-1,1);
    x = x.reshape(-1,1);

    # Check type of x, but have to allow for a real A and a complex x, or vice versa
    if ((A.dtype == float) and (x.dtype == float)) or ((A.dtype == complex) and (x.dtype == complex)):
        if x.dtype != A.dtype:
            print "A and x are of different dtypes"
            assert(0)
    
    # Note that when left preconditioning is used in the method, "krylov", the krylov method 
    # returns the preconditioned residual, not the actual residual
    if left_precon:
        normr = mynorm(ravel(psolve( b - matvec(x).reshape(-1,1) )))
    else:
        normr = mynorm(ravel(b - matvec(x).reshape(-1,1) ))

    normb = mynorm(ravel(b))
    if normb == 0.0:
        normb = 1.0

    # Multiply by 50, because GMRES halts based on preconditioned residual, 
    # not actual residual.  This discrepancy creates some wiggle room for the 
    # actual residual to be a little larger than ||b||_2 * tol.
    if (flag == 0) and (normr > 50*normb*tol):
        print "Krylov method returned with falsely reported convergence, || r ||_2 = %e" % normr
        assert(0)
    elif(normr > 50*normb*tol):
        print "Krylov method did not converge in the maximum allowed number of iterations.  || r ||_2 = %e, flag = %d" % (normr, flag)

    return x

# Simple test if krylov method halts when tolerance is met
# If a system's spectrum is such that the residual is not reduced 
# significantly until the final iteration, as is the case with most 
# random matrices, then this test will fail. 
def test_krylov_picky(krylov, A ,b, x0=None, tol=1e-6, restrt=None, maxiter=None, M=None, callback=None, left_precon=True, use_restrt=True):
    from scipy.sparse.linalg.interface import aslinearoperator
    
    if use_restrt:
        (x,flag) = krylov(A, b, x0=x0, tol=tol, restrt=restrt, maxiter=maxiter, M=M, callback=callback)
    else:
        (x,flag) = krylov(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, callback=callback)
    
    # Generalize A and M to just matvec's
    A = aslinearoperator(A)
    if M != None:
        M = aslinearoperator(M)
    else:
        M = aslinearoperator(scipy.sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype))
    
    matvec = A.matvec
    psolve = M.matvec
    
    b = b.reshape(-1,1);
    x = x.reshape(-1,1);
    
    # Check type of x
    if ((A.dtype == float) and (x.dtype == float)) or ((A.dtype == complex) and (x.dtype == complex)):
        if x.dtype != A.dtype:
            print "A and x are of different dtypes"
            assert(0)

    # Note that GMRES tests for only the preconditioned residual, not the actual residual
    if left_precon:
        normr = mynorm(ravel(psolve( b - matvec(x).reshape(-1,1) )))
    else:
        normr = mynorm(ravel(b - matvec(x).reshape(-1,1) ))
 

    if normr < 0.1*mynorm(ravel(b))*tol:
        print "Krylov method iterated past tolerance, || r ||_2 = %e" % normr
        assert(0)
    
    normb = mynorm(ravel(b))
    if normb == 0.0:
        normb = 1.0

    if  (flag == 0) and (normr > 50*normb*tol):
        print "Krylov method returned with falsely reported convergence, || r ||_2 = %e" % normr
        assert(0)
    elif(normr > 50*normb*tol):
        print "Krylov method did not converge in the maximum allowed number of iterations.  || r ||_2 = %e, flag = %d" % (normr, flag)

    return x

# Use this function to define the mat-vec routine preconditioners tested below
def matvec_precon(x, M=None):
    if isspmatrix(M):
        return M*x
    else:
        return dot(M,x)


#################################################################################################
#                              Begin Real Test Routines                                         #
#################################################################################################

def test_solver(krylov):
    '''
    Test method defined by function ptr, krylov, on a number of test cases
    Assume that the call string of krylov is the same as the library scipy krylov methods
    '''
    
    A = zeros((10,10))
    M = zeros((10,9))
    v1 = zeros((10,1))
    v2 = zeros((11,1))
    
    # Call GMRES with all zero matrices 
    (x,flag) = krylov(A,v1)
    if (not (x == zeros((10,))).all()) or (flag != 0):
        print 'krylov failed with all zero input'
        assert(0)
    
    (x,flag) = krylov(A,v1, x0=v1)
    if (not (x == zeros((10,))).all()) or (flag != 0):
        print 'krylov failed with all zero input'
        assert(0)
    
    (x,flag) = krylov(A,v1, M=A)
    if (not (x == zeros((10,))).all()) or (flag != 0):
        print 'krylov failed with all zero input'
        assert(0)
    
    (x,flag) = krylov(A,v1, M=A, x0=v1)
    if (not (x == zeros((10,))).all()) or (flag != 0):
        print 'krylov failed with all zero input'
        assert(0)
    
    # Try some invalid iteration numbers, invalid dimensions are handled 
    # by the make_system function in scipy.sparse.linalg.isolve.utils 
    try: krylov(A, v1, maxiter=0)
    except ValueError:
        pass
    else:
        print "krylov failed to detect bad maxiter"
        assert(0)
    
    try: krylov(A, v1, restrt=0)
    except ValueError:
        pass
    else:
        print "krylov failed to detect back restrt"
        assert(0)


    
    # Run Numerical Tests
    n_max = 12              #max system size to test

    # Build and test list of random dense matrices
    cases = []
    for i in range(1,n_max):
        cases.append( (mat(rand(i,i)), rand(i,1)) )
        cases.append( (mat(rand(i,i)), zeros((i,))) )

    for i in range(len(cases)):
        test_krylov(krylov, cases[i][0], cases[i][1])
        test_krylov(krylov, cases[i][0], ravel(cases[i][1]))
        test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],1) )
        test_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],) )
    
    # Build and test list of random dense matrices that are single precision
    cases = []
    for i in range(1,n_max):
        cases.append( (mat(rand(i,i), dtype=float32), array(rand(i,1),dtype=float32)) )

    for i in range(len(cases)):
        test_krylov(krylov, cases[i][0], cases[i][1], tol=5e-5)
    
   # Build and test list of random sparse matrices
    cases = []
    for i in range(1,n_max):
        A = rand(i,i); A[ abs(A) < 0.5 ] = 0.0; A = csr_matrix(A); A = A + scipy.sparse.eye(i,i);
        cases.append( (A, rand(i,1)) )

    for i in range(len(cases)):
        test_krylov(krylov, cases[i][0], cases[i][1])
        test_krylov(krylov, cases[i][0], ravel(cases[i][1]))
        test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],1) )
        test_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],) )
 
    # Build and test list of random sparse matrices that are single precision
    cases = []
    for i in range(1,n_max):
        A = rand(i,i); A[ abs(A) < 0.5 ] = 0.0; A = csr_matrix(A, dtype=float32); A = A + scipy.sparse.eye(i,i,dtype=float32);
        cases.append( (A, array(rand(i,1),dtype=float32)) )

    for i in range(len(cases)):
        test_krylov(krylov, cases[i][0], cases[i][1], tol=5e-5)


    # Build and test list of random diagonally dominant dense matrices, with diagonal preconditioner
    cases = []
    for i in range(1,n_max):
        A = mat(rand(i,i)+i*eye(i,i))
        cases.append(( A, rand(i,), mat(diag(1.0/diag(A))) ))
        A = mat(rand(i,i)+i*eye(i,i))
        cases.append(( A, zeros((i,1)), mat(diag(1.0/diag(A))) ))

    # test with matrix and mat-vec routine as the preconditioner
    for i in range(len(cases)):
        a = test_krylov(krylov, cases[i][0], cases[i][1], M=cases[i][2] )
        b = test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, M=cases[i][2] )
        x0 = rand(cases[i][1].shape[0],1)
        c = test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, M=cases[i][2] )
        
        cases[i][0].psolve = lambda x:matvec_precon(x, M=cases[i][2])
        d = test_krylov(krylov, cases[i][0], cases[i][1])
        e = test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12)
        f = test_krylov(krylov, cases[i][0], cases[i][1], x0=x0)
        
        if (mynorm(ravel(a-d)) > 1e-10) or (mynorm(ravel(b-e)) > 1e-10) or (mynorm(ravel(c-f)) > 1e-10):
            print "matrix and mat-vec routine preconditioning yielded different results"
            assert(0)

    # test that a preconditioner (1) speeds up convergence 
    # (2) convergence is halted when tol is satisfied, and iterations go no further
    # This will also test whether callback_fcn works properly
    cases = []
    import pyamg
    for i in range(10,12):
        A = pyamg.poisson( (i,i), format='csr' )
        SA =  pyamg.smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2')
        A.psolve = SA.psolve
        cases.append(( A, rand(A.shape[0],1)    ))

    for i in range(len(cases)):
        A = cases[i][0]; b = cases[i][1]; x0=rand(A.shape[0],1);
        
        # Test with preconditioning
        r = zeros(A.shape[0])
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov_picky(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn)
        r = r[r.nonzero()[0]][1:]
        iters1 = max(r.shape)
        
        # Test w/o preconditioning
        del A.psolve
        r = zeros(A.shape[0])
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn)
        r = r[r.nonzero()[0]][1:]
        iters2 = max(r.shape)

        if (iters2 - iters1) < 20:
            print "Preconditioning not speeding up convergence"
            assert(0)


    # test krylov with different numbers of iterations and restarts.  
    # use a callback function to make sure that the correct number of iterations are taken.
    # here, we write to the next open spot in an array each time callback is called, so that 
    # an array out-of-bounds error will happen if krylov iterates too many times
    cases=[]
    for i in range(10,13):
        A = pyamg.poisson( (i,i), format='csr' )
        cases.append(( A, rand(A.shape[0],1)    ))
    print "\n----- Ignore GMRES convergence failure until further notice ----"
    for i in range(len(cases)):
        A = cases[i][0]; b=cases[i][1]; x0=rand(A.shape[0],1);
        
        restrt=1; maxiter=int(ceil(A.shape[0]/8))
        r = zeros(restrt*maxiter+1)
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, restrt=restrt, maxiter=maxiter, tol=1e-12)
        shape1 = r.shape
        r = r[r.nonzero()[0]][0:]
        shape2 = r.shape
        if shape1 != shape2:
            print "callback used an incorrect number of times"
            assert(0)

        restrt=2; maxiter=int(ceil(A.shape[0]/8))
        r = zeros(restrt*maxiter + 1)
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, restrt=restrt, maxiter=maxiter, tol=1e-12)
        shape1 = r.shape
        r = r[r.nonzero()[0]][0:]
        shape2 = r.shape
        if shape1 != shape2:
            print "callback used an incorrect number of times"
            assert(0)

        restrt=3; maxiter=int(ceil(A.shape[0]/8))
        r = zeros(restrt*maxiter + 1)
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, restrt=restrt, maxiter=maxiter, tol=1e-12)
        shape1 = r.shape
        r = r[r.nonzero()[0]][0:]
        shape2 = r.shape
        if shape1 != shape2:
            print "callback used an incorrect number of times"
            assert(0)

    print "---- Stop Ignoring GMRES convergence failures ----\n"


def van_der_vorst_plots(krylov):
    ''' Run the tests from Henk van der Vorst with method, krylov.
        The tests are from, 
        "Iterative Krylov Methods for Large Linear Systems", pg 79-82
        As of now, all but the second example work. However, his second 
        example is poorly defined, so its unkown if the same problem 
        is being tested below.
    '''
    
    import pylab
    from scipy import vstack
    from scipy.linalg import pinv2
    from scipy.sparse import spdiags
    # Run tests with known residual history from Henk A. van der Vorst
    cases=[]
    x0 = zeros((200,1))
    # Example 1, pg 79, evenly distributed spectrum
    B = mat(spdiags(arange(1,201), [0], 200, 200).todense())
    S = mat(spdiags(vstack( (ones((1,200)), 0.9*ones((1,200))) ), [0,1], 200, 200).todense())
    cases.append(( S*B*mat(pinv2(S)), S*(B*ones((200,1))), x0 ))
    
    # Example 2, pg 80, first two eigenvalues are close
    B = mat(spdiags(arange(1,201), [0], 200, 200).todense())
    B[1,1] = 1.1
    S = mat(spdiags(vstack( (ones((1,200)), 0.9*ones((1,200))) ), [0,1], 200, 200).todense())
    cases.append(( S*B*mat(pinv2(S)), S*(B*ones((200,1))), x0 ))
     
    # Example 3, pg 81, complex conjugate eigenpair
    B = mat(spdiags(arange(1,201), [0], 200, 200).todense())
    B[0,1] = 1.0; B[1,0] = -1.0; B[1,1]=1.0;
    S = mat(spdiags(vstack( (ones((1,200)), 0.9*ones((1,200))) ), [0,1], 200, 200).todense())
    cases.append(( S*B*mat(pinv2(S)), S*(B*ones((200,1))), x0 ))

    # Example 4, pg 82, defective matrix
    B = mat(spdiags(arange(1,201), [0], 200, 200).todense())
    B[0,1] = 1.0; B[1,2] = 1.0; B[1,1] = 1.0; B[2,2] = 1.0;
    S = mat(spdiags(vstack( (ones((1,200)), 0.9*ones((1,200))) ), [0,1], 200, 200).todense())
    cases.append(( S*B*mat(pinv2(S)), S*(B*ones((200,1))), x0 ))
     
    for i in range(len(cases)):
        A = cases[i][0]; b=cases[i][1]; x0=cases[i][2];
        r = zeros(101 + 1)
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        krylov(cases[i][0], cases[i][1], x0=x0, tol=1e-20, restrt=1, maxiter=101, callback=callback_fcn)    
        r = r[r.nonzero()[0]][0:]
        r_ratio= r[1:]/r[0:-1]
        pylab.figure(i+1)
        pylab.plot(array(range(1,max(r_ratio.shape)+1)), r_ratio)
        pylab.xlabel('Iteration Number')
        pylab.ylabel('Residual Norm Ratio')
        pylab.title('Example %d from van der Vorst, pg 79-82' % (i+1))
        pylab.axis( [0, 101, 0.2, 1.0] )

    pylab.show()


def compare_krylov(m1, m2):
    ''' compare two krylov methods, m1 and m2
        conduct timings and compare answers '''
    import pyamg

    # Run a few tests and compare answers
    cases = []
    # Random Dense matrices
    for i in range(10,13):
        cases.append( {'A' : mat(rand(i,i)),   'b' : rand(i,1),  'x0': zeros((i,1)), 'tol' : 1e-5} ) 
        cases.append( {'A' : mat(rand(i,i)),   'x0' : rand(i,1), 'b' : zeros((i,1)), 'tol' : 1e-5} ) 
        cases.append( {'A' : mat(rand(i,i)),   'x0' : rand(i,1), 'b' : rand(i,1),    'tol' : 1e-5} ) 
    
    # Random Sparse Matrices
    for i in range(10,13):
        A  = rand(i,i); A[ abs(A) < 0.5 ] = 0.0; A = csr_matrix(A); A = A + scipy.sparse.eye(i,i);
        A2 = rand(i,i); A2[ abs(A2) < 0.5 ] = 0.0; A2 = csr_matrix(A2); A2 = A2 + scipy.sparse.eye(i,i);
        cases.append( {'A' : A,   'b' : rand(i,1),   'x0' : zeros((i,1)), 'tol' : 1e-5} ) 
        cases.append( {'A' : A2,   'x0' : rand(i,1),  'b' : zeros((i,1)),  'tol' : 1e-5} ) 

    # Preconditioned Poisson Problems
    for i in range(10,13):
        A = pyamg.poisson( (i,i), format='csr' )
        SA =  pyamg.smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2')
        A.psolve = SA.psolve
        cases.append( {'A' : A,   'b' : rand(A.shape[0],1),   'x0' : zeros((A.shape[0],1)),  'tol' : 1e-5} ) 
        cases.append( {'A' : A,   'x0' : rand(A.shape[0],1),  'b'  : zeros((A.shape[0],1)),  'tol' : 1e-16} ) 
        cases.append( {'A' : A,   'x0' : rand(A.shape[0],1),  'b'  : rand(A.shape[0],1),     'tol' : 1e-16} ) 

    # Compare answers
    for i in range(len(cases)):
        (x1,flag1) = m1(cases[i]['A'], cases[i]['b'], x0=cases[i]['x0'], tol=cases[i]['tol'])
        (x2,flag2) = m2(cases[i]['A'], cases[i]['b'], x0=cases[i]['x0'], tol=cases[i]['tol'])
        x1 = ravel(x1); x2 = ravel(x2)

        if flag1 != flag2:
            print "test %d produced difference convergence flags, flag1 = %d, flag2 = %d" % (i, flag1, flag2)

        print "Test %d, || x1-x2 ||_2 = %e" % (i, mynorm(ravel(x1-x2)))

    import time
    # Conduct timings
    # Compare a 10x10 dense matrix
    cases = []
    cases.append( {'name' : '10 x 10 Dense Matrix\nMaxiter = 10',
                   'A' : mat(rand(10,10)),   'b' : rand(10,1),  'x0' : zeros((10,1)), 'maxiter' : 10, 'num_timings' : 100} ) 

    # Compare a 100x100 dense matrix
    cases.append( {'name' : '100 x 100 Dense Matrix\nMaxiter = 100',
                   'A' : mat(rand(100,100)),   'b' : rand(100,1),  'x0' : zeros((100,1)), 'maxiter' : 100, 'num_timings' : 20} ) 

    # Compare a 500x500 dense matrix
    cases.append( {'name' : '500 x 500 Dense Matrix\nMaxiter = 125',
                   'A' : mat(rand(500,500)),   'b' : rand(500,1),  'x0' : zeros((500,1)), 'maxiter' : 125, 'num_timings' : 20} ) 
    
    # Compare a 100x100 sparse matrix
    A = pyamg.poisson( (10,10), format='csr' )
    cases.append( {'name' : '100 x 100 Sparse Matrix\nMaxiter = 100',
                   'A' : A,   'b' : rand(100,1),  'x0' : zeros((100,1)), 'maxiter' : 100, 'num_timings' : 100} ) 

    # Compare a 1024 x 1024 sparse matrix
    A = pyamg.poisson( (32,32), format='csr' )
    cases.append( {'name' : '1024 x 1024 Sparse Matrix\nMaxiter=250',
                   'A' : A,   'b' : rand(1024,1),  'x0' : zeros((1024,1)), 'maxiter' : 250, 'num_timings' : 20} ) 

    # Compare a 6400 x 6400 sparse matrix
    A = pyamg.poisson( (80,80), format='csr' )
    cases.append( {'name' : '6400 x 6400 Sparse Matrix\nMaxiter = 100',
                   'A' : A,   'b' : rand(6400,1),  'x0' : zeros((6400,1)), 'maxiter' : 100, 'num_timings' : 12} ) 

    # Compare a 6400 x 6400 sparse matrix, with SA preconditioning
    A = pyamg.poisson( (80,80), format='csr' )
    SA =  pyamg.smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2')
    A.psolve = SA.psolve
    cases.append( {'name' : '6400 x 6400 Sparse Matrix With SA Preconditioning\nMaxiter = 25',
                   'A' : A,   'b' : rand(6400,1),  'x0' : zeros((6400,1)), 'maxiter' : 25, 'num_timings' : 27} ) 

    # Compare a 16 384 x 16 384 sparse matrix
    #A = pyamg.poisson( (128,128), format='csr' )
    #cases.append( {'name' : '16 384 x 16 384 Sparse Matrix\nMaxiter = 100',
    #                'A' : A,   'b' : rand(16384,1),  'x0' : zeros((16384,1)), 'maxiter' : 100, 'num_timings' : 12} ) 

    # Compare a 16 384 x 16 384 sparse matrix, with preconditioning
    #A = pyamg.poisson( (128,128), format='csr' )
    #SA =  pyamg.smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2')
    #A.psolve = SA.psolve
    #cases.append( {'name' : '16 384 x 16 384 Sparse Matrix With SA Preconditioning\nMaxiter = 25', 
    #               'A' : A,   'b' : rand(16384,1),  'x0' : zeros((16384,1)), 'maxiter' : 25, 'num_timings' : 20} ) 
    
    print "\n"
    for i in range(len(cases)):
        t1 = 0.0
        for j in range(cases[i]['num_timings']):
            t1start = time.time()
            m1(cases[i]['A'], cases[i]['b'], x0=cases[i]['x0'], maxiter=cases[i]['maxiter'])
            t1end = time.time()
            if j > 1:
                t1 += (t1end - t1start)

        t1avg = t1/(cases[i]['num_timings'] - 2.0)
        
        t2 = 0.0
        for j in range(cases[i]['num_timings']):
            t2start = time.time()
            m2(cases[i]['A'], cases[i]['b'], x0=cases[i]['x0'], maxiter=cases[i]['maxiter'])
            t2end = time.time()
            if j > 1:
                t2 += (t2end - t2start)
                
        t2avg = t2/(cases[i]['num_timings'] - 2.0)
        
        print cases[i]['name'] + "\n" + ("method 1: %e \n" % t1avg) + ("method 2: %e \n" % t2avg)


#################################################################################################
#                           Begin Complex Test Routines                                         #
#################################################################################################

def test_complex_solver(krylov):
    '''
    Test method defined by function ptr, krylov, on a number of test cases
    Assume that the call string of krylov is the same as the library scipy krylov methods
    '''
    import pyamg
    from scipy import complex64, real, imag
    # The Real Test routing tests for invalid and all zero inputs
    
    # Run Numerical Tests
    n_max = 12              #max system size to test

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
        test_krylov(krylov, cases[i][0], cases[i][1])
        test_krylov(krylov, cases[i][0], ravel(cases[i][1]))
        test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],1) )
        test_krylov(krylov, cases[i][0], cases[i][1], x0=(rand(cases[i][1].shape[0],)+1.0j*rand(cases[i][1].shape[0],)) )
    
    # Build and test list of random dense matrices that are single precision
    cases = []
    for i in range(1,n_max):
        cases.append( (mat(rand(i,i), dtype=float32), array(rand(i,1),dtype=complex64)) )

    for i in range(len(cases)):
        test_krylov(krylov, cases[i][0], cases[i][1], tol=5e-5)
    
   # Build and test list of random sparse matrices
    cases = []
    for i in range(1,n_max):
        Areal = rand(i,i); Areal[ abs(Areal) < 0.5 ] = 0.0; Areal = csr_matrix(Areal); Areal = Areal + scipy.sparse.eye(i,i); 
        Aimag = rand(i,i); Aimag[ abs(Aimag) < 0.5 ] = 0.0; Aimag = csr_matrix(Aimag); 
        Aimag = Aimag + scipy.sparse.eye(i,i); Aimag.data = Aimag.data + 1.0j*rand(Aimag.data.shape[0],)
        Aimag2 = rand(i,i); Aimag2[ abs(Aimag2) < 0.5 ] = 0.0; Aimag2 = csr_matrix(Aimag2); 
        Aimag2 = Aimag2 + scipy.sparse.eye(i,i); Aimag2.data = 1.0j*Aimag2.data
        
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
        test_krylov(krylov, cases[i][0], cases[i][1])
        test_krylov(krylov, cases[i][0], ravel(cases[i][1]))
        test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],1) )
        test_krylov(krylov, cases[i][0], cases[i][1], x0=(rand(cases[i][1].shape[0],)+1.0j*rand(cases[i][1].shape[0],)) )
 
    # Build and test list of random sparse matrices that are single precision
    cases = []
    for i in range(1,n_max):
        A = rand(i,i); A[ abs(A) < 0.5 ] = 0.0; A = csr_matrix(A, dtype=complex64); A = A + scipy.sparse.eye(i,i,dtype=complex64);
        A.data = A.data + 1.0j*rand(A.data.shape[0],)
        cases.append( (A, array(rand(i,1),dtype=complex64)) )

    for i in range(len(cases)):
        test_krylov(krylov, cases[i][0], cases[i][1], tol=5e-5)


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
        a = test_krylov(krylov, cases[i][0], cases[i][1], M=cases[i][2] )
        b = test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, M=cases[i][2], x0=real(x0) )
        c = test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, M=cases[i][2] )
        
        cases[i][0].psolve = lambda x:matvec_precon(x, M=cases[i][2])
        d = test_krylov(krylov, cases[i][0], cases[i][1])
        e = test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, x0=real(x0))
        f = test_krylov(krylov, cases[i][0], cases[i][1], x0=x0)
        
        if (mynorm(ravel(a-d)) > 1e-10) or (mynorm(ravel(b-e)) > 1e-10) or (mynorm(ravel(c-f)) > 1e-10):
            print "matrix and mat-vec routine preconditioning yielded different results"
            assert(0)

# No complex analogue for this yet
#    # test that a preconditioner (1) speeds up convergence 
#    # (2) convergence is halted when tol is satisfied, and iterations go no further
#    # This will also test whether callback_fcn works properly
#    cases = []
#    import pyamg
#    for i in range(10,12):
#        A = pyamg.poisson( (i,i), format='csr' )
#        SA =  pyamg.smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2')
#        A.psolve = SA.psolve
#        cases.append(( A, rand(A.shape[0],1)    ))
#
#    for i in range(len(cases)):
#        A = cases[i][0]; b = cases[i][1]; x0=rand(A.shape[0],1);
#        
#        # Test with preconditioning
#        r = zeros(A.shape[0])
#        r[0] = mynorm(ravel(b - A*x0))
#        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
#        test_krylov_picky(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn)
#        r = r[r.nonzero()[0]][1:]
#        iters1 = max(r.shape)
#        
#        # Test w/o preconditioning
#        del A.psolve
#        r = zeros(A.shape[0])
#        r[0] = mynorm(ravel(b - A*x0))
#        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
#        test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn)
#        r = r[r.nonzero()[0]][1:]
#        iters2 = max(r.shape)
#
#        if (iters2 - iters1) < 20:
#            print "Preconditioning not speeding up convergence"
#            assert(0)


    # test krylov with different numbers of iterations and restarts.  
    # use a callback function to make sure that the correct number of iterations are taken.
    # here, we write to the next open spot in an array each time callback is called, so that 
    # an array out-of-bounds error will happen if krylov iterates too many times
    cases=[]
    for i in range(10,13):
        A = pyamg.poisson( (i,i), format='csr' ); A.data = 10*A.data + 0.1j*rand(A.data.shape[0],)
        cases.append(( A, rand(A.shape[0],1)    ))
    print "\n----- Ignore GMRES convergence failure until further notice ----"
    for i in range(len(cases)):
        A = cases[i][0]; b=cases[i][1]; x0=rand(A.shape[0],1);
        
        restrt=1; maxiter=int(ceil(A.shape[0]/8))
        r = zeros(restrt*maxiter+1)
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, restrt=restrt, maxiter=maxiter, tol=1e-12)
        shape1 = r.shape
        r = r[r.nonzero()[0]][0:]
        shape2 = r.shape
        if shape1 != shape2:
            print "callback used an incorrect number of times"
            assert(0)

        restrt=2; maxiter=int(ceil(A.shape[0]/8))
        r = zeros(restrt*maxiter + 1)
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, restrt=restrt, maxiter=maxiter, tol=1e-12)
        shape1 = r.shape
        r = r[r.nonzero()[0]][0:]
        shape2 = r.shape
        if shape1 != shape2:
            print "callback used an incorrect number of times"
            assert(0)

        restrt=3; maxiter=int(ceil(A.shape[0]/8))
        r = zeros(restrt*maxiter + 1)
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, restrt=restrt, maxiter=maxiter, tol=1e-12)
        shape1 = r.shape
        r = r[r.nonzero()[0]][0:]
        shape2 = r.shape
        if shape1 != shape2:
            print "callback used an incorrect number of times"
            assert(0)

    print "---- Stop Ignoring GMRES convergence failures ----\n"


def compare_complex_krylov(m1, m2):
    ''' compare two krylov methods, m1 and m2
        conduct timings and compare answers '''
    import pyamg

    # Run a few tests and compare answers
    cases = []
    # Random Dense matrices
    for i in range(10,13):
        cases.append( {'A' : mat(rand(i,i) + 1.0j*rand(i,i)),   'b' : 1.0j*rand(i,1),  'x0': zeros((i,1)), 'tol' : 1e-5} ) 
        cases.append( {'A' : mat(1.0j*rand(i,i)),   'x0' : rand(i,1), 'b' : 1.0j*rand(i,1)+rand(i,1), 'tol' : 1e-5} ) 
        cases.append( {'A' : mat(rand(i,i)),   'x0' : rand(i,1)+1.0j*rand(i,1), 'b' : 1.0j*rand(i,1),    'tol' : 1e-5} ) 
    
    # Random Sparse Matrices
    for i in range(10,13):
        A  = rand(i,i); A[ abs(A) < 0.5 ] = 0.0; A = csr_matrix(A); A = A + scipy.sparse.eye(i,i); A.data = A.data + 1.0j*rand(A.data.shape[0],)
        A2 = rand(i,i); A2[ abs(A2) < 0.5 ] = 0.0; A2 = csr_matrix(A2); A2 = A2 + scipy.sparse.eye(i,i);  A2.data = 1.0j*A2.data
        cases.append( {'A' : A,   'b' : 1.0j*rand(i,1),  'x0' : zeros((i,1)), 'tol' : 1e-5} ) 
        cases.append( {'A' : A2,  'b' : rand(i,1) + 1.0j*rand(i,1),  'x0' : rand(i,1)+1.0j*rand(i,1),  'tol' : 1e-5} ) 

# currently no complex laplacian SA preconditioner
#    # Preconditioned Poisson Problems
#    for i in range(10,13):
#        A = pyamg.poisson( (i,i), format='csr' )
#        SA =  pyamg.smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2')
#        A.psolve = SA.psolve
#        cases.append( {'A' : A,   'b' : rand(A.shape[0],1),   'x0' : zeros((A.shape[0],1)),  'tol' : 1e-5} ) 
#        cases.append( {'A' : A,   'x0' : rand(A.shape[0],1),  'b'  : zeros((A.shape[0],1)),  'tol' : 1e-16} ) 
#        cases.append( {'A' : A,   'x0' : rand(A.shape[0],1),  'b'  : rand(A.shape[0],1),     'tol' : 1e-16} ) 

    # Compare answers
    for i in range(len(cases)):
        (x1,flag1) = m1(cases[i]['A'], cases[i]['b'], x0=cases[i]['x0'], tol=cases[i]['tol'])
        (x2,flag2) = m2(cases[i]['A'], cases[i]['b'], x0=cases[i]['x0'], tol=cases[i]['tol'])
        x1 = ravel(x1); x2 = ravel(x2)

        if flag1 != flag2:
            print "test %d produced difference convergence flags, flag1 = %d, flag2 = %d" % (i, flag1, flag2)

        print "Test %d, || x1-x2 ||_2 = %e" % (i, mynorm(ravel(x1-x2)))

    import time
    # Conduct timings
    # Compare a 10x10 dense matrix
    cases = []
    cases.append( {'name' : '10 x 10 Dense Matrix\nMaxiter = 10',
                   'A' : mat(rand(10,10)+1.0j*rand(10,10)),   'b' : rand(10,1) + 1.0j*rand(10,1),  'x0' : zeros((10,1)), 'maxiter' : 10, 'num_timings' : 100} ) 

    # Compare a 100x100 dense matrix
    cases.append( {'name' : '100 x 100 Dense Matrix\nMaxiter = 100',
                   'A' : mat(rand(100,100)+1.0j*rand(100,100)),   'b' : rand(100,1) + 1.0j*rand(100,1),  'x0' : zeros((100,1)), 'maxiter' : 100, 'num_timings' : 20} ) 

    # Compare a 500x500 dense matrix
    cases.append( {'name' : '500 x 500 Dense Matrix\nMaxiter = 125',
                   'A' : mat(rand(500,500)+1.0j*rand(500,500)),   'b' : rand(500,1) + 1.0j*rand(500,1),  'x0' : zeros((500,1)), 'maxiter' : 125, 'num_timings' : 20} ) 
    
    # Compare a 100x100 sparse matrix
    A = pyamg.poisson( (10,10), format='csr' ); A.data = 10*A.data + 0.1j*rand(A.data.shape[0],)
    cases.append( {'name' : '100 x 100 Sparse Matrix\nMaxiter = 100',
                   'A' : A,   'b' : rand(100,1) + 1.0j*rand(100,1),  'x0' : zeros((100,1)), 'maxiter' : 100, 'num_timings' : 100} ) 

    # Compare a 1024 x 1024 sparse matrix
    A = pyamg.poisson( (32,32), format='csr' ); A.data = 10*A.data + 0.1j*rand(A.data.shape[0],)
    cases.append( {'name' : '1024 x 1024 Sparse Matrix\nMaxiter=250',
                   'A' : A,   'b' : rand(1024,1) + 1.0j*rand(1024,1),  'x0' : zeros((1024,1)), 'maxiter' : 250, 'num_timings' : 20} ) 

    # Compare a 6400 x 6400 sparse matrix
    A = pyamg.poisson( (80,80), format='csr' ); A.data = 10*A.data + 0.1j*rand(A.data.shape[0],) 
    cases.append( {'name' : '6400 x 6400 Sparse Matrix\nMaxiter = 100',
                   'A' : A,   'b' : rand(6400,1) + 1.0j*rand(6400,1),  'x0' : zeros((6400,1)), 'maxiter' : 100, 'num_timings' : 12} ) 
    
    # currently now SA precon for complex case
    # Compare a 6400 x 6400 sparse matrix, with SA preconditioning
    #A = pyamg.poisson( (80,80), format='csr' )
    #SA =  pyamg.smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2')
    #A.psolve = SA.psolve
    #cases.append( {'name' : '6400 x 6400 Sparse Matrix With SA Preconditioning\nMaxiter = 25',
    #               'A' : A,   'b' : rand(6400,1),  'x0' : zeros((6400,1)), 'maxiter' : 25, 'num_timings' : 27} ) 

    # Compare a 16 384 x 16 384 sparse matrix
    #A = pyamg.poisson( (128,128), format='csr' )
    #cases.append( {'name' : '16 384 x 16 384 Sparse Matrix\nMaxiter = 100',
    #                'A' : A,   'b' : rand(16384,1),  'x0' : zeros((16384,1)), 'maxiter' : 100, 'num_timings' : 12} ) 

    # Compare a 16 384 x 16 384 sparse matrix, with preconditioning
    #A = pyamg.poisson( (128,128), format='csr' )
    #SA =  pyamg.smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2')
    #A.psolve = SA.psolve
    #cases.append( {'name' : '16 384 x 16 384 Sparse Matrix With SA Preconditioning\nMaxiter = 25', 
    #               'A' : A,   'b' : rand(16384,1),  'x0' : zeros((16384,1)), 'maxiter' : 25, 'num_timings' : 20} ) 
    
    print "\n"
    for i in range(len(cases)):
        t1 = 0.0
        for j in range(cases[i]['num_timings']):
            t1start = time.time()
            m1(cases[i]['A'], cases[i]['b'], x0=cases[i]['x0'], maxiter=cases[i]['maxiter'])
            t1end = time.time()
            if j > 1:
                t1 += (t1end - t1start)

        t1avg = t1/(cases[i]['num_timings'] - 2.0)
        
        t2 = 0.0
        for j in range(cases[i]['num_timings']):
            t2start = time.time()
            m2(cases[i]['A'], cases[i]['b'], x0=cases[i]['x0'], maxiter=cases[i]['maxiter'])
            t2end = time.time()
            if j > 1:
                t2 += (t2end - t2start)
                
        t2avg = t2/(cases[i]['num_timings'] - 2.0)
        
        print cases[i]['name'] + "\n" + ("method 1: %e \n" % t1avg) + ("method 2: %e \n" % t2avg)



def test_fgmres(krylov):
    ''' Test FGMRES by using a variable preconditioner.  
    We use gmres here as the variable preconditioner. ''' 
    
    from scipy.sparse.linalg.interface import LinearOperator

    # Real and Imaginary Tests
    cases = []
    # Random Dense matrices
    #for i in [100, 121, 144]:
    #    cases.append( {'A' : mat(rand(i,i) + 1.0j*rand(i,i)),   'b' : 1.0j*rand(i,1),  'x0': zeros((i,1)), 'tol' : 1e-5} ) 
    #    cases.append( {'A' : mat(1.0j*rand(i,i)),   'x0' : rand(i,1), 'b' : 1.0j*rand(i,1)+rand(i,1), 'tol' : 1e-8} ) 
    #    cases.append( {'A' : mat(rand(i,i)),   'x0' : rand(i,1)+1.0j*rand(i,1), 'b' : 1.0j*rand(i,1),    'tol' : 1e-5} ) 
    #    cases.append( {'A' : mat(rand(i,i)),   'x0' : rand(i,1), 'b' : rand(i,1),    'tol' : 1e-8} ) 
    #    cases.append( {'A' : mat(rand(i,i)),   'x0' : zeros((i,1)), 'b' : rand(i,1),    'tol' : 1e-5} ) 
    
    # Random Sparse Matrices
    for i in [300, 349, 402]:
        A  = rand(i,i); A[ abs(A) < 0.75 ] = 0.0; A = csr_matrix(A); A = A + scipy.sparse.eye(i,i); A.data = A.data + 1.0j*rand(A.data.shape[0],)
        A2 = rand(i,i); A2[ abs(A2) < 0.75 ] = 0.0; A2 = csr_matrix(A2); A2 = A2 + scipy.sparse.eye(i,i);  A2.data = 1.0j*A2.data
        A3 = rand(i,i); A3[ abs(A3) < 0.75 ] = 0.0; A3 = csr_matrix(A3); A3 = A3 + scipy.sparse.eye(i,i); 
        cases.append( {'A' : A,   'b' : 1.0j*rand(i,1),  'x0' : zeros((i,1)), 'tol' : 1e-5} ) 
        cases.append( {'A' : A2,  'b' : rand(i,1) + 1.0j*rand(i,1),  'x0' : rand(i,1)+1.0j*rand(i,1),  'tol' : 1e-8} ) 
        cases.append( {'A' : A3,  'b' : rand(i,1),  'x0' : rand(i,1),  'tol' : 1e-5} ) 
        cases.append( {'A' : A3,  'b' : rand(i,1),  'x0' : zeros((i,1)),  'tol' : 1e-8} ) 
    
    # Run Tests
    maxiter=5
    print "\n\nKrylov method speed up by flexible preconditioning (%d iterations of gmres) on random sparse matrices, both imaginary and real" % maxiter
    for i in range(len(cases)):
        A = cases[i]['A']; b = cases[i]['b']; tol = cases[i]['tol']; x0 = cases[i]['x0']
        precon = lambda guess:gmres(A, b, x0=guess, tol=tol, maxiter=maxiter)[0]
        M = LinearOperator(A.shape, matvec=precon, dtype=A.dtype)
        
        # With preconditioning
        print "Running Krylov WITH Preconditioning"
        r = zeros(A.shape[0] + 1)
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov(krylov, A, b, x0=x0, tol=tol, M=M, callback=callback_fcn, left_precon=False)
        r = r[r.nonzero()[0]][0:]
        num_iters1 = r.shape[0]
    
        # Without preconditioning
        print "Running Krylov withOUT Preconditioning"
        r = zeros(A.shape[0] + 1)
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov(krylov, A, b, x0=x0, tol=tol, callback=callback_fcn)
        r = r[r.nonzero()[0]][0:]
        num_iters2 = r.shape[0]
        
        speed_up = num_iters2 - num_iters1
        print "%d iteration speed up,  system size: %d x %d\n" % (speed_up, A.shape[0], A.shape[1])




def test_sparse_solver(krylov):
    '''
    Test method defined by function ptr, krylov, on a number of sparse test cases
    Assume that the call string of krylov is the same as the library scipy krylov methods
    Restarts are not tested in this method
    '''
    from scipy import complex64, real
    from scipy.linalg import qr

    A = csr_matrix(zeros((10,10)))
    v1 = zeros((10,1))
    
    # Call Krylov with all zero matrices 
    (x,flag) = krylov(A,v1)
    if (not (x == zeros((10,))).all()) or (flag != 0):
        print 'krylov failed with all zero input'
        assert(0)
    
    (x,flag) = krylov(A,v1, x0=v1)
    if (not (x == zeros((10,))).all()) or (flag != 0):
        print 'krylov failed with all zero input'
        assert(0)
    
    (x,flag) = krylov(A,v1, M=A)
    if (not (x == zeros((10,))).all()) or (flag != 0):
        print 'krylov failed with all zero input'
        assert(0)
    
    (x,flag) = krylov(A,v1, M=A, x0=v1)
    if (not (x == zeros((10,))).all()) or (flag != 0):
        print 'krylov failed with all zero input'
        assert(0)
    
    # Try some invalid iteration numbers, invalid dimensions are handled 
    # by the make_system function in scipy.sparse.linalg.isolve.utils 
    try: krylov(A, v1, maxiter=0)
    except ValueError:
        pass
    else:
        print "krylov failed to detect bad maxiter"
        assert(0)

    
    # Run Numerical Tests
    n_max = 12              #max system size to test
   
    print "\nTest Battery 1"
    # Build and test list of random sparse matrices
    cases = []
    for i in range(1,n_max):
        A = rand(i,i); A[ abs(A) < 0.5 ] = 0.0; A = csr_matrix(A); A = A + scipy.sparse.eye(i,i);
        cases.append( (A, rand(i,1)) )

    for i in range(len(cases)):
        test_krylov(krylov, cases[i][0], cases[i][1], left_precon=False, use_restrt=False)
        test_krylov(krylov, cases[i][0], ravel(cases[i][1]), left_precon=False, use_restrt=False)
        test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, left_precon=False, use_restrt=False)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],1), left_precon=False, use_restrt=False)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],), left_precon=False, use_restrt=False)
 
    print "\nTest Battery 2"
    # Build and test list of random sparse matrices that are single precision
    cases = []
    for i in range(1,n_max):
        A = rand(i,i); A[ abs(A) < 0.5 ] = 0.0; A = csr_matrix(A, dtype=float32); A = A + scipy.sparse.eye(i,i,dtype=float32);
        cases.append( (A, array(rand(i,1),dtype=float32)) )

    for i in range(len(cases)):
        test_krylov(krylov, cases[i][0], cases[i][1], tol=5e-5, left_precon=False, use_restrt=False)


    print "\nTest Battery 3"
    # Build and test list of random diagonally dominant matrices, with diagonal preconditioner
    from scipy import diag
    cases = []
    for i in range(15,20):
        A = rand(i,i); A[ abs(A) < 0.5 ] = 0.0; A = csr_matrix(A); A = A + 2*i*scipy.sparse.eye(i,i);
        M = csr_matrix(diag(1.0/A.diagonal())); 
        cases.append( (A, rand(i,1), M) )

    # test with matrix and mat-vec routine as the preconditioner
    for i in range(len(cases)):
        a = test_krylov(krylov, cases[i][0], cases[i][1], M=cases[i][2], left_precon=False, use_restrt=False)
        b = test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, M=cases[i][2], left_precon=False, use_restrt=False)
        x0 = rand(cases[i][1].shape[0],1)
        c = test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, M=cases[i][2], left_precon=False, use_restrt=False)
        
        cases[i][0].psolve = lambda x:matvec_precon(x, M=cases[i][2])
        d = test_krylov(krylov, cases[i][0], cases[i][1], left_precon=False, use_restrt=False)
        e = test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, left_precon=False, use_restrt=False)
        f = test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, left_precon=False, use_restrt=False)
        
        if (mynorm(ravel(a-d)) > 1e-10) or (mynorm(ravel(b-e)) > 1e-10) or (mynorm(ravel(c-f)) > 1e-10):
            print "matrix and mat-vec routine preconditioning yielded different results"
            assert(0)

    print "\nTest Battery 4"
    # test that a preconditioner (1) speeds up convergence 
    # (2) convergence is halted when tol is satisfied, and iterations go no further
    # This will also test whether callback_fcn works properly
    cases = []
    import pyamg
    for i in range(10,12):
        A = pyamg.poisson( (i,i), format='csr' )
        SA =  pyamg.smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2')
        A.psolve = SA.psolve
        cases.append(( A, rand(A.shape[0],1)    ))

    for i in range(len(cases)):
        A = cases[i][0]; b = cases[i][1]; x0=rand(A.shape[0],1);
        
        # Test with preconditioning
        r = zeros(A.shape[0])
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov_picky(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, left_precon=False, use_restrt=False)
        r = r[r.nonzero()[0]][1:]
        iters1 = max(r.shape)
        
        # Test w/o preconditioning
        del A.psolve
        r = zeros(A.shape[0])
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, left_precon=False, use_restrt=False)
        r = r[r.nonzero()[0]][1:]
        iters2 = max(r.shape)

        if (iters2 - iters1) < 20:
            print "Preconditioning not speeding up convergence"
            assert(0)


    print "\nTest Battery 5"
    # test krylov with different numbers of iterations 
    # use a callback function to make sure that the correct number of iterations are taken.
    # here, we write to the next open spot in an array each time callback is called, so that 
    # an array out-of-bounds error will happen if krylov iterates too many times
    cases=[]
    for i in range(10,13):
        A = pyamg.poisson( (i,i), format='csr' )
        cases.append(( A, rand(A.shape[0],1)    ))
    print "\n----- Ignore GMRES convergence failure until further notice ----"
    for i in range(len(cases)):
        A = cases[i][0]; b=cases[i][1]; x0=rand(A.shape[0],1);
        
        maxiter=int(ceil(A.shape[0]/8))
        r = zeros(maxiter+1)
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, maxiter=maxiter, tol=1e-12, left_precon=False, use_restrt=False)
        shape1 = r.shape
        r = r[r.nonzero()[0]][0:]
        shape2 = r.shape
        if shape1 != shape2:
            print "callback used an incorrect number of times"
            assert(0)

    print "---- Stop Ignoring GMRES convergence failures ----\n"


    # ----------------------------------- Begin Complex Tests ------------------------------------
    
    print "\nTest Battery 6 (complex)"
    # Build and test list of random sparse matrices
    cases = []
    for i in range(1,n_max):
        Areal = rand(i,i); Areal[ abs(Areal) < 0.5 ] = 0.0; Areal = csr_matrix(Areal); Areal = Areal + scipy.sparse.eye(i,i); 
        Aimag = rand(i,i); Aimag[ abs(Aimag) < 0.5 ] = 0.0; Aimag = csr_matrix(Aimag); 
        Aimag = Aimag + scipy.sparse.eye(i,i); Aimag.data = Aimag.data + 1.0j*rand(Aimag.data.shape[0],)
        Aimag2 = rand(i,i); Aimag2[ abs(Aimag2) < 0.5 ] = 0.0; Aimag2 = csr_matrix(Aimag2); 
        Aimag2 = Aimag2 + scipy.sparse.eye(i,i); Aimag2.data = 1.0j*Aimag2.data
        # This does what is expected, passing in an orthogonal matrix results in a great CGNE performance
        #Areal = csr_matrix(qr(Areal.todense())[0])
        #Aimag = csr_matrix(qr(Aimag.todense())[0])
        #Aimag2 = csr_matrix(qr(Aimag2.todense())[0])

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
        test_krylov(krylov, cases[i][0], cases[i][1], left_precon=False, use_restrt=False)
        test_krylov(krylov, cases[i][0], ravel(cases[i][1]), left_precon=False, use_restrt=False)
        test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, left_precon=False, use_restrt=False)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=rand(cases[i][1].shape[0],1), left_precon=False, use_restrt=False)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=(rand(cases[i][1].shape[0],)+1.0j*rand(cases[i][1].shape[0],)), left_precon=False, use_restrt=False)
 
    print "\nTest Battery 7 (complex)"
    # Build and test list of random sparse matrices that are single precision
    cases = []
    for i in range(1,n_max):
        A = rand(i,i); A[ abs(A) < 0.5 ] = 0.0; A = csr_matrix(A, dtype=complex64); A = A + scipy.sparse.eye(i,i,dtype=complex64);
        A.data = A.data + 1.0j*rand(A.data.shape[0],)
        cases.append( (A, array(rand(i,1),dtype=complex64)) )

    for i in range(len(cases)):
        test_krylov(krylov, cases[i][0], cases[i][1], tol=5e-5, left_precon=False, use_restrt=False)


    print "\nTest Battery 8 (complex)"
    # Build and test list of random diagonally dominant matrices, with diagonal preconditioner
    cases = []
    for i in range(1,n_max):
        A = rand(i,i); A[ abs(A) < 0.5 ] = 0.0; A = csr_matrix(A); A = A + 2*i*scipy.sparse.eye(i,i); A.data = 0.01j*rand(A.data.shape[0],)
        M = csr_matrix(diag(1.0/A.diagonal())); 
        cases.append( (A, rand(i,1), M) )

    # test with matrix and mat-vec routine as the preconditioner
    for i in range(len(cases)):
        x0 = rand(cases[i][1].shape[0],1) + 1.0j*rand(cases[i][1].shape[0],1)
        a = test_krylov(krylov, cases[i][0], cases[i][1], M=cases[i][2], left_precon=False, use_restrt=False)
        b = test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, M=cases[i][2], x0=real(x0), left_precon=False, use_restrt=False)
        c = test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, M=cases[i][2], left_precon=False, use_restrt=False)
        
        cases[i][0].psolve = lambda x:matvec_precon(x, M=cases[i][2])
        d = test_krylov(krylov, cases[i][0], cases[i][1], left_precon=False, use_restrt=False)
        e = test_krylov(krylov, cases[i][0], cases[i][1], tol=1e-12, x0=real(x0), left_precon=False, use_restrt=False)
        f = test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, left_precon=False, use_restrt=False)
        
        if (mynorm(ravel(a-d)) > 1e-10) or (mynorm(ravel(b-e)) > 1e-10) or (mynorm(ravel(c-f)) > 1e-10):
            print "matrix and mat-vec routine preconditioning yielded different results"
            assert(0)

# No complex analogue for this yet
#    # test that a preconditioner (1) speeds up convergence 
#    # (2) convergence is halted when tol is satisfied, and iterations go no further
#    # This will also test whether callback_fcn works properly
#    cases = []
#    import pyamg
#    for i in range(10,12):
#        A = pyamg.poisson( (i,i), format='csr' )
#        SA =  pyamg.smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2')
#        A.psolve = SA.psolve
#        cases.append(( A, rand(A.shape[0],1)    ))
#
#    for i in range(len(cases)):
#        A = cases[i][0]; b = cases[i][1]; x0=rand(A.shape[0],1);
#        
#        # Test with preconditioning
#        r = zeros(A.shape[0])
#        r[0] = mynorm(ravel(b - A*x0))
#        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
#        test_krylov_picky(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn)
#        r = r[r.nonzero()[0]][1:]
#        iters1 = max(r.shape)
#        
#        # Test w/o preconditioning
#        del A.psolve
#        r = zeros(A.shape[0])
#        r[0] = mynorm(ravel(b - A*x0))
#        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
#        test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn)
#        r = r[r.nonzero()[0]][1:]
#        iters2 = max(r.shape)
#
#        if (iters2 - iters1) < 20:
#            print "Preconditioning not speeding up convergence"
#            assert(0)


    print "\nTest Battery 9 (complex)"
    # test krylov with different numbers of iterations and restarts.  
    # use a callback function to make sure that the correct number of iterations are taken.
    # here, we write to the next open spot in an array each time callback is called, so that 
    # an array out-of-bounds error will happen if krylov iterates too many times
    cases=[]
    for i in range(10,13):
        A = pyamg.poisson( (i,i), format='csr' ); A.data = 10*A.data + 0.1j*rand(A.data.shape[0],)
        cases.append(( A, rand(A.shape[0],1)    ))
    print "\n----- Ignore GMRES convergence failure until further notice ----"
    for i in range(len(cases)):
        A = cases[i][0]; b=cases[i][1]; x0=rand(A.shape[0],1);
        
        maxiter=int(ceil(A.shape[0]/8))
        r = zeros(maxiter+1)
        r[0] = mynorm(ravel(b - A*x0))
        callback_fcn = lambda x:Calc_NormResidual(x, b, A, r)
        test_krylov(krylov, cases[i][0], cases[i][1], x0=x0, callback=callback_fcn, maxiter=maxiter, tol=1e-12, left_precon=False, use_restrt=False)
        shape1 = r.shape
        r = r[r.nonzero()[0]][0:]
        shape2 = r.shape
        if shape1 != shape2:
            print "callback used an incorrect number of times"
            assert(0)

    print "---- Stop Ignoring GMRES convergence failures ----\n"

