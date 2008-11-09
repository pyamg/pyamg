"""Compatible Relaxation"""

__docformat__ = "restructuredtext en"

#import time

from numpy import array, dot, multiply, power, sqrt, sum, ones, arange, \
        abs, inf, ceil, zeros, where, bool
from numpy.random import random
from scipy.linalg import norm
from scipy.sparse import isspmatrix, csr_matrix

from pyamg.relaxation import gauss_seidel

__all__ = ['CR','binormalize']

def CR(S, method='habituated'):
    """Use Compatible Relaxation to compute a C/F splitting 

    Parameters
    ----------
    S : csr_matrix
        sparse matrix (n x n) usually matrix A of Ax=b
    method : {'habituated','concurrent'}
        Method used during relaxation:
            - concurrent: GS relaxation on F-points, leaving e_c = 0
            - habituated: full relaxation, setting e_c = 0
    maxiter : int
        maximum number of outer iterations

    Return
    ------
    splitting : array
        C/F list of 1's (coarse pt) and 0's (fine pt) (n x 1)
   

    References
    ----------
        - Compatible relaxation: coarse set selection by relaxation as described in
            Livne, O.E., Coarsening by compatible relaxation.
            Numer. Linear Algebra Appl. 11, No. 2-3, 205-227 (2004).


    Example
    -------

    """
    # parameters (paper notation)
    maxiter = 20    # (lambda) max number of outer iterations
    ntests = 3      # (nu) number of random tests to do per iteration
    nrelax = 4      # (eta) number of relaxation sweeps per test

    smagic = 1      # (s) parameter in [1,5] to account for fillin 
    gamma = 1.5     # (gamma) cycle index.  use 1.5 for 2d
    G = 30          # (G) number of equivalence classes (# of bins)
    tdepth = 1      # (t) drop depth on parse of L bins
    delta = 0       # (delta) drop threshold on parse of L bins
    alphai = 0.25   # (alpha_inc) quota increase

    # initializaitons    
    alpha = 0.0     # coarsening ratio
    beta = inf      # quality criterion
    beta1 = inf     # quality criterion, older
    beta2 = inf     # quality criterion, oldest
    n=S.shape[0]    # problem size
    nC = 0          # number of current Coarse points
    rhs = zeros((n,1)); # rhs for Ae=0

    if not isspmatrix(S): raise TypeError('expecting sparse matrix')

    #S = binormalize(S)
    
    splitting = zeros( (S.shape[0],1), dtype='uint8' )
   
    # out iterations ---------------
    for m in range(0,maxiter):

        #print "[m = %d"%m
        mu = 0.0        # convergence rate
        E = zeros((n,1))  # slowness measure

        # random iterations ---------------
        for k in range(0,ntests):
            #print "[..k = %d"%k

            e  = 0.5*( 1 + random((n,1)))
            e[splitting>0] = 0

            enorm = norm(e)

            # relaxation iterations ---------------
            for l in range(0,nrelax):
                #print "[....l = %d"%l

                if method == 'habituated':
                    gauss_seidel(S,e,zeros((n,1)),iterations=1)
                    e[splitting>0]=0
                elif method == 'concurrent':
                    raise NotImplementedError, 'not implemented: need an F-smoother'
                else:
                    raise NotImplementedError, 'method not recognized.  need habituated or concurrent'

                enorm_old = enorm
                enorm     = norm(e)

                if enorm <= 1e-14:
                    # break out of loops
                    ntests = k
                    nrelax = l
                    maxiter = m
            # end relax

            # check slowness
            E = where( abs(e)>E, abs(e), E )

            # update convergence rate
            mu = mu + enorm/enorm_old
        # end random tests
        mu = mu/ntests

        # work
        alpha = float(nC)/n

        W = (1 + (smagic-1)*gamma*alpha)/(1-gamma*alpha)
        
        # quality criterion
        beta2 = beta1
        beta1 = beta
        beta = power(max([mu, 0.1]), 1.0 / W)
        
        # check if we're doing well
        if (beta>beta1 and beta1>beta2) or m==(maxiter-1) or max(E)<1e-13:
            return splitting.ravel()

        # now add points
        #
        # update limit on addtions to splitting (C)
        if alpha < 1e-13:
            alpha=0.25
        else:
            alpha = (1-alphai) * alpha + alphai * (1/gamma)

        nCmax = ceil( alpha * n )

        L = ceil( G * E / E.max() ).ravel()

        binid=G

        # add whole bins (and tdepth nodes) at a time
        u = zeros((n,1))
        #troots = zeros((n,1),dtype='bool')
        while nC < nCmax:
            if delta > 0:
                raise NotImplementedError
            if tdepth != 1:
                raise NotImplementedError

            (roots,) = where(L==binid)

            for root in roots:
                if L[root]>=0:
                    cols = S[root,:].indices
                    splitting[root] = 1    # add roots
                    nC += 1
                    L[cols]=-1
            binid -= 1

            #L[troots] = -1          # mark t-rings visited
            #u[:]=0.0
            #u[roots] = 1.0
            #for depth in range(0,tdepth):
            #    u = abs(S) * u
            #(troots,tmp) = where(u>0)

        #print "alpha = %g" % alpha
        #print "beta  = %g" % beta
        #print "mu    = %g" % mu

    return splitting.ravel()

def binormalize( A, tol=1e-5, maxiter=10):
    """Binormalize matrix A

    Attempt to create unit l_1 norm rows

    Usage
    -----
    B = binormalize( A, tol=1e-8, maxiter=20 )
    n = A.shape[0]
    C = B.multiply(B)
    print C.sum(1) # check binormalization

    Parameters
    ----------
    A : csr_matrix
        sparse matrix (n x n)
    tol : float
        tolerance
    x : array
        guess at the diagonal
    maxiter : int
        maximum number of iterations to try

    Return
    ------
        C : csr_matrix
            diagonally scaled A, C=DAD
         
    
    Notes
    -----
        - BIN Algorithm to binormalize a matrix following:

          Livne, Golub, Scaling by Binormalization, 2003

        - Goal: Scale A so that l_1 norm of the rows are equal to 1:
                |\sum_{j=1}^N |b_{ij}| - 1| \leq tol
                o B=DAD -> b_{ij}
                o easily done with tol=0 if B=DA, but this is not symmetric
                o algorithm is O(N log (1.0/tol))

    Example
    -------

    """
    if not isspmatrix(A): raise TypeError('expecting sparse matrix A')

    n  = A.shape[0]
    it = 0
    x = ones((n,1)).ravel()

    # 1.
    B = A.multiply(A).tocsc()  # power(A,2) inconsistent for numpy, scipy.sparse
    d=B.diagonal().ravel()
    
    # 2.
    beta    = B * x
    betabar = (1.0/n) * dot(x,beta)
    stdev = rowsum_stdev(x,beta)

    #3
    #t=0.0
    while stdev > tol and it < maxiter:
        for i in range(0,n):
            # solve equation x_i, keeping x_j's fixed
            # see equation (12)
            c2 = (n-1)*d[i]
            c1 = (n-2)*(beta[i] - d[i]*x[i])
            c0 = -d[i]*x[i]*x[i] + 2*beta[i]*x[i] - n*betabar
            if (-c0 < 1e-13):
                print 'warning: A nearly un-binormalizable...'
                return ones((n,1))
            else:
                # see equation (12)
                xnew = (-2*c0)/(c1 + sqrt(c1*c1 - 4*c0*c2))
            dx = xnew - x[i]

            #ttmp=time.time()

            #betabar = betabar + (1.0/n)*(dx * x.T * B[:,i] + dx*beta[i] + d[i]*dx*dx)
            #beta = beta + dx*array(B[:,i].todense()).ravel()

            Bcol = dx*B[:,i]
            betabar = betabar + (1.0/n)*(dot(x[Bcol.indices],Bcol.data) + dx*beta[i] + d[i]*dx*dx)
            beta[Bcol.indices] += Bcol.data

            #t+=time.time()-ttmp

            x[i] = xnew
        stdev = rowsum_stdev(x,beta)
        #print "time %d: "%it, t
        it+=1

    # rescale for unit 2-norm
    d = sqrt(x)
    D=csr_matrix( ( d.ravel(), (arange(0,n),arange(0,n))), shape=(n,n))
    C = D * A * D
    beta = C.multiply(C).sum(1)
    scale = sqrt((1.0/n) * sum(beta))
    return (1/scale)*C


def rowsum_stdev(x,beta):
    """Compute row sum standard deviation

    Compute for approximation x, the std dev of the row sums
    s(x) = ( 1/n \sum_k  (x_k beta_k - betabar)^2 )^(1/2)
    with betabar = 1/n x'*beta

    Parameters
    ----------
        x : array
        beta : array

    Notes
    -----
        equation (7) in Livne/Golub

    return 
    ------
        s(x)/betabar : float
    """
    n=x.size
    betabar = (1.0/n) * dot(x,beta)
    stdev   = sqrt((1.0/n)*sum(power(multiply(x,beta) - betabar,2)))
    return stdev/betabar
