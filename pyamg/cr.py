from scipy.sparse import isspmatrix, csr_matrix
from numpy import array, dot, multiply, power, sqrt, sum, ones, arange
from numpy.random import random

def cr(S,method='concurrent',maxit=20):
    """
    Usage
    =====

    Input
    =====
        S      : sparse matrix (n x n) usually matrix A of Ax=b
        method : concurrent or habituated

    Output
    ======
        splitting : C/F list of 1's (coarse pt) and 0's (fine pt) (n x 1)
    
    Notes
    =====
        - Compatible relaxation: coarse set selection by relaxation as described in
            Livne, O.E., Coarsening by compatible relaxation.
            Numer. Linear Algebra Appl. 11, No. 2-3, 205-227 (2004).
        - concurrent: GS relaxation on F-points, leaving e_c = 0
        - habituated: full relaxation, setting e_c = 0

    Example
    =======

    """
    alpha_inc = 0.25
    gamma = 1.5

    if not isspmatrix(S): raise TypeError('expecting sparse matrix A')

    A = binormalize(A)
    
    # TODO  <-- works up to here
    raise NotImplementedError, 'TODO:  not implemented beyond this point'

    splitting = empty( S.shape[0], dtype='uint8' )
   
    e  = 0.5*( 1 + random((n,1)))
    e[splitting>0]=0

    for m in range(0,maxit):
        # perform relaxation
        eold=e
        e = 0.5*( 1 + random((n,1)))
        e[splitting>0]=0
        if method == 'habituated':
            gauss_seidel(A,e,zeros((n,1)),iterations=4)
            e[splitting>0]=0
        elif method == 'concurrent':
            raise NotImplementedError, 'not implemented: need an F-smoother'

        # get quality
        mu = norm(e)/norm(eold) 
        mu = max([mu,0.1])

    return splitting

def binormalize( A, tol=1e-5, maxit=10):
    """
    Usage
    =====
    B = binormalize( A, tol=1e-8, maxit=20 )
    n = A.shape[0]
    C = B.multiply(B)
    print C.sum(1) # check binormalization

    Input
    =====
        A      : sparse matrix (n x n)
        tol    : tolerance
        x      : guess at the diagonal
        maxti  : maximum number of iterations to try

    Output
    ======
        d      : diagonal of D for use with B=DAD
         
    
    Notes
    =====
        - BIN Algorithm to binormalize a matrix following:
          Livne, Golub, Scaling by Binormalization, 2003
        - Goal: Scale A so that l_1 norm of the rows are equal to 1:
                |\sum_{j=1}^N |b_{ij}| - 1| \leq tol
                o B=DAD -> b_{ij}
                o easily done with tol=0 if B=DA, but this is not symmetric
                o algorithm is O(N log (1.0/tol))

    Example
    =======

    """
    if not isspmatrix(A): raise TypeError('expecting sparse matrix A')

    n  = A.shape[0]
    it = 0
    x = ones((n,1)).ravel()

    # 1.
    B = A.multiply(A)  # power(A,2) inconsistent for numpy, scipy.sparse
    d=B.diagonal().ravel()
    
    # 2.
    beta    = B * x
    betabar = (1.0/n) * dot(x,beta)
    stdev = rowsum_stdev(x,beta)

    #3
    while stdev>tol and it<maxit:
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
            if i==0 and it==0:
                print 'bi ', beta[i]
                print 'di ', d[i]
                print 'xi ', x[i]
                print 'c2 ', c2
                print 'c1 ', c1
                print 'c0 ', c0
                print 'xn ', xnew
            dx = xnew - x[i]
            betabar = betabar + (1.0/n)*(dx * x.T * B[:,i] + dx*beta[i] + d[i]*dx*dx)
            beta = beta + dx*array(B[:,i].todense()).ravel()
            x[i] = xnew
        stdev = rowsum_stdev(x,beta)
        print '%d %g' % (it,stdev)
        it+=1

    # rescale for unit 2-norm
    d = sqrt(x)
    D=csr_matrix( ( d.ravel(), (arange(0,n),arange(0,n))), shape=(n,n))
    C = D * A * D
    beta = C.multiply(C).sum(1)
    scale = sqrt((1.0/n) * sum(beta))
    return (1/scale)*C


def rowsum_stdev(x,beta):
    """
    Compute for approximation x, the std dev of the row sums
    s(x) = ( 1/n \sum_k  (x_k beta_k - betabar)^2 )^(1/2)
    with betabar = 1/n x'*beta

    equation (7) in Livne/Golub

    return s(x)/betabar
    """
    n=x.size
    betabar = (1.0/n) * dot(x,beta)
    stdev   = sqrt((1.0/n)*sum(power(multiply(x,beta) - betabar,2)))
    return stdev/betabar
