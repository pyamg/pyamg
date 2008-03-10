from scipy.sparse import isspmatrix, csr_matrix
from numpy import array, dot, multiply, power, sqrt, sum, ones, arange, abs, inf
from numpy.random import random
from scipy.linalg import norm

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
    # parameters
    maxit = 10      # max number of outer iterations
    ntests = 3      # number of random tests to do per iteration
    nrelax = 4      # number of relaxation sweeps per test

    smagic = 1      # parameter between 1 and 5 to account for fillin at the second level
    gamma = 1.5     # cycle index.  use 1.5 for 2d


    # initializaitons    
    alpha = 0.0     # coarsening ratio
    beta1 = inf     # quality criterion
    n=S.shape[0]    # problem size
    nC = 0          # number of current Coarse points
    rhs = zeros(n,1); # rhs for Ae=0

    if not isspmatrix(S): raise TypeError('expecting sparse matrix A')

    A = binormalize(A)
    
    splitting = empty( S.shape[0], dtype='uint8' )
   
    # out iterations ---------------
    for m in range(0,maxit):

        mu = 0.0        # convergence rate
        E = zeros((n,1))  # slowness measure

        # random iterations ---------------
        for k in range(0,ntests)

            e  = 0.5*( 1 + random((n,1)))
            e[splitting>0] = 0

            enorm = norm(e)

            # relaxation iterations ---------------
            for l in range(0,nrelax)

                if method == 'habituated':
                    gauss_seidel(A,e,zeros((n,1)),iterations=1)
                    e[splitting>0]=0
                elif method == 'concurrent':
                    raise NotImplementedError, 'not implemented: need an F-smoother'

                enorm_old = enorm
                enorm     = norm(e)

                if enorm <= 1e-14:
                    # break out of loops
                    ntests = k
                    nrelax = l
                    maxit = m
            # end relax

            # check slowness
            E = where( abs(e)>E, abs(e), E )

            # update convergence rate
            mu = mu + enorm/enorm_old
        # end random tests

        # work
        alpha = nC/n
        W = (1 + (s-1)*gamma*alpha)/(1-gamma*alpha)
        
        # quality criterion
        beta2 = beta1
        beta1 = beta
        beta = power(max([mu,0.1],1.0/W))
        
        # check if we're doing well
        if (beta>beta1 and beta1>beta2) or m==(maxiters-1) or max(E)<1e-13:
            return splitting

        #
        #
        # TODO up to section 4.4 in Livne
        raise NotImplementedError


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
        maxit  : maximum number of iterations to try

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
