"""Adaptive Smoothed Aggregation"""

__docformat__ = "restructuredtext en"

from numpy import sqrt, ravel, diff, zeros, zeros_like, inner, concatenate, \
                  asarray, hstack, ascontiguousarray, isinf, dot
from numpy.random import randn, rand
from scipy.sparse import csr_matrix, coo_matrix, bsr_matrix

from pyamg.multilevel import multilevel_solver
from pyamg.strength import symmetric_strength_of_connection
from pyamg.relaxation import gauss_seidel

from aggregation import smoothed_aggregation_solver
from aggregate import standard_aggregation
from smooth import jacobi_prolongation_smoother
from tentative import fit_candidates

__all__ = ['adaptive_sa_solver']



def adaptive_sa_solver(A, num_candidates=1, candidate_iters=5, 
        improvement_iters=0, epsilon=0.1,
        max_levels=10, max_coarse=100, aggregation=None,
        **kwargs):
    """Create a multilevel solver using Adaptive Smoothed Aggregation (aSA)


    Parameters
    ----------

    A : {csr_matrix, bsr_matrix}
        Square matrix in CSR or BSR format
    num_candidates : {integer} : default 1
        Number of near-nullspace candidates to generate
    candidate_iters : {integer} : default 5
        Number of smoothing passes/multigrid cycles used at each level of 
        the adaptive setup phase
    improvement_iters : {integer} : default 0
        Number of times each candidate is improved
    epsilon : {float} : default 0.10
        Target convergence factor

    Optional Parameters
    -------------------

    max_levels: {integer}
        Maximum number of levels to be used in the multilevel solver.
    max_coarse: {integer}
        Maximum number of variables permitted on the coarse grid.
    symmetric: {boolean} : default True
        True if A is symmetric, False otherwise
    rescale: {boolean} : default True
        If True, symmetrically rescale A by the diagonal
        i.e. A -> D * A * D,  where D is diag(A)^-0.5
    aggregation: {sequence of csr_matrix objects}
        List of csr_matrix objects that describe a user-defined
        multilevel aggregation of the degrees of freedom.
        For instance [ Agg0, Agg1 ] defines a three-level hierarchy
        where the dimensions of A, Agg0 and Agg1 are compatible, i.e.
        Agg0.shape[1] == A.shape[0] and Agg1.shape[1] == Agg0.shape[0].
  
                    

    Notes
    -----
        Unlike the standard Smoothed Aggregation (SA) method, adaptive SA
        does not require knowledge of near-nullspace candidate vectors.
        Instead, an adaptive procedure computes one or more candidates 
        'from scratch'.  This approach is useful when no candidates are known
        or the candidates have been invalidated due to changes to matrix A.
        

    Example
    -------
        TODO

    References
    ----------

        Brezina, Falgout, MacLachlan, Manteuffel, McCormick, and Ruge
        "Adaptive Smoothed Aggregation ($\alpha$SA) Multigrid"
        SIAM Review Volume 47,  Issue 2  (2005)
        http://www.cs.umn.edu/~maclach/research/aSA2.pdf

    """
    
    ###
    # develop first candidate
    B,AggOps = initial_setup_stage(A, candidate_iters, epsilon, 
            max_levels, max_coarse, aggregation)

    kwargs['aggregate'] = ('predefined',AggOps)

    ###
    # develop additional candidates
    for i in range(num_candidates - 1):
        ml = smoothed_aggregation_solver(A, B=B, **kwargs)
        x = general_setup_stage(ml, candidate_iters)
        B = hstack((B,x))

    ###
    # improve candidates
    for i in range(improvement_iters):
        for i in range(B.shape[1]):
            B = B[:,1:]
            ml = smoothed_aggregation_solver(A, B=B, **kwargs)
            x = general_setup_stage(ml, candidate_iters)
            B = hstack((B,x))

    return smoothed_aggregation_solver(A, B=B, **kwargs)


def relax_candidate(A, x, candidate_iters):
    opts = kwargs.copy()
    opts['max_levels']    = 1
    opts['coarse_solver'] = None

    ml = smoothed_aggregation_solver(A, **opts)

    for i in range(candidate_iters):
        ml.presmooth(A,x,b)
        ml.postsmooth(A,x,b)
   
   

def initial_setup_stage(A, candidate_iters, epsilon, max_levels, max_coarse, aggregation):
    """Computes a complete aggregation and the first near-nullspace candidate


    """
    if aggregation is not None:
        max_coarse = 0
        max_levels = len(aggregation) + 1

    # aSA parameters
    # candidate_iters - number of test relaxation iterations
    # epsilon - minimum acceptable relaxation convergence factor

    #step 1
    A_l = A
    x   = rand(A_l.shape[0],1) # TODO see why randn() fails here
    skip_f_to_i = False

    def relax(A,x):
        gauss_seidel(A, x, zeros_like(x), iterations=candidate_iters, sweep='symmetric')

    #step 2
    relax(A_l,x)

    #step 3
    #TODO test convergence rate here

    As     = [A]
    Ps     = []
    AggOps = []

    while len(AggOps) + 1 < max_levels and A_l.shape[0] > max_coarse:
        if aggregation is None:
            C_l   = symmetric_strength_of_connection(A_l)
            AggOp = standard_aggregation(C_l)                                  #step 4b
        else:
            AggOp = aggregation[len(AggOps)]
        T_l,x = fit_candidates(AggOp,x)                                        #step 4c
        P_l   = jacobi_prolongation_smoother(A_l,T_l)                          #step 4d
        A_l   = P_l.T.asformat(P_l.format) * A_l * P_l                         #step 4e

        AggOps.append(AggOp)
        Ps.append(P_l)
        As.append(A_l)

        if A_l.shape <= max_coarse:  break

        if not skip_f_to_i:
            x_hat = x.copy()                                                   #step 4g
            relax(A_l,x)                                                       #step 4h
            x_A_x = dot(x.T,A_l*x)
            err_ratio = (x_A_x/dot(x_hat.T,A_l*x_hat))**(1.0/candidate_iters) 
            if err_ratio < epsilon:                                            #step 4i
                #print "sufficient convergence, skipping"
                skip_f_to_i = True
                if x_A_x == 0:
                    x = x_hat  #need to restore x

    # extend coarse-level candidate to the finest level
    for A_l,P in reversed(zip(As[1:],Ps)):
        relax(A_l,x)
        x = P * x
    relax(A,x)

    return x,AggOps  #first candidate


def general_setup_stage(ml, candidate_iters):
    levels = ml.levels

    x = rand(levels[0].A.shape[0],1)
    b = zeros_like(x)

    x = ml.solve(b, x0=x, tol=1e-10, maxiter=candidate_iters)

    #TEST FOR CONVERGENCE HERE

    def make_bridge(P):
        M,N  = P.shape
        K    = P.blocksize[0]
        bnnz = P.indptr[-1]
        data = zeros( (bnnz, K+1, K), dtype=P.dtype )
        data[:,:-1,:] = P.data
        return bsr_matrix( (data, P.indices, P.indptr), shape=( (K+1)*(M/K), N) )

    def expand_candidates(B_old,x):
        B = zeros( (x.shape[0], B_old.shape[1] + 1), dtype=x.dtype)

        B[:B_old.shape[0],:B_old.shape[1]] = B_old
        B[:,-1] = x.reshape(-1)
        return B

    for i in range(len(ml.levels) - 2):
        # add candidate to B
        B = expand_candidates(levels[i].B,x)

        T,R = fit_candidates(levels[i].AggOp,B)
        x = R[:,-1].reshape(-1,1)

        levels[i].P   = jacobi_prolongation_smoother(levels[i].A,T)
        levels[i].R   = levels[i].P.T.asformat(levels[i].P.format)
        levels[i+1].A = levels[i].R * levels[i].A * levels[i].P
        levels[i+1].P = make_bridge(levels[i+1].P) 
        levels[i+1].R = levels[i+1].P.T.asformat(levels[i+1].P.format)

        solver = multilevel_solver(levels[i+1:])
        x = solver.solve(zeros_like(x), x0=x, tol=1e-12, maxiter=candidate_iters)

    for lvl in reversed(levels[:-2]):
        x = lvl.P * x
        gauss_seidel(lvl.A, x, zeros_like(x), iterations=candidate_iters, sweep='symmetric')

    return x


