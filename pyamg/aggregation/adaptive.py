"""Adaptive Smoothed Aggregation"""

__docformat__ = "restructuredtext en"

from numpy import sqrt, ravel, diff, zeros, zeros_like, inner, concatenate, \
                  asarray, hstack, ascontiguousarray, isinf, dot
from numpy.random import randn, rand
from scipy.sparse import csr_matrix, coo_matrix, bsr_matrix

from pyamg.multilevel import multilevel_solver
from pyamg.strength import symmetric_strength_of_connection
from pyamg.relaxation import gauss_seidel
from pyamg.utils import approximate_spectral_radius, diag_sparse, norm

from aggregate import standard_aggregation
from smooth import jacobi_prolongation_smoother
from tentative import fit_candidates

__all__ = ['adaptive_sa_solver']


def sa_hierarchy(A,B,AggOps):
    """
    Construct multilevel hierarchy using Smoothed Aggregation
        Inputs:
          A  - matrix
          B  - fine-level near nullspace candidates be approximated

        Ouputs:
          (As,Ps,Ts,Bs) - tuple of lists
                  - As -  
                  - Ps - smoothed prolongators
                  - Ts - tentative prolongators
                  - Bs - near nullspace candidates
    """
    As = [A]
    Ps = []
    Ts = []
    Bs = [B]

    for AggOp in AggOps:
        P,B = fit_candidates(AggOp,B)
        I   = jacobi_prolongation_smoother(A,P)
        A   = I.T.asformat(I.format) * A * I
        As.append(A)
        Ts.append(P)
        Ps.append(I)
        Bs.append(B)
    return As,Ps,Ts,Bs


def adaptive_sa_solver(A, num_candidates=1, candidate_iters=5, improvement_iters=0, 
        max_levels=10, max_coarse=100, epsilon=0.1, theta=0.0, omega=4.0/3.0,  
        symmetric=True, rescale=True,  aggregation=None):
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

    Optional Parameters
    -------------------

    max_levels: {integer}
        Maximum number of levels to be used in the multilevel solver.
    max_coarse: {integer}
        Maximum number of variables permitted on the coarse grid.
    epsilon : {float} : default 0.10
        Target convergence factor
    theta: {float}
        Strength of connection parameter used in aggregation.
    omega: {float} : default 4.0/3.0
        Damping parameter used in prolongator smoothing (0 < omega < 2)
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
        SIAM Review Volume 47 ,  Issue 2  (2005)
        http://www.cs.umn.edu/~maclach/research/aSA2.pdf

    """
    
    if A.shape[0] <= max_coarse:
        return multilevel_solver( [A], [] )

    ###
    # develop first candidate
    x,AggOps = initial_setup_stage(A, candidate_iters,
            max_levels=max_levels, max_coarse=max_coarse, 
            epsilon=epsilon, theta=theta, aggregation=aggregation )

    #TODO make fit_candidates work for small Bs
    x /= norm(x)
    
    # create SA hierarchy using first candidate
    As,Ps,Ts,Bs = sa_hierarchy(A,x,AggOps)

    ###
    # develop additional candidates
    for i in range(num_candidates - 1):
        x = general_setup_stage(As, Ps, Ts, Bs, AggOps, candidate_iters)
        x /= norm(x)  #TODO fix in fit_candidates
        B = hstack((Bs[0],x))
        As,Ps,Ts,Bs = sa_hierarchy(A,B,AggOps)

    ###
    # improve candidates
    for i in range(improvement_iters):
        B = Bs[0]
        for i in range(B.shape[1]):
            B = B[:,1:]
            As,Ps,Ts,Bs = sa_hierarchy(A,B,AggOps)
            x = general_setup_stage(As,Ps,Ts,Bs,AggOps,mu)
            B = hstack((B,x))
        As,Ps,Ts,Bs = sa_hierarchy(A,B,AggOps)

    #TODO use levels/smoothed_aggregation_solver throughout method
    class asa_level:
        pass

    levels = []
    for A,P,T,B,AggOp in zip(As[:-1],Ps,Ts,Bs,AggOps):
        levels.append( asa_level() )
        levels[-1].A = A
        levels[-1].P = P
        levels[-1].T = T
        levels[-1].B = B
        levels[-1].AggOp = AggOp
    levels.append( asa_level() )
    levels[-1].A = As[-1]

    return multilevel_solver(levels)

def initial_setup_stage(A, candidate_iters, max_levels, max_coarse, epsilon, theta, aggregation):
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
            C_l   = symmetric_strength_of_connection(A_l,theta)
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

    return x,AggOps  #first candidate,aggregation


def general_setup_stage(As, Ps, Ts, Bs, AggOps, candidate_iters):
    A = As[0]

    x = rand(A.shape[0],1)
    b = zeros_like(x)

    x = multilevel_solver(As,Ps).solve(b, x0=x, tol=1e-10, maxiter=candidate_iters)

    #TEST FOR CONVERGENCE HERE

    temp_Ps = []
    temp_As = [A]

    def make_bridge(P):
        M,N  = P.shape
        K    = P.blocksize[0]
        bnnz = P.indptr[-1]
        data = zeros( (bnnz, K+1, K), dtype=P.dtype )
        data[:,:-1,:] = P.data
        return bsr_matrix( (data, P.indices, P.indptr), shape=( (K+1)*(M/K), N) )

    for i in range(len(As) - 2):
        B_old = Bs[i]
        B = zeros( (x.shape[0], B_old.shape[1] + 1), dtype=x.dtype)

        B[:B_old.shape[0],:B_old.shape[1]] = B_old
        B[:,-1] = x.reshape(-1)

        T,R = fit_candidates(AggOps[i],B)

        P = jacobi_prolongation_smoother(A,T)
        A = P.T.asformat(P.format) * A * P

        temp_Ps.append(P)
        temp_As.append(A)

        bridge = make_bridge(Ps[i+1])

        solver = multilevel_solver( [A] + As[i+2:], [bridge] + Ps[i+2:] )

        x = R[:,-1].reshape(-1,1)
        x = solver.solve(zeros_like(x), x0=x, tol=1e-8, maxiter=candidate_iters)

    for A,P in reversed(zip(temp_As,temp_Ps)):
        x = P * x
        gauss_seidel(A, x, zeros_like(x), iterations=candidate_iters, sweep='symmetric')

    return x


