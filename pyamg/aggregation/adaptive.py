"""Adaptive Smoothed Aggregation"""

__docformat__ = "restructuredtext en"

from numpy import sqrt, ravel, diff, zeros, ones, zeros_like, inner, concatenate, \
                  asarray, hstack, ascontiguousarray, isinf, dot
from numpy.random import randn, rand
from scipy.sparse import csr_matrix, coo_matrix, bsr_matrix, isspmatrix_csr

from pyamg.multilevel import multilevel_solver
from pyamg.strength import symmetric_strength_of_connection
from pyamg.relaxation import gauss_seidel, kaczmarz_gauss_seidel
from pyamg.relaxation.smoothing import setup_smoothers
import pyamg.relaxation

from aggregation import smoothed_aggregation_solver
from aggregate import standard_aggregation
from smooth import jacobi_prolongation_smoother, energy_prolongation_smoother, \
     kaczmarz_richardson_prolongation_smoother, kaczmarz_jacobi_prolongation_smoother
from tentative import fit_candidates

__all__ = ['adaptive_sa_solver']



def adaptive_sa_solver(A, num_candidates=1, candidate_iters=5, 
        improvement_iters=0, epsilon=0.1,
        max_levels=10, max_coarse=100, aggregation=None,
        prepostsmoother=('gauss_seidel', {'sweep':'symmetric'}),
        smooth=('energy', {'degree' : 1}),
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
    prepostsmoother : {string or dict} 
        Pre- and post-smoother used in the adaptive method
    smooth : {string or dict }
        ['jacobi', 'richardson', 'energy', 'kaczmarz_jacobi', 'kaczmarz_richardson', None]
        Method used used to smooth the tentative prolongator.
    
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
            max_levels, max_coarse, aggregation, prepostsmoother, smooth)
    # Normalize B
    B = (1.0/sqrt(inner(ravel(B), ravel(B))))*B
    
    kwargs['aggregate'] = ('predefined',AggOps)

    ###
    # develop additional candidates
    for i in range(num_candidates - 1):
        x = general_setup_stage( smoothed_aggregation_solver(A, B=B, presmoother=prepostsmoother, 
                                                            postsmoother=prepostsmoother, smooth=smooth, **kwargs), 
                                candidate_iters, prepostsmoother, smooth)
        
        # Normalize x and add to candidate list
        x = (1.0/sqrt(inner(ravel(x), ravel(x))))*x
        B = hstack((B,x))

    ###
    # improve candidates
    for i in range(improvement_iters):
        for j in range(B.shape[1]):
            B = B[:,1:]
            x = general_setup_stage( smoothed_aggregation_solver(A, B=B, presmoother=prepostsmoother, 
                                                                 postsmoother=prepostsmoother, smooth=smooth,**kwargs), 
                                     candidate_iters, prepostsmoother, smooth)
            
            # Normalize x and add to candidate list
            x = (1.0/sqrt(inner(ravel(x), ravel(x))))*x
            B = hstack((B,x))

    return smoothed_aggregation_solver(A, B=B, presmoother=prepostsmoother, 
                                       postsmoother=prepostsmoother, smooth=smooth,**kwargs)


#def relax_candidate(A, x, candidate_iters, prepostsmoother, smooth):
#    #Currently this fcn is not called anywhere, can this be removed?  
#    opts = kwargs.copy()
#    opts['max_levels']    = 1
#    opts['coarse_solver'] = None
#
#    ml = smoothed_aggregation_solver(A, presmoother=prepostsmoother, 
#                                     postsmoother=prepostsmoother, smooth=smooth,**opts)
#
#    for i in range(candidate_iters):
#        ml.presmooth(A,x,b)
#        ml.postsmooth(A,x,b)
   
   

def initial_setup_stage(A, candidate_iters, epsilon, max_levels, max_coarse, aggregation, prepostsmoother, smooth):
    """Computes a complete aggregation and the first near-nullspace candidate


    """
    def unpack_arg(v):
        if isinstance(v,tuple):
            return v[0],v[1]
        else:
            return v,{}

    if aggregation is not None:
        max_coarse = 0
        max_levels = len(aggregation) + 1

    # aSA parameters
    # candidate_iters - number of test relaxation iterations
    # epsilon - minimum acceptable relaxation convergence factor

    #step 1
    A_l = A
    x   = randn(A_l.shape[0],1) # TODO see why randn() fails here
    skip_f_to_i = False

    def relax(A,x):
        def unpack_arg(v):
            if isinstance(v,tuple):
                return v[0],v[1]
            else:
                return v,{}

        fn, kwargs = unpack_arg(prepostsmoother)
        if fn == 'gauss_seidel':
            gauss_seidel(A, x, zeros_like(x), iterations=candidate_iters, sweep='symmetric')
        elif fn == 'kaczmarz_gauss_seidel':
            kaczmarz_gauss_seidel(A, x, zeros_like(x), iterations=candidate_iters, sweep='symmetric')
        else:
            raise TypeError('Unrecognized smoother')
    
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
        
        if False:                                                              #step 4d
            #P_l   = jacobi_prolongation_smoother(A_l,T_l)                          
            P_l   = energy_prolongation_smoother(A_l, T_l, None, x, degree=1)
            #if len(AggOps) == 0:
            #    P_l   = energy_prolongation_smoother(A_l, T_l, None, x, degree=2)
            #else:
            #    P_l   = energy_prolongation_smoother(A_l, T_l, None, x, degree=1)
        else: 
            fn, kwargs = unpack_arg(smooth)
            if fn == 'jacobi':
                P_l = jacobi_prolongation_smoother(A_l, T_l, **kwargs)
            elif fn == 'richardson':
                P_l = richardson_prolongation_smoother(A_l, T_l, **kwargs)
            elif fn == 'energy':
                P_l = energy_prolongation_smoother(A_l, T_l, None, x, **kwargs)
            elif fn == 'kaczmarz_richardson':
                P_l = kaczmarz_richardson_prolongation_smoother(A_l, T_l, **kwargs)
            elif fn == 'kaczmarz_jacobi':
                P_l = kaczmarz_jacobi_prolongation_smoother(A_l, T_l, **kwargs)
            elif fn == None:
                P_l = T_l
            else:
                raise ValueError('unrecognized prolongation smoother method %s' % str(fn))

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
                print "sufficient convergence, skipping"
                skip_f_to_i = True
                if x_A_x == 0:
                    x = x_hat  #need to restore x

    # extend coarse-level candidate to the finest level
    for A_l,P in reversed(zip(As[1:],Ps)):
        relax(A_l,x)
        x = P * x
    relax(A,x)

    return x,AggOps  #first candidate


def general_setup_stage(ml, candidate_iters, prepostsmoother, smooth):
         
    def unpack_arg(v):
        if isinstance(v,tuple):
            return v[0],v[1]
        else:
            return v,{}
    
    levels = ml.levels

    x = randn(levels[0].A.shape[0],1)
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
        
        if False:
            #levels[i].P   = jacobi_prolongation_smoother(levels[i].A, T)
            levels[i].P   = energy_prolongation_smoother(levels[i].A, T, None, R, degree=1)
            #if i == 0:
            #    levels[i].P   = energy_prolongation_smoother(levels[i].A, T, None, R, degree=2)
            #else:
            #    levels[i].P   = energy_prolongation_smoother(levels[i].A, T, None, R, degree=1)
        else: 
            fn, kwargs = unpack_arg(smooth)
            if fn == 'jacobi':
                levels[i].P = jacobi_prolongation_smoother(levels[i].A, T, **kwargs)
            elif fn == 'richardson':
                levels[i].P = richardson_prolongation_smoother(levels[i].A, T, **kwargs)
            elif fn == 'energy':
                levels[i].P = energy_prolongation_smoother(levels[i].A, T, None, R, **kwargs)
            elif fn == 'kaczmarz_richardson':
                levels[i].P = kaczmarz_richardson_prolongation_smoother(levels[i].A, T, **kwargs)
            elif fn == 'kaczmarz_jacobi':
                levels[i].P = kaczmarz_jacobi_prolongation_smoother(levels[i].A, T, **kwargs)
            elif fn == None:
                levels[i].P = T
            else:
                raise ValueError('unrecognized prolongation smoother method %s' % str(fn))
        
        levels[i].R   = levels[i].P.T.asformat(levels[i].P.format)
        levels[i+1].A = levels[i].R * levels[i].A * levels[i].P
        levels[i+1].P = make_bridge(levels[i+1].P) 
        levels[i+1].R = levels[i+1].P.T.asformat(levels[i+1].P.format)

        solver = multilevel_solver(levels[i+1:])
        setup_smoothers(solver, presmoother=prepostsmoother, postsmoother=prepostsmoother)

        x = solver.solve(zeros_like(x), x0=x, tol=1e-12, maxiter=candidate_iters)


    def unpack_arg(v):
        if isinstance(v,tuple):
            return v[0],v[1]
        else:
            return v,{}
    
    fn, kwargs = unpack_arg(prepostsmoother)
    for lvl in reversed(levels[:-2]):
        x = lvl.P * x
        if fn == 'gauss_seidel':
            gauss_seidel(lvl.A, x, zeros_like(x), iterations=candidate_iters, sweep='symmetric')
        elif fn == 'kaczmarz_gauss_seidel':
            kaczmarz_gauss_seidel(lvl.A, x, zeros_like(x), iterations=candidate_iters, sweep='symmetric')
        else:
            raise TypeError('Unrecognized smoother')

    
    return x


