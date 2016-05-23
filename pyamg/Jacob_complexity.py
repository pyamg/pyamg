from numpy import log10, abs, sum, zeros, mean
from pyamg.util.utils import levelize_strength_or_aggregation, \
    levelize_smooth_or_improve_candidates
from pyamg import multilevel_solver

__all__ = ['cycle_complexity', 'setup_complexity']

def unpack_arg(v):
    if isinstance(v,tuple):
        return v[0],v[1]
    else:
        return v,{}

def setup_complexity(sa, strength, smooth, improve_candidates, aggregate, presmoother,
        postsmoother, keep, max_levels, max_coarse, coarse_solver, symmetry):
    '''
    Given a solver hierarchy, sa, and all of the setup parameters,
    compute abstractly the "work" required to form the solver hierarchy
    '''
    max_levels, max_coarse, strength = levelize_strength_or_aggregation(strength, max_levels, max_coarse)
    max_levels, max_coarse, aggregate = levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)
    improve_candidates = levelize_smooth_or_improve_candidates(improve_candidates, max_levels)
    smooth = levelize_smooth_or_improve_candidates(smooth, max_levels)
    work = 0.0
    nlevels = len(sa.levels)

    # Convert everything to list
    if isinstance(presmoother, tuple):
        presmoother = [ presmoother ]  
    if isinstance(postsmoother, tuple): 
        postsmoother = [ postsmoother ]
    if isinstance(presmoother, str):
        presmoother = [ (presmoother,{}) ]  
    if isinstance(postsmoother, str): 
        postsmoother = [ (postsmoother,{}) ]
    # Repeat final smoothing strategy till end of hiearchy
    for i in range(len(presmoother), len(sa.levels)):
        presmoother.append(presmoother[-1])
    for i in range(len(postsmoother), len(sa.levels)):
        postsmoother.append(postsmoother[-1])

    for i,lvl in enumerate(sa.levels):
        # Compute work for smoothing P
        if i < nlevels - 1:
            fn,kwargs = unpack_arg(smooth[i])
            # Account for the mat-mat mult A*P
            try:
                # This is if you're using energy-min, then 
                # account for the roughly 6 mat-mat additions
                maxiter = kwargs['maxiter']
                work += 6*lvl.P.nnz*maxiter
            except:
                maxiter = 1
            work += lvl.A.nnz*(lvl.P.nnz/float(lvl.P.shape[0]))*maxiter
            # Account for constraint enforcement mat-vec operations
            #work += lvl.P.nnz*lvl.B.shape[1]

        # Compute work for computing SoC
        if i < nlevels - 1:
            fn,kwargs = unpack_arg(strength[i])
            #work += lvl.A.nnz*(lvl.A.nnz/float(lvl.A.shape[0]))
            
            # Compute the work for kwargs['k'] > 2
            #     (nnz to compute) * (average stencil size in matrix that you're multiplying)
            if fn == 'evolution':
                work += lvl.A.nnz*( (lvl.A**(int(kwargs['k']/2))).nnz / float(lvl.A.shape[0]))
        
        # Compute work for computing RAP
        if i < nlevels -1:
            work += lvl.A.nnz*(lvl.P.nnz/float(lvl.P.shape[0]))*2
        
        # Compute work for any Schwarz relaxation
        if i < nlevels -1:
            fn1,kwargs1 = unpack_arg(presmoother[i])
            fn2,kwargs2 = unpack_arg(postsmoother[i])
            if (fn1 == 'schwarz') or (fn2 == 'schwarz'):
                S = lvl.A
            if (fn1 == 'strength_based_schwarz') or (fn2 == 'strength_based_schwarz'):
                S = lvl.C
            if( (fn1.find('schwarz') > 0) or (fn2.find('schwarz') > 0) ):
                rowlen = S.indptr[1:] - S.indptr[:-1]
                work += sum(rowlen**3)

        # Compute work for smoothing B
        if i < nlevels - 1:
            fn,kwargs = unpack_arg(improve_candidates[i])
            
            if fn is not None:
                # Compute cost multiplier for relaxation method
                cost_factor = 1
                if fn.endswith(('nr','ne')):
                    cost_factor *= 2
                if kwargs.has_key('sweep'):
                    if kwargs['sweep'] == 'symmetric':
                        cost_factor *= 2
                if kwargs.has_key('iterations'):
                    cost_factor *= kwargs['iterations']
                if kwargs.has_key('degree'):
                    cost_factor *= kwargs['degree']

                work += cost_factor*lvl.A.nnz*lvl.B.shape[1]
    
    return work / float(sa.levels[0].A.nnz)


def cycle_complexity(solver, presmoothing, postsmoothing, cycle):
    ''' Calculate cycle complexity for a given hierarchy

    Parameters
    ----------
    solver : {pyamg hierarchy}
        solver hierarchy
    presmoothing {smoother list}
        pre-smoothing strategy, see doc for change_smoothers for format
        assumed to be same length as post_smoother
    postsmoothing {smoother list}
        post-smoothing strategy, see doc for change_smoothers for format
        assumed to be same length as pre_smoother
    cycle : {character}
        'V' or 'W' or 'K' or 'AMLI'

    Returns
    -------
    Cycle Complexity
    
    Notes
    -----
    For work per digit-of-accuracy, divide cycle complexity by 
        abs(log10(r_ratio))
    
    '''
    
    # Convert everything to list
    if isinstance(presmoothing, tuple):
        presmoothing = [ presmoothing ]  
    if isinstance(postsmoothing, tuple): 
        postsmoothing = [ postsmoothing ]
    if isinstance(presmoothing, str):
        presmoothing = [ (presmoothing,{}) ]  
    if isinstance(postsmoothing, str): 
        postsmoothing = [ (postsmoothing,{}) ]
    # Repeat final smoothing strategy till end of hiearchy
    for i in range(len(presmoothing), len(solver.levels)):
        presmoothing.append(presmoothing[-1])
    for i in range(len(postsmoothing), len(solver.levels)):
        postsmoothing.append(postsmoothing[-1])

    if len(presmoothing) != len(postsmoothing):
        raise ValueError("presmoothing and postsmoothing must be same length")

    # First determine cost per nnz for the smoothing strategy on each level
    smoother_cost = []
    for i in range(len(presmoothing)):
        pre_factor = 1
        if presmoothing[i][0].endswith(('nr','ne')):
            pre_factor *= 2
        if presmoothing[i][1].has_key('sweep'):
            if presmoothing[i][1]['sweep'] == 'symmetric':
                pre_factor *= 2
        if presmoothing[i][1].has_key('iterations'):
            pre_factor *= presmoothing[i][1]['iterations']
        if presmoothing[i][1].has_key('degree'):
            pre_factor *= presmoothing[i][1]['degree']

        post_factor = 1
        if postsmoothing[i][0].endswith(('nr','ne')):
            post_factor *= 2
        if postsmoothing[i][1].has_key('sweep'):
            if postsmoothing[i][1]['sweep'] == 'symmetric':
                post_factor *= 2
        if postsmoothing[i][1].has_key('iterations'):
            post_factor *= postsmoothing[i][1]['iterations']
        if postsmoothing[i][1].has_key('degree'):
            post_factor *= postsmoothing[i][1]['degree']

        smoother_cost.append(pre_factor + post_factor)
    
    # Begin code from PyAMG (we just scale by smoother cost, instead of the 2
    # used by PyAMG
    cycle = str(cycle).upper()
    nnz = [ level.A.nnz for level in solver.levels ]
    
    # Compute work for any Schwarz relaxation 
    #   
    # The multiplier is the average row length, which is how many times each
    # the residual (on average) must be computed for each row.  This will
    # multiply the nnz.
    #
    # schwarz_work is the cost of multiplying with the
    # A[region_i, region_i]^{-1}
    schwarz_multiplier = zeros((len(presmoothing),))
    schwarz_work = zeros((len(presmoothing),))
    for i,lvl in enumerate(solver.levels[:-1]):
        fn1,kwargs1 = unpack_arg(presmoothing[i])
        fn2,kwargs2 = unpack_arg(postsmoothing[i])
        if (fn1 == 'schwarz') or (fn2 == 'schwarz'):
            S = lvl.A
        if (fn1 == 'strength_based_schwarz') or (fn2 == 'strength_based_schwarz'):
            S = lvl.C
        if( (fn1.find('schwarz') > 0) or (fn2.find('schwarz') > 0) ):
            rowlen = S.indptr[1:] - S.indptr[:-1]
            schwarz_work[i] = sum(rowlen**2)
            schwarz_multiplier[i] = mean(rowlen)
            nnz[i] = nnz[i]*schwarz_multiplier[i]
    
    def V(level):
        if len(solver.levels) == 1:
            return nnz[0]
        elif level == len(solver.levels) - 2:
            return smoother_cost[level]*nnz[level] + nnz[level + 1] + schwarz_work[level]
        else:
            return smoother_cost[level]*nnz[level] + schwarz_work[level] + V(level + 1)
    
    def W(level):
        if len(solver.levels) == 1:
            return nnz[0]
        elif level == len(solver.levels) - 2:
            return smoother_cost[level]*nnz[level] + nnz[level + 1] + schwarz_work[level]
        else:
            return smoother_cost[level]*nnz[level] + schwarz_work[level] + 2*W(level + 1)
    
    def F(level):
        if len(solver.levels) == 1:
            return nnz[0]
        elif level == len(solver.levels) - 2:
            return smoother_cost[level]*nnz[level] + nnz[level + 1] + schwarz_work[level]
        else:
            return smoother_cost[level]*nnz[level] + schwarz_work[level] + F(level + 1) + V(level + 1)

    if cycle == 'V':
        flops = V(0)
    elif (cycle == 'W') or (cycle == 'AMLI'):
        flops = W(0)
    elif cycle == 'F':
        flops = F(0)
    else:
        raise TypeError('Unrecognized cycle type (%s)' % cycle)
    
    return float(flops) / float(solver.levels[0].A.nnz)

