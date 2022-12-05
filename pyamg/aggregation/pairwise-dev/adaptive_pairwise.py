from pyamg.multilevel import multilevel_solver, multilevel_solver_set
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
from pyamg.util.linalg import approximate_spectral_radius
from pyamg.strength import classical_strength_of_connection,\
    symmetric_strength_of_connection, evolution_strength_of_connection,\
    energy_based_strength_of_connection, distance_strength_of_connection,\
    algebraic_distance, affinity_distance
from pyamg.aggregation.aggregate import standard_aggregation, naive_aggregation,\
    lloyd_aggregation, pairwise_aggregation
from pyamg.util.utils import blocksize, relaxation_as_linear_operator
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_bsr
import numpy as np
import scipy
import pdb

# ----------------------------- To do ----------------------------- #
# - Does aggregate.py need to include strength functions if they are passed in through kwargs?
# - Should probably use bad guy to construct coarse grids in multiple pairwise matchings
# - Set up SA to do various forms of weighted tentative prolongation 
# - Generalize pairwise aggregation routines to non-symmetric matrices 
# - Motivate choice of C-points based on something... 
#   + Note, this is only matters for e.g. root-node. Should not 
#     use extra work to compute when used for SA-type solvers. 
# - Need to distinguish in pairwise_aggregation() if the user passes in a user-written 
#   function or a reference to a pyAMG SOC function...
# - Check angle of theta in Jacob's diffusion again - CW or counter-CW?
# - Check Preis alg. with Pasqua's implementation
# - Check Notay approximation of minimum 
# - Should only compute matching on finest level once, because it will always be the same.
# ---> Does Panayot smooth and develop a new bad guy on every level? Yes, doesn't seem to help though...

# SHOULD MAYBE MOVE SMOOTHING OF TARGETS TO HERE INSTEAD OF IN SA SO THAT WE CAN KEEP TRACK OF WHAT TARGETS ARE USED.

def reconstruct_hierarchy(solver_set, A, new_B, symmetry,
                          aggregate, presmoother,
                          postsmoother, smooth,
                          strength, max_levels,
                          max_coarse, coarse_solver,
                          diagonal_dominance, keep, **kwargs):

    if not isinstance(solver_set, multilevel_solver_set):
        raise TypeError("Must pass in multilevel solver set.")

    if not isinstance(new_B, np.ndarray):
        raise ValueError("Target vectors must be ndarray of size nxb, for b hierarchies and problem size n.")

    num_solvers = solver_set.num_hierarchies
    num_badguys = new_B.shape[1]
    n = new_B.shape[0]
    if (n != A.shape[0]):
        raise ValueError("Target vectors must have same size as matrix.")

    # If less target vectors provided than solvers in set, remove solvers without target
    # vector to reconstruct.
    diff = num_solvers - num_badguys
    if (diff > 0):
        print "Less target vectors provided than hierachies in solver. Removing hierarchies."
        for i in range(0,diff):
            solver_set.remove_hierarchy(num_solvers-1)
            num_solvers -= 1

    # Reconstruct each solver in hierarchy using new target vectors
    print "Reconstructing hierarchy."
    for i in range(0,num_solvers):
        solver_set.replace_hierarchy(hierarchy=smoothed_aggregation_solver(A, B=new_B[:,i],
                                                                           symmetry=symmetry,
                                                                           aggregate=aggregate,
                                                                           presmoother=presmoother,
                                                                           postsmoother=postsmoother,
                                                                           smooth=smooth, strength=strength,
                                                                           max_levels=max_levels,
                                                                           max_coarse=max_coarse,
                                                                           coarse_solver=coarse_solver,
                                                                           diagonal_dominance=diagonal_dominance,
                                                                           improve_candidates=None, keep=keep,
                                                                           **kwargs),
                                     ind=i )

    # If more bad guys are provided than stored in initial hierarchy
    if diff < 0:
        print "More target vectors provided than hierachies in solver. Adding hierarchies."
        for i in range(num_solvers,num_badguys):
            solver_set.add_hierarchy(hierarchy=smoothed_aggregation_solver(A, B=new_B[:,i],
                                                                           symmetry=symmetry,
                                                                           aggregate=aggregate,
                                                                           presmoother=presmoother,
                                                                           postsmoother=postsmoother,
                                                                           smooth=smooth, strength=strength,
                                                                           max_levels=max_levels,
                                                                           max_coarse=max_coarse,
                                                                           coarse_solver=coarse_solver,
                                                                           diagonal_dominance=diagonal_dominance,
                                                                           improve_candidates=None, keep=keep,
                                                                           **kwargs)  )



# Function to compute dot product and A-norm of bad guys
def test_targets(A,B):

    num = B.shape[1]
    # Get A-norm of each target
    for i in range(0,num):
        norm = np.dot(B[:,i].T*A, B[:,i]) / np.dot(B[:,i].T,B[:,i])
        print "A-norm of target ",i," = ",norm

    angles = np.zeros((num,num))
    # Get angle between targets
    print "\nAngle between targets in A-norm - "
    for i in range(0,num):
        norm_i = np.sqrt( np.dot(B[:,i].T*A, B[:,i]) )
        for j in range(i,num):
            norm_j = np.sqrt( np.dot(B[:,j].T,B[:,j]) )
            ang = np.dot(B[:,i].T*A, B[:,j]) / (norm_j*norm_i)
            try:
                angles[i,j] = np.arccos(ang)
            except:
                angles[i,j] = -1
                pdb.set_trace()

    np.set_printoptions(precision=3, suppress=True)
    print angles,"\n"
    return angles


def global_ritz_process(A, B1, B2=None, weak_tol=15., level=0, verbose=False):
    """
    Helper function that compresses two sets of targets B1 and B2 into one set
    of candidates. This is the Ritz procedure.

    Parameters
    ---------
    A : {sparse matrix}
        SPD matrix used to compress the candidates so that the weak
        approximation property is satisfied.
    B1 : {array}
        n x m1 array of m1 potential candidates
    B2 : {array}
        n x m2 array of m2 potential candidates
    weak_tol : {float}
        The constant in the weak approximation property.

    Returns
    -------
    New set of candidates forming an Euclidean orthogonal and energy
    orthonormal subset of span(B1,B2). The candidates that trivially satisfy
    the weak approximation property are deleted.
    """

    if B2 is not None:
        B = np.hstack((B1, B2.reshape(-1,1)))
    else:
        B = B1

    # Orthonormalize the vectors.
    [Q,R] = scipy.linalg.qr(B, mode='economic')

    # Formulate and solve the eigenpairs problem returning eigenvalues in
    # ascending order.
    QtAQ = scipy.dot(Q.conjugate().T, A*Q)        # WAP  
    [E,V] = scipy.linalg.eigh(QtAQ)

    # Make sure eigenvectors are real. Eigenvalues must be already real.
    try:
        V = np.real(V)
    except:
        import pdb; pdb.set_trace()

    # Compute Ritz vectors and normalize them in energy. Also, mark vectors
    # that trivially satisfy the weak approximation property.
    V = scipy.dot(Q, V)
    num_candidates = -1
    entire_const = weak_tol / approximate_spectral_radius(A)
    if verbose:
        print
        print tabs(level), "WAP const", entire_const
    for j in range(V.shape[1]):
        V[:,j] /= np.sqrt(E[j])
        # verify energy norm is 1
        if verbose:
            print tabs(level), "Vector 1/e", j, 1./E[j], "ELIMINATED" if 1./E[j] <= entire_const else ""
        if 1./E[j] <= entire_const:
            num_candidates = j
            break

    if num_candidates == 0:
        num_candidates = 1
    if num_candidates == -1:
        num_candidates = V.shape[1]

    if verbose:
        # print tabs(level), "Finished global ritz process, eliminated", B.shape[1]-num_candidates, "candidates", num_candidates, ". candidates remaining"
        print

    return V[:, :num_candidates]


# Auxillary function to compute aggregates on finest level
def get_aggregate(A, strength, aggregate, diagonal_dominance, B, **kwargs):

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        else:
            return v, {}

    # Compute the strength-of-connection matrix C, where larger
    # C[i,j] denote stronger couplings between i and j.
    fn, kwargs = unpack_arg(strength)
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(A, **kwargs)
    elif fn == 'classical':
        C = classical_strength_of_connection(A, **kwargs)
    elif fn == 'distance':
        C = distance_strength_of_connection(A, **kwargs)
    elif (fn == 'ode') or (fn == 'evolution'):
        if 'B' in kwargs:
            C = evolution_strength_of_connection(A, **kwargs)
        else:
            C = evolution_strength_of_connection(A, B, **kwargs)
    elif fn == 'energy_based':
        C = energy_based_strength_of_connection(A, **kwargs)
    elif fn == 'predefined':
        C = kwargs['C'].tocsr()
    elif fn == 'algebraic_distance':
        C = algebraic_distance(A, **kwargs)
    elif fn == 'affinity':
        C = affinity_distance(A, **kwargs)
    elif fn is None:
        C = A.tocsr()
    else:
        raise ValueError('unrecognized strength of connection method: %s' %
                         str(fn))

    # Avoid coarsening diagonally dominant rows
    flag, kwargs = unpack_arg(diagonal_dominance)
    if flag:
        C = eliminate_diag_dom_nodes(A, C, **kwargs)

    # Compute the aggregation matrix AggOp (i.e., the nodal coarsening of A).
    # AggOp is a boolean matrix, where the sparsity pattern for the k-th column
    # denotes the fine-grid nodes agglomerated into k-th coarse-grid node.
    fn, kwargs = unpack_arg(aggregate)
    if fn == 'standard':
        AggOp = standard_aggregation(C, **kwargs)[0]
    elif fn == 'naive':
        AggOp = naive_aggregation(C, **kwargs)[0]
    elif fn == 'lloyd':
        AggOp = lloyd_aggregation(C, **kwargs)[0]
    elif fn == 'pairwise':
        AggOp = pairwise_aggregation(A, B, **kwargs)
    elif fn == 'predefined':
        AggOp = kwargs['AggOp'].tocsr()
    else:
        raise ValueError('unrecognized aggregation method %s' % str(fn))

    return AggOp


def adaptive_pairwise_solver(A, B=None, symmetry='hermitian',
					  desired_convergence=0.7,
                      test_iterations = 10, 
                      test_cycle = 'V',
                      test_accel = None,
                      strength = None,
                      smooth = None,
                      aggregate = ('drake', {'levels': 2}),
					  presmoother=('block_gauss_seidel',
                                   {'sweep': 'symmetric'}),
                      postsmoother=('block_gauss_seidel',
                                   {'sweep': 'symmetric'}),
                      max_levels=30, max_coarse=100,
                      diagonal_dominance=False,
                      coarse_solver='pinv', keep=False,
                      additive=False, reconstruct=False,
                      max_hierarchies=10, use_ritz=False,
                      improve_candidates=[('block_gauss_seidel',
                                         {'sweep': 'symmetric',
                                         'iterations': 4})],
                      **kwargs):

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        elif v is None:
            return None
        else:
            return v, {}

    if isspmatrix_bsr(A):
        warn("Only currently implemented for CSR matrices.")

    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        try:
            A = csr_matrix(A)
            warn("Implicit conversion of A to CSR",
                 SparseEfficiencyWarning)
        except:
            raise TypeError('Argument A must have type csr_matrix or\
                             bsr_matrix, or be convertible to csr_matrix')

    if (symmetry != 'symmetric') and (symmetry != 'hermitian') and\
            (symmetry != 'nonsymmetric'):
        raise ValueError('expected \'symmetric\', \'nonsymmetric\' or\
                         \'hermitian\' for the symmetry parameter ')

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    A = A.asfptype()
    A.symmetry = symmetry
    n = A.shape[0]
    test_rhs = np.zeros((n,1))

    # SHOULD I START WITH CONSTANT VECTOR OR SMOOTHED RANDOM VECTOR?
    # Right near nullspace candidates
    if B is None:
        B = np.kron(np.ones((A.shape[0]/blocksize(A), 1), dtype=A.dtype),
                    np.eye(blocksize(A)))
    else:
        B = np.asarray(B, dtype=A.dtype)
        if len(B.shape) == 1:
            B = B.reshape(-1, 1)
        if B.shape[0] != A.shape[0]:
            raise ValueError('The near null-space modes B have incorrect \
                              dimensions for matrix A')
        if B.shape[1] < blocksize(A):
            raise ValueError('B.shape[1] must be >= the blocksize of A')

    # Improve near nullspace candidates by relaxing on A B = 0
    if improve_candidates is not None:
        fn, temp_args = unpack_arg(improve_candidates[0])
    else:
        fn = None

    if fn is not None:
        b = np.zeros((A.shape[0], 1), dtype=A.dtype)
        B = relaxation_as_linear_operator((fn, temp_args), A, b) * B
        if A.symmetry == "nonsymmetric":
            AH = A.H.asformat(A.format)
            BH = relaxation_as_linear_operator((fn, temp_args), AH, b) * BH

    # Empty set of solver hierarchies 
    solvers = multilevel_solver_set()
    target = B
    B = B
    cf = 1.0

    # Aggregation process on the finest level is the same each iteration.
    # To prevent repeating processes, we compute it here and provide it to the 
    # solver construction. 
    # ---> NOTE THIS IS ONLY TRUE IN CASE OF SINGLE PAIRWISE AGGREGATION...
    # AggOp = get_aggregate(A, strength=strength, aggregate=aggregate,
    #                       diagonal_dominance=diagonal_dominance, B=B)
    # if isinstance(aggregate,tuple):
    #     aggregate = [('predefined', {'AggOp': AggOp}), aggregate]
    # elif isinstance(aggregate,list):
    #     aggregate.insert(0, ('predefined', {'AggOp': AggOp}))
    # else:
    #     raise TypeError("Aggregate variable must be list or tuple.")

    # Continue adding hierarchies until desired convergence factor achieved,
    # or maximum number of hierarchies constructed
    it = 0
    cfs = [1]

    while (cf > desired_convergence) and (it < max_hierarchies):

        # Make target vector orthogonal and energy orthonormal and reconstruct hierarchy
        if use_ritz and it>0:
            B = global_ritz_process(A, B, weak_tol=100)
            reconstruct_hierarchy(solver_set=solvers, A=A, new_B=B, symmetry=symmetry,
                                aggregate=aggregate, presmoother=presmoother,
                                postsmoother=postsmoother, smooth=smooth,
                                strength=strength, max_levels=max_levels,
                                max_coarse=max_coarse, coarse_solver=coarse_solver,
                                diagonal_dominance=diagonal_dominance,
                                keep=keep, **kwargs)
            print "Hierarchy reconstructed."          
        # Otherwise just add new hierarchy to solver set.
        else:
            pdb.set_trace()
            solvers.add_hierarchy( smoothed_aggregation_solver(A, B=B[:,0:1], symmetry=symmetry,
                                                               aggregate=aggregate,
                                                               presmoother=presmoother,
                                                               postsmoother=postsmoother,
                                                               smooth=smooth, strength=strength,
                                                               max_levels=max_levels,
                                                               max_coarse=max_coarse,
                                                               diagonal_dominance=diagonal_dominance,
                                                               coarse_solver=coarse_solver,
                                                               improve_candidates=improve_candidates,
                                                               keep=keep, **kwargs) )
        # Test for convergence factor using new hierarchy.
        x0 = np.random.rand(n,1)
        residuals = []
        target = solvers.solve(test_rhs, x0=x0, tol=1e-12, maxiter=test_iterations,
                               cycle=test_cycle, accel=test_accel, residuals=residuals,
                               additive=additive)
        cf = residuals[-1]/residuals[-2]
        cfs.append(cf)
        B = np.hstack((target,B))  
        # TEST IDEA TO REMOVE HIERARCHY IF IT DOESNT IMPROVE CONVERGENCE
        # if cfs[-1] >= cfs[-2]:
        #     solvers.remove_hierarchy(it)
        #     print "Hierarchy not added, did not improve convergence factor."
        # else:
        #     print "Added new hierarchy, convergence factor = ",cf
            # it += 1
        print "Added new hierarchy, convergence factor = ",cf       
        it += 1




    B = B[:,:-1]
    # B2 = global_ritz_process(A, B, weak_tol=1.0)
    angles = test_targets(A, B)
    # angles = test_targets(A, B2)

    # -------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------- #

    # b = np.zeros((n,1))
    # asa_residuals = []
    # sol = solvers.solve(b, x0, tol=1e-8, residuals=asa_residuals, accel=None)
    # asa_conv_factors = np.zeros((len(asa_residuals)-1,1))
    # for i in range(0,len(asa_residuals)-1):
    #   asa_conv_factors[i] = asa_residuals[i]/asa_residuals[i-1]

    # print "Original adaptive SA/AMG - ", np.mean(asa_conv_factors[1:])


    # if reconstruct:
    #     reconstruct_hierarchy(solver_set=solvers, A=A, new_B=B2, symmetry=symmetry,
    #                         aggregate=aggregate, presmoother=presmoother,
    #                         postsmoother=postsmoother, smooth=smooth,
    #                         strength=strength, max_levels=max_levels,
    #                         max_coarse=max_coarse, coarse_solver=coarse_solver,
    #                         diagonal_dominance=diagonal_dominance,
    #                         keep=keep, **kwargs)
    #     print "Hierarchy reconstructed."


    # asa_residuals2 = []
    # sol = solvers.solve(b, x0, tol=1e-8, residuals=asa_residuals2, accel=None)
    # asa_conv_factors2 = np.zeros((len(asa_residuals2)-1,1))
    # for i in range(0,len(asa_residuals2)-1):
    #   asa_conv_factors2[i] = asa_residuals2[i]/asa_residuals2[i-1]

    # print "Ritz adaptive SA/AMG - ", np.mean(asa_conv_factors2[1:])


    # if reconstruct:
    #     reconstruct_hierarchy(solver_set=solvers, A=A, new_B=B[:,:-1], symmetry=symmetry,
    #                         aggregate=aggregate, presmoother=presmoother,
    #                         postsmoother=postsmoother, smooth=smooth,
    #                         strength=strength, max_levels=max_levels,
    #                         max_coarse=max_coarse, coarse_solver=coarse_solver,
    #                         diagonal_dominance=diagonal_dominance,
    #                         keep=keep, **kwargs)
    #     print "Hierarchy reconstructed."


    # asa_residuals2 = []
    # sol = solvers.solve(b, x0, tol=1e-8, residuals=asa_residuals2, accel=None)
    # asa_conv_factors2 = np.zeros((len(asa_residuals2)-1,1))
    # for i in range(0,len(asa_residuals2)-1):
    #   asa_conv_factors2[i] = asa_residuals2[i]/asa_residuals2[i-1]

    # print "Original(-1) SA/AMG - ", np.mean(asa_conv_factors2[1:])


    # if reconstruct:
    #     reconstruct_hierarchy(solver_set=solvers, A=A, new_B=B2[:,:-1], symmetry=symmetry,
    #                         aggregate=aggregate, presmoother=presmoother,
    #                         postsmoother=postsmoother, smooth=smooth,
    #                         strength=strength, max_levels=max_levels,
    #                         max_coarse=max_coarse, coarse_solver=coarse_solver,
    #                         diagonal_dominance=diagonal_dominance,
    #                         keep=keep, **kwargs)
    #     print "Hierarchy reconstructed."


    # asa_residuals2 = []
    # sol = solvers.solve(b, x0, tol=1e-8, residuals=asa_residuals2, accel=None)
    # asa_conv_factors2 = np.zeros((len(asa_residuals2)-1,1))
    # for i in range(0,len(asa_residuals2)-1):
    #   asa_conv_factors2[i] = asa_residuals2[i]/asa_residuals2[i-1]

    # print "Ritz(-1) SA/AMG - ", np.mean(asa_conv_factors2[1:])


    # pdb.set_trace()
    # -------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------------------- #

    return solvers



