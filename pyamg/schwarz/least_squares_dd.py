"""Spectral Domain Decomposition - Least Squares."""


from warnings import warn
import numpy as np
from scipy.sparse import csr_array, issparse, SparseEfficiencyWarning, coo_array, csc_array, hstack
from scipy.linalg import eig, eigh

from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.util.utils import eliminate_diag_dom_nodes, get_blocksize, asfptype, \
    levelize_strength_or_aggregation, levelize_smooth_or_improve_candidates
from pyamg.strength import classical_strength_of_connection, \
    symmetric_strength_of_connection, evolution_strength_of_connection, \
    energy_based_strength_of_connection, distance_strength_of_connection, \
    algebraic_distance, affinity_distance
from pyamg.aggregation.aggregate import standard_aggregation, naive_aggregation, \
    lloyd_aggregation, balanced_lloyd_aggregation, \
    metis_aggregation, pairwise_aggregation
from pyamg import amg_core

import pdb
import time


def least_squares_dd_solver(B, BT=None, A=None,
                                       symmetry='hermitian', 
                                       strength='symmetric',
                                       aggregate='standard',
                                       presmoother=('basic'),
                                       postsmoother=('basic'),
                                       kappa=500,
                                       nev=None,
                                       threshold=None,
                                       max_levels=10, max_coarse=10,
                                       diagonal_dominance=False,
                                       filteringA=0.0,
                                       filteringB=0.0,
                                       keep=False, **kwargs):
    if A is not None:
        A_provided = True
        if not issparse(A) or A.format not in ('csr'):
            try:
                A = csr_array(A)
                warn('Implicit conversion of A to CSR', SparseEfficiencyWarning)
            except Exception as e:
                raise TypeError('Argument A must have type csr_array or bsr_array, '
                                'or be convertible to csr_array') from e
    else:
        A_provided = False

    if not issparse(B) or B.format not in ('csr'):
        try:
            B = csr_array(B)
            warn('Implicit conversion of B to CSR', SparseEfficiencyWarning)
        except Exception as e:
            raise TypeError('Argument B must have type csr_array or bsr_array, '
                            'or be convertible to csr_array') from e
            
    if BT is None:
        BT = B.T.conjugate().tocsr()
        BT.sort_indices()
        BT_provided = False
    else:
        BT_provided = True
        if not issparse(BT) or BT.format not in ('csr'):
            try:
                BT = csr_array(BT)
                warn('Implicit conversion of BT to CSR', SparseEfficiencyWarning)
            except Exception as e:
                raise TypeError('Argument BT must have type csr_array or bsr_array, '
                                'or be convertible to csr_array') from e

    B = asfptype(B)
    BT = asfptype(BT)

    if A is None:
        A = BT @ B
        A.tocsr()
        A.sort_indices()
    A = asfptype(A)
    A = A.tocsr()
    
    
    if symmetry not in ('symmetric', 'hermitian', 'nonsymmetric'):
        raise ValueError('Expected "symmetric", "nonsymmetric" or "hermitian" '
                         'for the symmetry parameter ')
    A.symmetry = symmetry

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    # Levelize the user parameters, so that they become lists describing the
    # desired user option on each level.
    max_levels, max_coarse, strength =\
        levelize_strength_or_aggregation(strength, max_levels, max_coarse)
    max_levels, max_coarse, aggregate =\
        levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)
    # improve_candidates =\
    #     levelize_smooth_or_improve_candidates(improve_candidates, max_levels)
    # smooth = levelize_smooth_or_improve_candidates(smooth, max_levels)

    # Construct multilevel structure
    levels = []
    levels.append(MultilevelSolver.Level())
    levels[-1].A = A          # Normal Equation Matrix
    levels[-1].B = B          # Least Squares Matrix
    levels[-1].BT = BT        # A is supposed to be spectrally equivalent to BT B and share the same sparsity pattern
    levels[-1].BT_provided = BT_provided
    levels[-1].A_provided = A_provided

    pre_smoother = []
    post_smoother = []
    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        _extend_hierarchy(levels, strength, aggregate, kappa, nev, threshold, diagonal_dominance, filteringA,filteringB, keep)
        print("Hierarchy extended")
        sm = ('schwarz', {'subdomain': levels[-2].subdomain, 'subdomain_ptr': levels[-2].subdomain_ptr, 'iterations':2 if len(pre_smoother)>0 else 1, 'sweep':'symmetric'})
        pre_smoother.append(sm)
        sm = ('schwarz', {'subdomain': levels[-2].subdomain, 'subdomain_ptr': levels[-2].subdomain_ptr, 'iterations':2 if len(post_smoother)>0 else 1, 'sweep':'symmetric'})
        post_smoother.append(sm)
        # smoother.append(sm)

    if(pre_smoother == []):
        pre_smoother = ('schwarz')
        post_smoother = pre_smoother
    ml = MultilevelSolver(levels, **kwargs)
    
    change_smoothers(ml, pre_smoother, post_smoother)
    # pre_smoother = ('gauss_seidel', {'sweep': 'symmetric'})
    # post_smoother = ('gauss_seidel', {'sweep': 'symmetric'})
    # pre_smoother = ('jacobi')
    # post_smoother = ('jacobi')
    change_smoothers(ml, pre_smoother, post_smoother)
    return ml


def _extend_hierarchy(levels, strength, aggregate, kappa, nev, threshold, diagonal_dominance, filteringA, filteringB, keep=True):
    """Extend the multigrid hierarchy.

    Service routine to implement the strength of connection, aggregation,
    tentative prolongation construction, and prolongation smoothing.  Called by
    smoothed_aggregation_solver.

    """
    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}

    A = levels[-1].A
    B = levels[-1].B
    BT = levels[-1].BT

    # Partition the graph
    C = abs(A).tocsr()
    for i in range(C.data.shape[0]):
        C.data[i] = 1.0

    # Compute the aggregation matrix AggOp (i.e., the nodal coarsening of A).
    # AggOp is a boolean matrix, where the sparsity pattern for the k-th column
    # denotes the fine-grid nodes agglomerated into k-th coarse-grid node.
    fn, kwargs = unpack_arg(aggregate[len(levels)-1])
    Cnodes = None
    if fn == 'standard':
        AggOp, Cnodes = standard_aggregation(C, **kwargs)
    elif fn == 'naive':
        AggOp, Cnodes = naive_aggregation(C, **kwargs)
    elif fn == 'lloyd':
        AggOp, Cnodes = lloyd_aggregation(C, **kwargs)
    elif fn == 'balanced lloyd':
        if 'pad' in kwargs:
            kwargs['A'] = A
        AggOp, Cnodes = balanced_lloyd_aggregation(C, **kwargs)
    elif fn == 'metis':
        if(len(levels) == 1):
            AggOp = metis_aggregation(C, **kwargs)
        else:
            ratio = levels[-2].N/16/levels[-1].A.shape[0]
            # ratio = max(levels[-2].nev)*4/levels[-1].A.shape[0]
            # AggOp = metis_aggregation(C, ratio=ratio)
            AggOp = metis_aggregation(C, **kwargs)
    elif fn == 'pairwise':
        AggOp = pairwise_aggregation(A, **kwargs)[0]
    elif fn == 'predefined':
        AggOp = kwargs['AggOp'].tocsr()
    else:
        raise ValueError(f'Unrecognized aggregation method {fn!s}')

    AggOp = AggOp.tocsc()
    AggOp = _remove_empty_columns(AggOp)
    AggOp = _add_columns_containing_isolated_nodes(AggOp)
    AggOp = AggOp.tocsr()
    levels[-1].AggOp = AggOp
    levels[-1].AggOpT = AggOp.T.tocsr()
    levels[-1].N = AggOp.shape[1]  # number of coarse grid points
    levels[-1].nonoverlapping_subdomain = [None]*levels[-1].N
    levels[-1].overlapping_subdomain = [None]*levels[-1].N
    levels[-1].PoU = [None]*levels[-1].N
    levels[-1].overlapping_rows = [None]*levels[-1].N
    levels[-1].nIi = np.zeros(levels[-1].N, dtype=np.int32)
    levels[-1].ni = np.zeros(levels[-1].N, dtype=np.int32)
    levels[-1].nev = np.zeros(levels[-1].N, dtype=np.int32)
    v_mult = np.zeros(AggOp.shape[0])
    blocksize = np.zeros(levels[-1].N, dtype=np.int32)
    v_row_mult = np.zeros(B.shape[0])

    t0 = time.perf_counter()

    nodes_vs_subdomains_r = []
    nodes_vs_subdomains_c = []
    nodes_vs_subdomains_v = []
    for i in range(levels[-1].N):
        levels[-1].nonoverlapping_subdomain[i] = np.asarray(
            levels[-1].AggOpT.indices[levels[-1].AggOpT.indptr[i]:levels[-1].AggOpT.indptr[i + 1]], dtype=np.int32)
        levels[-1].nIi[i] = len(levels[-1].nonoverlapping_subdomain[i])
        levels[-1].overlapping_subdomain[i] = []
        for j in levels[-1].nonoverlapping_subdomain[i]:
            # Get the subdomain
            levels[-1].overlapping_subdomain[i].append(
                A.indices[A.indptr[j]:A.indptr[j + 1]])
        # print(levels[-1].overlapping_subdomain[i])
        levels[-1].overlapping_subdomain[i] = np.concatenate(
            levels[-1].overlapping_subdomain[i], dtype=np.int32)
        levels[-1].overlapping_subdomain[i] = np.unique(
            levels[-1].overlapping_subdomain[i])
        levels[-1].ni[i] = len(levels[-1].overlapping_subdomain[i])
        
        # Get the overlapping rows
        levels[-1].overlapping_rows[i] = []
        for j in levels[-1].nonoverlapping_subdomain[i]:
            # Get the subdomain
            levels[-1].overlapping_rows[i].append(BT.indices[BT.indptr[j]:BT.indptr[j + 1]])
        levels[-1].overlapping_rows[i] = np.concatenate(
            levels[-1].overlapping_rows[i], dtype=np.int32)
        levels[-1].overlapping_rows[i] = np.unique(levels[-1].overlapping_rows[i])
        v_row_mult[levels[-1].overlapping_rows[i]] += 1
        
        blocksize[i] = len(levels[-1].overlapping_subdomain[i])
        # Loop over the subdomain and get the PoU
        v_mult[levels[-1].overlapping_subdomain[i]] += 1
        nodes_vs_subdomains_r.append(levels[-1].overlapping_subdomain[i])
        nodes_vs_subdomains_c.append(i*np.ones(len(levels[-1].overlapping_subdomain[i]), dtype=np.int32))
        nodes_vs_subdomains_v.append(np.ones(len(levels[-1].overlapping_subdomain[i])))
    nodes_vs_subdomains_r = np.concatenate(nodes_vs_subdomains_r, dtype=np.int32)
    nodes_vs_subdomains_c = np.concatenate(nodes_vs_subdomains_c, dtype=np.int32)
    nodes_vs_subdomains_v = np.concatenate(nodes_vs_subdomains_v, dtype=np.float64)
    levels[-1].nodes_vs_subdomains = csr_array((nodes_vs_subdomains_v, (nodes_vs_subdomains_r, nodes_vs_subdomains_c)), shape=(A.shape[0], levels[-1].N))
    levels[-1].nodes_vs_subdomains = levels[-1].nodes_vs_subdomains.tocsr()
    levels[-1].T = levels[-1].nodes_vs_subdomains.T @ levels[-1].nodes_vs_subdomains
    
    levels[-1].T.data = np.ones(levels[-1].T.data.shape, dtype=levels[-1].T.data.dtype)
    k_c = levels[-1].T @ np.ones(levels[-1].T.shape[0], dtype=levels[-1].T.data.dtype)
    levels[-1].number_of_colors = max(k_c)    
    levels[-1].multiplicity = max(v_row_mult)
    
    for i in range(levels[-1].N):
        levels[-1].PoU[i] = []
        for j in levels[-1].overlapping_subdomain[i]:
            # levels[-1].PoU[i].append(1/v_mult[j])
            # Get the subdomain
            if j in levels[-1].nonoverlapping_subdomain[i]:
                # levels[-1].PoU[i].append(1/v_mult[j])
                levels[-1].PoU[i].append(1)
            else:
                levels[-1].PoU[i].append(0.0)
        levels[-1].PoU[i] = np.array(levels[-1].PoU[i])

    t1 = time.perf_counter()
    print("Aggregation time = ", t1 - t0)

    t0 = time.perf_counter()

    # Sanity check PoU correct
    temp = np.random.rand(A.shape[0])
    temp2 = 0*temp
    for i in range(levels[-1].N):
        temp2[levels[-1].overlapping_subdomain[i]] += levels[-1].PoU[i] * temp[levels[-1].overlapping_subdomain[i]]
    if(np.linalg.norm(temp2 - temp) > 1e-14*np.linalg.norm(temp)):
        warn('Partition of unity is incorrect. This can happen if the partitioning strategy \
            did not yield a nonoverlapping cover of the set of nodes')
        
    levels[-1].subdomain = np.zeros(np.sum(blocksize), dtype=np.int32)
    levels[-1].subdomain_ptr = np.zeros(levels[-1].N + 1, dtype=np.int32)
    levels[-1].subdomain_ptr[0] = 0
    for i in range(levels[-1].N):
        levels[-1].subdomain_ptr[i+1] = levels[-1].subdomain_ptr[i] + blocksize[i]
        levels[-1].subdomain[levels[-1].subdomain_ptr[i]:levels[-1].subdomain_ptr[i + 1]] = levels[-1].overlapping_subdomain[i]

    levels[-1].submatrices_ptr = np.zeros(levels[-1].N + 1, dtype=np.int32)
    levels[-1].submatrices_ptr[0] = 0
    for i in range(levels[-1].N):
        levels[-1].submatrices_ptr[i+1] = levels[-1].submatrices_ptr[i] + \
            blocksize[i]*blocksize[i]
    levels[-1].submatrices = np.zeros(levels[-1].submatrices_ptr[-1],dtype=A.data.dtype)
    amg_core.extract_subblocks(A.indptr, A.indices, A.data, levels[-1].submatrices, \
                                levels[-1].submatrices_ptr, levels[-1].subdomain, \
                                    levels[-1].subdomain_ptr, \
                                        int(levels[-1].subdomain_ptr.shape[0]-1), A.shape[0])

    levels[-1].auxiliary = np.zeros(levels[-1].submatrices_ptr[-1])

    t1 = time.perf_counter()
    print("POU check time = ", t1 - t0)

    t0 = time.perf_counter()

    # loop over subdomains
    BTT = BT.T.conjugate().tocsr()
    ### Hussam's original python implementation
    ### This ended up being a serious bottleneck
    for i in range(levels[-1].N):
        bi = _extract_row_block(B,levels[-1].overlapping_rows[i])
        bit = _extract_row_block(BTT,levels[-1].overlapping_rows[i])
        bit = bit.T.conjugate()
        a = bit @ bi
        a = a.tocsr()
        a.sort_indices()
        b = levels[-1].auxiliary[levels[-1].submatrices_ptr[i]:levels[-1].submatrices_ptr[i+1]]
        
        levels[-1].auxiliary[levels[-1].submatrices_ptr[i]:levels[-1].submatrices_ptr[i+1]] = \
            extract_diagonal_block(a, levels[-1].overlapping_subdomain[i]).flatten()

    # rows_indptr = np.zeros(levels[-1].N+1, dtype=np.int32)
    # cols_indptr = np.zeros(levels[-1].N+1, dtype=np.int32)
    # for i in range(levels[-1].N):
    #     rows_indptr[i+1] = rows_indptr[i] + len(levels[-1].overlapping_rows[i])
    #     cols_indptr[i+1] = cols_indptr[i] + len(levels[-1].overlapping_subdomain[i])

    # rows_flat = np.concatenate(levels[-1].overlapping_rows).astype(np.int32, copy=False)
    # cols_flat = np.concatenate(levels[-1].overlapping_subdomain).astype(np.int32, copy=False)
    # amg_core.local_outer_product(
    #     B.shape[0], B.shape[1],
    #     B.indptr, B.indices, B.data,
    #     BTT.indptr, BTT.indices, BTT.data,
    #     rows_flat, rows_indptr,
    #     cols_flat, cols_indptr,
    #     levels[-1].auxiliary, levels[-1].submatrices_ptr)

    if threshold is None:
        levels[-1].threshold = max(0.1,((kappa/levels[-1].number_of_colors) - 1)/ levels[-1].multiplicity)
        # levels[-1].threshold = max(0.1,((kappa/3) - 1)/levels[-1].multiplicity)
    else:
        levels[-1].threshold = threshold
    p_r = []
    p_c = []
    p_v = []
    counter = 0
    levels[-1].min_ev = 1e12
    
    t1 = time.perf_counter()
    print("loop 1 time = ", t1 - t0)

    t0 = time.perf_counter()
    for i in range(levels[-1].N):
        b = levels[-1].auxiliary[levels[-1].submatrices_ptr[i]:levels[-1].submatrices_ptr[i+1]]
        bb = np.reshape(b,(int(np.sqrt(len(b))),int(np.sqrt(len(b)))))

        a = levels[-1].submatrices[levels[-1].submatrices_ptr[i]:levels[-1].submatrices_ptr[i+1]]
        aa = np.reshape(a,(int(np.sqrt(len(a))),int(np.sqrt(len(a)))))

        d = np.diag(levels[-1].PoU[i],0)
        dad = d @ aa @ d
        
        normbb = np.linalg.norm(bb, ord=2)
        bb = bb + np.eye(bb.shape[0]) * (1e-10*normbb)
        
        E, V = eigh(dad,bb)
        # E, V = eig(bb,dad)
        # E = 1/E
        if nev is not None:
            E = E[-nev:]
            levels[-1].min_ev = min(levels[-1].min_ev, E[-nev])
            V = V[:,-nev:]
            levels[-1].nev[i] = nev
            for j in range(len(E)):
                p_r.append(levels[-1].subdomain[levels[-1].subdomain_ptr[i]:levels[-1].subdomain_ptr[i + 1]])
                p_c.append([counter]*len(levels[-1].subdomain[levels[-1].subdomain_ptr[i]:levels[-1].subdomain_ptr[i + 1]]))
                counter += 1
                p_v.append(d@V[:,j])
        else:
            counter_nev = 0
            for j in range(len(E)):
                # if j == len(E) - 1:
                #     counter_nev += 1
                #     levels[-1].min_ev = min(levels[-1].min_ev, E[-1])
                #     p_r.append(levels[-1].subdomain[levels[-1].subdomain_ptr[i]:levels[-1].subdomain_ptr[i + 1]])
                #     p_c.append([counter]*len(levels[-1].subdomain[levels[-1].subdomain_ptr[i]:levels[-1].subdomain_ptr[i + 1]]))
                #     counter += 1
                #     p_v.append(d@V[:,j])
                # else:
                    if E[j] > levels[-1].threshold:
                        counter_nev += 1
                        levels[-1].min_ev = min(levels[-1].min_ev, E[j])
                        p_r.append(levels[-1].subdomain[levels[-1].subdomain_ptr[i]:levels[-1].subdomain_ptr[i + 1]])
                        p_c.append([counter]*len(levels[-1].subdomain[levels[-1].subdomain_ptr[i]:levels[-1].subdomain_ptr[i + 1]]))
                        counter += 1
                        p_v.append(d@V[:,j])
            levels[-1].nev[i] = counter_nev
    if(len(p_r)) == 0:
        p_r = [[0]]
        p_c = [[0]]
        p_v = [[1]]
        counter = 1
    p_r = np.concatenate(p_r, dtype=np.int32)
    p_c = np.concatenate(p_c, dtype=np.int32)
    p_v = np.concatenate(p_v, dtype=np.float64)
    levels[-1].P = csr_array((p_v, (p_r, p_c)), shape=(A.shape[0], counter))
    levels[-1].R = levels[-1].P.T.conjugate().tocsr()

    t1 = time.perf_counter()
    print("loop 2 time = ", t1 - t0)

    t0 = time.perf_counter()
    B = B @ levels[-1].P
    BT = levels[-1].R @ BT
    if filteringB > 0:
        norm_fro_B = np.linalg.norm(abs(B.data))
        print("B NNZ before filtering", len(B.data))
        # _filter_matrix(B, norm_fro_B * np.sqrt(filtering))
        _filter_matrix(B, norm_fro_B * (filteringB))
        B.eliminate_zeros()
        print("B NNZ after filtering", len(B.data))
        norm_fro_BT = np.linalg.norm(abs(BT.data))
        print("BT NNZ before filtering", len(BT.data))
        # _filter_matrix(BT,norm_fro_BT * np.sqrt(filtering))
        _filter_matrix(BT,norm_fro_BT * (filteringB))
        BT.eliminate_zeros()
        print("BT NNZ after filtering", len(BT.data))

    A = BT @ B
    if filteringA > 0:
        A = 0.5*(A + A.T.conjugate())
        print("NNZ before filtering", len(A.data))
        # norm_fro_A = np.linalg.norm(A.data)
        norm_fro_A = max(abs(A.data))
        _filter_matrix(A,norm_fro_A * filteringA)
        A.eliminate_zeros()
        print("NNZ after filtering", len(A.data))

    t1 = time.perf_counter()
    print("P^TAP time = ", t1 - t0)

    levels.append(MultilevelSolver.Level())
    levels[-1].A = A
    levels[-1].B = B
    levels[-1].BT = BT

def _extract_row_block(A, row_idx):
    """Extract the row block of a sparse matrix A given the row index."""
    indices = []
    data = []
    indptr = [0]
    for i in range(len(row_idx)):
        idx = row_idx[i]
        indptr.append(indptr[i] + A.indptr[idx + 1] - A.indptr[idx])
        indices.extend(A.indices[A.indptr[idx]:A.indptr[idx + 1]])
        data.extend(A.data[A.indptr[idx]:A.indptr[idx + 1]])
    return csr_array((data, indices, indptr), shape=(len(row_idx),A.shape[1]))

def extract_diagonal_block(A, indices):
    """Extract the diagonal block of a sparse matrix A given the row index."""
    b = np.zeros((len(indices),len(indices)), dtype=A.data.dtype)
    for i in range(len(indices)):
        idx = indices[i]
        for j in A.indices[A.indptr[idx]:A.indptr[idx + 1]]:
            if j in indices:
                jj = np.where(indices == j)[0]
                b[i,jj] = A.data[A.indptr[idx]:A.indptr[idx + 1]][A.indices[A.indptr[idx]:A.indptr[idx + 1]] == j][0]
    return b

# def extract_nonzero_block(A, indices):
#     """Extract the nonzero block of a sparse matrix A given the row index."""
#     b = np.zeros((len(indices),len(indices)), dtype=A.data.dtype)
#     for i in range(len(indices)):
#         idx = indices[i]
#         for j in A.indices[A.indptr[idx]:A.indptr[idx + 1]]:
#             b[i,j] = A.data[A.indptr[idx]:A.indptr[idx + 1]][A.indices[A.indptr[idx]:A.indptr[idx + 1]] == j][0]
#     return b    
def _remove_empty_columns(A):
    """Remove empty columns from a sparse matrix."""
    if( not issparse(A)):
        raise TypeError('Argument A must be a sparse matrix')
    if A.format != 'csc':
        raise TypeError('Argument A must be a csc sparse matrix')
    m,n = A.shape
    ones = np.ones(m, dtype=A.dtype)
    s = ones @ A
    ptr = [0]
    for i in range(n):
        if s[i] > 0:
            ptr.append(A.indptr[i+1])
    A_new = csc_array((A.data, A.indices, ptr),[m,len(ptr)-1])
    return A_new

def _add_columns_containing_isolated_nodes(A):
    m,n = A.shape
    x = A @ np.ones((n,1))
    loc_isolated_nodes = np.where(x==0)[0]
    n_isolated_nodes = len(loc_isolated_nodes)
    if n_isolated_nodes == 0:
        return A
    else:
        x_r = loc_isolated_nodes
        x_c = np.array(list(range(n_isolated_nodes)))
        x_v = x_r*0+1
        x = csc_array((x_v,(x_r,x_c)),shape=(m,n_isolated_nodes))
        # x = 0*x
        # x[loc_isolated_nodes] += 1
        A = hstack([A,x])
        return A
        
def _filter_matrix(A,threshold):
    if( not issparse(A)):
        raise TypeError('Argument A must be a sparse matrix')
    if A.format != 'csc' and A.format != 'csr':
        raise TypeError('Argument A must be a csc sparse matrix')

    m = len(A.indptr) - 1
    for i in range(m):
        for k in range(A.indptr[i],A.indptr[i+1]):
            if A.indices[k] != i:
                if abs(A.data[k]) < threshold:
                    A.data[k] = 0.0
                    
