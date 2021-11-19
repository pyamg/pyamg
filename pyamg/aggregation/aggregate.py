"""Aggregation methods."""


import numpy as np
import scipy.sparse as sparse
import scipy.sparse.csgraph
from pyamg import amg_core
from pyamg.graph import lloyd_cluster, asgraph
import warnings

__all__ = ['standard_aggregation', 'naive_aggregation', 'lloyd_aggregation', 'balanced_lloyd_aggregation']


def standard_aggregation(C):
    """Compute the sparsity pattern of the tentative prolongator.

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator
    Cpts : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import standard_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.toarray()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> standard_aggregation(A)[0].toarray() # two aggregates
    matrix([[1, 0],
            [1, 0],
            [0, 1],
            [0, 1]], dtype=int8)
    >>> A = csr_matrix([[1,0,0],[0,1,1],[0,1,1]])
    >>> A.toarray()                      # first vertex is isolated
    matrix([[1, 0, 0],
            [0, 1, 1],
            [0, 1, 1]])
    >>> standard_aggregation(A)[0].toarray() # one aggregate
    matrix([[0],
            [1],
            [1]], dtype=int8)

    See Also
    --------
    amg_core.standard_aggregation

    """
    if not sparse.isspmatrix_csr(C):
        raise TypeError('expected csr_matrix')

    if C.shape[0] != C.shape[1]:
        raise ValueError('expected square matrix')

    index_type = C.indptr.dtype
    num_rows = C.shape[0]

    Tj = np.empty(num_rows, dtype=index_type)  # stores the aggregate #s
    Cpts = np.empty(num_rows, dtype=index_type)  # stores the Cpts

    fn = amg_core.standard_aggregation

    num_aggregates = fn(num_rows, C.indptr, C.indices, Tj, Cpts)
    Cpts = Cpts[:num_aggregates]

    if num_aggregates == 0:
        # return all zero matrix and no Cpts
        return sparse.csr_matrix((num_rows, 1), dtype='int8'),\
            np.array([], dtype=index_type)
    else:

        shape = (num_rows, num_aggregates)
        if Tj.min() == -1:
            # some nodes not aggregated
            mask = Tj != -1
            row = np.arange(num_rows, dtype=index_type)[mask]
            col = Tj[mask]
            data = np.ones(len(col), dtype='int8')
            return sparse.coo_matrix((data, (row, col)), shape=shape).tocsr(), Cpts
        else:
            # all nodes aggregated
            Tp = np.arange(num_rows+1, dtype=index_type)
            Tx = np.ones(len(Tj), dtype='int8')
            return sparse.csr_matrix((Tx, Tj, Tp), shape=shape), Cpts


def naive_aggregation(C):
    """Compute the sparsity pattern of the tentative prolongator.

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator
    Cpts : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import naive_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.toarray()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> naive_aggregation(A)[0].toarray() # two aggregates
    matrix([[1, 0],
            [1, 0],
            [0, 1],
            [0, 1]], dtype=int8)
    >>> A = csr_matrix([[1,0,0],[0,1,1],[0,1,1]])
    >>> A.toarray()                      # first vertex is isolated
    matrix([[1, 0, 0],
            [0, 1, 1],
            [0, 1, 1]])
    >>> naive_aggregation(A)[0].toarray() # two aggregates
    matrix([[1, 0],
            [0, 1],
            [0, 1]], dtype=int8)

    See Also
    --------
    amg_core.naive_aggregation

    Notes
    -----
    Differs from standard aggregation.  Each dof is considered.  If it has been
    aggregated, skip over.  Otherwise, put dof and any unaggregated neighbors
    in an aggregate.  Results in possibly much higher complexities than
    standard aggregation.

    """
    if not sparse.isspmatrix_csr(C):
        raise TypeError('expected csr_matrix')

    if C.shape[0] != C.shape[1]:
        raise ValueError('expected square matrix')

    index_type = C.indptr.dtype
    num_rows = C.shape[0]

    Tj = np.empty(num_rows, dtype=index_type)  # stores the aggregate #s
    Cpts = np.empty(num_rows, dtype=index_type)  # stores the Cpts

    fn = amg_core.naive_aggregation

    num_aggregates = fn(num_rows, C.indptr, C.indices, Tj, Cpts)
    Cpts = Cpts[:num_aggregates]
    Tj = Tj - 1

    if num_aggregates == 0:
        # all zero matrix
        return sparse.csr_matrix((num_rows, 1), dtype='int8'), Cpts
    else:
        shape = (num_rows, num_aggregates)
        # all nodes aggregated
        Tp = np.arange(num_rows+1, dtype=index_type)
        Tx = np.ones(len(Tj), dtype='int8')
        return sparse.csr_matrix((Tx, Tj, Tp), shape=shape), Cpts


def lloyd_aggregation(C, naggs=None, measure=None, maxiter=5):
    """Aggregate nodes using Lloyd Clustering.

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix
    naggs : int
        number of aggregates or clusters
    measure : ['unit','abs','inv',None]
        Distance measure to use and assigned to each edge graph.

        For each nonzero value C[i,j]:
        =======  ===========================
        None     G[i,j] = C[i,j]
        'abs'    G[i,j] = abs(C[i,j])
        'inv'    G[i,j] = 1.0/abs(C[i,j])
        'unit'   G[i,j] = 1
        'sub'    G[i,j] = C[i,j] - min(C)
        =======  ===========================
    maxiter : int
        Maximum number of iterations to perform

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator.  Node i is in cluster j if AggOp[i,j] = 1.
    seeds : array
        array of seeds or Cpts, i.e., Cpts[i] = root node of aggregate i

    See Also
    --------
    amg_core.standard_aggregation

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import lloyd_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.toarray()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> lloyd_aggregation(A)[0].toarray() # one aggregate
    matrix([[1],
            [1],
            [1],
            [1]], dtype=int8)
    >>> # more seeding for two aggregates
    >>> Agg = lloyd_aggregation(A,ratio=0.5)[0].toarray()

    """
    C = asgraph(C)
    n = C.shape[0]

    if naggs is None:
        naggs = int(n / 10)

    if naggs <= 0 or naggs > n:
        raise ValueError('number of aggregates must be >=1 and <=n)')

    if measure is None:
        data = C.data
    elif measure == 'abs':
        data = np.abs(C.data)
    elif measure == 'inv':
        data = 1.0 / abs(C.data)
    elif measure == 'unit':
        data = np.ones_like(C.data).astype(float)
    elif measure == 'min':
        data = C.data - C.data.min()
    else:
        raise ValueError('unrecognized value measure=%s' % measure)

    if C.dtype == complex:
        data = np.real(data)

    if len(data) > 0:
        if data.min() < 0:
            raise ValueError('Lloyd aggregation requires a positive measure.')

    G = C.__class__((data, C.indices, C.indptr), shape=C.shape)

    distances, clusters, seeds = lloyd_cluster(G.T, naggs, maxiter=maxiter)

    if np.any(clusters < 0):
        warnings.warn('Lloyd aggregation encountered a point that is unaggregated.')

    if clusters.min() < 0:
        warnings.warn('Lloyd clustering did not cluster every point')

    row = (clusters >= 0).nonzero()[0]
    col = clusters[row]
    data = np.ones(len(row), dtype=np.int32)
    AggOp = sparse.coo_matrix((data, (row, col)), shape=(n, naggs)).tocsr()

    return AggOp, seeds


def balanced_lloyd_aggregation(C, num_clusters=None, c=None, rebalance_iters=5):
    """Aggregate nodes using Balanced Lloyd Clustering.

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix with positive weights
    num_clusters : int
        Number of seeds or clusters expected (default: C.shape[0] / 10)

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator
    seeds : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

    See Also
    --------
    amg_core.standard_aggregation

    Examples
    --------
    >>> import pyamg
    >>> from pyamg.aggregation.aggregate import balanced_lloyd_aggregation
    >>> data = pyamg.gallery.load_example('unit_square')
    >>> G = data['A']
    >>> xy = data['vertices'][:,:2]
    >>> G.data[:] = np.ones(len(G.data))

    >>> np.random.seed(787888)
    >>> AggOp, seeds = balanced_lloyd_aggregation(G)

    """

    if num_clusters is None:
        num_clusters = int(C.shape[0] / 10)

    if num_clusters < 1 or num_clusters > C.shape[0]:
        raise ValueError('num_clusters must be between 1 and n')

    if not (sparse.isspmatrix_csr(C) or sparse.isspmatrix_csc(C)):
        raise TypeError('expected csr_matrix or csc_matrix')

    if C.data.min() <= 0:
        raise ValueError('positive edge weights required')

    if C.dtype == complex:
        data = np.real(C.data)
    else:
        data = np.ones_like(C.data)

    G = C.__class__((data, C.indices, C.indptr), shape=C.shape)
    n = G.shape[0]

    if c is None:
        c = np.int32(np.random.choice(n, num_clusters, replace=False))
    else:
        num_clusters = len(c)

    maxsize = int(4*np.ceil((n / len(c))))

    Cptr = np.empty(len(c), dtype=np.int32)
    D = np.zeros((maxsize, maxsize), dtype=G.dtype)
    P = np.zeros((maxsize, maxsize), dtype=np.int32)
    CC  = np.arange(0, n, dtype=np.int32)
    L  = np.arange(0, n, dtype=np.int32)

    q = np.zeros(maxsize, dtype=G.dtype)
    d = np.empty(n, dtype=G.dtype)
    m = np.empty(n, dtype=np.int32)
    p = np.empty(n, dtype=np.int32)
    pc = np.empty(n, dtype=np.int32)
    s = np.empty(len(c), dtype=np.int32)

    energy = []
    AggOps = []
    EPs = []
    SIs = []
    centers = []
    for r in range(rebalance_iters):
        amg_core.lloyd_cluster_balanced(n,
                                        G.indptr, G.indices, G.data,
                                        Cptr, D.ravel(), P.ravel(), CC, L,
                                        q.ravel(),
                                        c, d, m, p,
                                        pc, s,
                                        True)
        AggOp = sparse.coo_matrix((np.ones(len(m)), (np.arange(len(m)),m))).tocsr()
        oldc = c.copy()

        AggOps.append(AggOp)
        centers.append(c)

        ee = np.sum(d**2)
        print('Total energy: ', ee)
        energy.append(ee)

        EP, SI, newc = _rebalance(G, c, m, d, num_clusters, Cptr, CC, L)
        c =  newc.copy()
        EPs.append(EP)
        SIs.append(SI)

    return AggOp, oldc, (EP, SI, energy, AggOps, EPs, SIs, centers)

def floyd_warshall(A, m, Cptr, C, L, a):
    """Call Floyd-Warshall on a subset of the graph
    """
    Va = np.int32(np.where(m==a)[0])
    N = len(Va)

    D = np.zeros((N, N))
    P = np.zeros((N, N), dtype=np.int32)
    _N = Cptr[a]+N
    if _N >= A.shape[0]:
        _N = None
    amg_core.graph.floyd_warshall(A.shape[0], A.indptr, A.indices, A.data,
                                  D.ravel(), P.ravel(), C[Cptr[a]:_N], L,
                                  m, a, N)
    return D, P

def _rebalance(G, c, m, d, num_clusters, Cptr, C, L):
    """
    A sparse matrix
    D[]       : (INOUT) FW distance array                               (max_size x max_size)
    d         : (INOUT) distance to cluster center                      (num_nodes x 1)
    m         : (INOUT) cluster index                                   (num_nodes x 1)
    """
    newc = c.copy()

    AggOp = sparse.coo_matrix((np.ones(len(m)), (np.arange(len(m)),m))).tocsr()
    Agg2Agg = AggOp.T @ G @ AggOp
    Agg2Agg = Agg2Agg.tocsr()

    E       = _elimination_penalty(G, m, d, num_clusters, Cptr, C, L)
    S, I, J = _split_improvement(G, m, d, num_clusters, Cptr, C, L)
    M = np.ones(num_clusters, dtype=bool)
    Elist = np.argsort(E)
    Slist = np.argsort(S)
    i_e = 0  # elimination index
    i_s = 0  # splitting index
    a_e = Elist[i_e]
    a_s = Slist[-1-i_s]
    if a_e == a_s:
        i_s += 1
        a_s = Slist[-1-i_s]
    #print(f"E:     {E}")
    #print(f"S:     {S}")
    #print(f"E srt: {E[Elist]}")
    #print(f"S srt: {S[Slist]}")
    #print(f"Elist: {Elist}")
    #print(f"Slist: {Slist}")
    #print(f"a_e: {a_e}   E[a_e]: {E[a_e]}")
    #print(f"a_s: {a_s}   S[a_s]: {S[a_s]}")
    # show eliminated and split aggregates
    # run one bellman_ford to get new clusters
    gamma = 1.0
    stopsplitting = False
    while E[a_e] < gamma * S[a_s] or stopsplitting:
        newc[a_e] = I[a_s]   # redefine centers
        newc[a_s] = J[a_s]   # redefine centers
        M[Agg2Agg.getrow(a_e).indices] = False  # cannot eliminate neighbors agg
        M[Agg2Agg.getrow(a_s).indices] = False  # cannot split neighbors agg
        if len(np.where(M==False)[0]) == num_clusters:
            break
        findanother = True                          # should we find another aggregate pair?
        stopsplitting = False                       # should we stop?
        pushtie = False                             # tie breaker
        while findanother:
            if not M[Elist[i_e]]:                   # if we have an invalid aggregate
                while i_e < num_clusters-1:         # increment elimination counter
                    i_e += 1
                    if M[Elist[i_e]]:               # if a valid aggregate is encountered
                        break
            if not M[Slist[-1-i_s]] or pushtie:     # if we have an invalid aggregate or we need a new one
                if pushtie:
                    pustie = False
                while i_s < num_clusters-1:         # increment elimination counter
                    i_s += 1
                    if M[Slist[-1-i_s]]:            # if a valid aggregate is encountered
                        break
            if i_s == num_clusters-1 or i_e == num_clusters-1:
                stopsplitting = True                # if we've looped through, stop
                break
            a_e = Elist[i_e]
            a_s = Slist[-1-i_s]
            if a_e != a_s:                          # if we found a new pair, then we're done
                findanother = False
            else:
                pushtie = True                      # otherwise push the tie breaker, and move on
    return E, S, newc

def _elimination_penalty(A, m, d, num_clusters, Cptr, C, L):
    E = np.inf * np.ones(num_clusters)
    for a in range(num_clusters):
        E[a] = 0
        Va = np.int32(np.where(m==a)[0])

        N = len(Va)
        D = np.zeros((N, N))
        P = np.zeros((N, N), dtype=np.int32)
        _N = Cptr[a]+N
        if _N >= A.shape[0]:
            _N = None
        amg_core.graph.floyd_warshall(A.shape[0], A.indptr, A.indices, A.data,
                                      D.ravel(), P.ravel(), C[Cptr[a]:_N], L,
                                      m, a, N)
        for _i, i in enumerate(Va):
            dmin = np.inf
            for _j, j in enumerate(Va):
                for k in A.getrow(j).indices:
                    if m[k] != m[j]:
                        if (d[k] + D[_i,_j] + A[j,k]) < dmin:
                            dmin = d[k] + D[_i,_j] + A[j,k]
            E[a] += dmin**2
        E[a] -= np.sum(d[Va]**2)
    return E

def _split_improvement(A, m, d, num_clusters, Cptr, C, L):
    S = np.inf * np.ones(num_clusters)
    I = -1 * np.ones(num_clusters)  # better cluster centers if split
    J = -1 * np.ones(num_clusters)  # better cluster centers if split
    for a in range(num_clusters):
        S[a] = np.inf
        Va = np.int32(np.where(m==a)[0])

        N = len(Va)
        D = np.zeros((N, N))
        P = np.zeros((N, N), dtype=np.int32)
        _N = Cptr[a]+N
        if _N >= A.shape[0]:
            _N = None
        amg_core.graph.floyd_warshall(A.shape[0], A.indptr, A.indices, A.data,
                                      D.ravel(), P.ravel(), C[Cptr[a]:_N], L,
                                      m, a, N)
        for _i, i in enumerate(Va):
            for _j, j in enumerate(Va):
                Snew = 0
                for _k, k in enumerate(Va):
                    if D[_k,_i] < D[_k,_j]:
                        Snew = Snew + D[_k,_i]**2
                    else:
                        Snew = Snew + D[_k,_j]**2
                if Snew < S[a]:
                    S[a] = Snew
                    I[a] = i
                    J[a] = j
        S[a] = np.sum(d[Va]**2) - S[a]
    return S, I, J

def _choice(p):
    """
    Parameters
    ----------
    p : array
        probabilities [0,1], with sum(p) == 1

    Return
    ------
    i : int
        index to a selected integer based on the distribution of p

    Notes
    -----
    For efficiency, there are no checks.

    TODO - needs testing
    """
    a = p / np.max(p)
    i = -1
    while True:
        i = np.random.randint(len(a))
        if np.random.rand() < a[i]:
            break
    return i

def kmeanspp_seed(G, nseeds):
    """
    Parameters
    ----------
    G : sparse matrix
        sparse graph on which to seed

    nseeds : int
        number of seeds

    Return
    ------
    C : array
        list of seeds

    Notes
    -----
    This is a reference algorithms, at O(n^3)

    TODO - needs testing
    """
    warnings.warn("kmeanspp_seed is O(n^3) -- use only for testing")

    n = G.shape[0]
    C = np.random.choice(n, 1, replace=False)
    for i in range(nseeds-1):
        d = scipy.sparse.csgraph.bellman_ford(G, directed=False, indices=C)
        d = d.min(axis=0)   # shortest path from a seed
        d = d**2            # distance squared
        p = d / np.sum(d)   # probability
        # newC = np.random.choice(n, 1, p=p, replace=False) # <- does not work properly
        newC = _choice(p)
        C = np.append(C, newC)
    return C
