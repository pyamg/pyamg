"""Algorithms related to graphs."""

from warnings import warn
import numpy as np
from scipy import sparse
# from icecream import ic

from . import amg_core


def asgraph(G):
    """Return (square) matrix as sparse."""
    if not (sparse.isspmatrix_csr(G) or sparse.isspmatrix_csc(G)):
        G = sparse.csr_matrix(G)

    if G.shape[0] != G.shape[1]:
        raise ValueError('expected square matrix')

    return G


def maximal_independent_set(G, algo='serial', k=None):
    """Compute a maximal independent vertex set for a graph.

    Parameters
    ----------
    G : sparse matrix
        Symmetric matrix, preferably in sparse CSR or CSC format
        The nonzeros of G represent the edges of an undirected graph.
    algo : {'serial', 'parallel'}
        Algorithm used to compute the MIS
            * serial   : greedy serial algorithm
            * parallel : variant of Luby's parallel MIS algorithm

    Returns
    -------
    S : array
        S[i] = 1 if vertex i is in the MIS
        S[i] = 0 otherwise

    Notes
    -----
    Diagonal entries in the G (self loops) will be ignored.

    Luby's algorithm is significantly more expensive than the
    greedy serial algorithm.

    """
    G = asgraph(G)
    N = G.shape[0]

    mis = np.empty(N, dtype='intc')
    mis[:] = -1

    if k is None:
        if algo == 'serial':
            fn = amg_core.maximal_independent_set_serial
            fn(N, G.indptr, G.indices, -1, 1, 0, mis)
        elif algo == 'parallel':
            fn = amg_core.maximal_independent_set_parallel
            fn(N, G.indptr, G.indices, -1, 1, 0, mis, np.random.rand(N), -1)
        else:
            raise ValueError('Unknown algorithm ({algo})')
    else:
        fn = amg_core.maximal_independent_set_k_parallel
        fn(N, G.indptr, G.indices, k, mis, np.random.rand(N), -1)

    return mis


def vertex_coloring(G, method='MIS'):
    """Compute a vertex coloring of a graph.

    Parameters
    ----------
    G : sparse matrix
        Symmetric matrix, preferably in sparse CSR or CSC format
        The nonzeros of G represent the edges of an undirected graph.
    method : string
        Algorithm used to compute the vertex coloring:

            * 'MIS' - Maximal Independent Set
            * 'JP'  - Jones-Plassmann (parallel)
            * 'LDF' - Largest-Degree-First (parallel)

    Returns
    -------
    coloring : array
        An array of vertex colors (integers beginning at 0)

    Notes
    -----
    Diagonal entries in the G (self loops) will be ignored.

    """
    G = asgraph(G)
    N = G.shape[0]

    coloring = np.empty(N, dtype='intc')

    if method == 'MIS':
        fn = amg_core.vertex_coloring_mis
        fn(N, G.indptr, G.indices, coloring)
    elif method == 'JP':
        fn = amg_core.vertex_coloring_jones_plassmann
        fn(N, G.indptr, G.indices, coloring, np.random.rand(N))
    elif method == 'LDF':
        fn = amg_core.vertex_coloring_LDF
        fn(N, G.indptr, G.indices, coloring, np.random.rand(N))
    else:
        raise ValueError('Unknown method ({method})')

    return coloring


def bellman_ford(G, centers, method='standard'):
    """Bellman-Ford iteration.

    Parameters
    ----------
    G : sparse matrix
        Directed graph with positive weights.
    centers : list
        Starting centers or source nodes
    method : string
        'standard': base implementation of Bellman-Ford
        'balanced': a balanced version of Bellman-Ford

    Returns
    -------
    distances : array
        Distance of each point to the nearest center
    nearest : array
        Index of the nearest center
    predecessors : array
        Predecessors in the array

    See Also
    --------
    pyamg.amg_core.bellman_ford
    scipy.sparse.csgraph.bellman_ford
    """
    G = asgraph(G)
    n = G.shape[0]

    if G.nnz > 0:
        if G.data.min() < 0:
            raise ValueError('Bellman-Ford is defined only for positive weights.')
    if G.dtype == complex:
        raise ValueError('Bellman-Ford is defined only for real weights.')

    centers = np.asarray(centers, dtype=np.int32)

    # allocate space for returns and working arrays
    distances = np.empty(n, dtype=G.dtype)
    nearest = np.empty(n, dtype=np.int32)
    predecessors = np.empty(n, dtype=np.int32)
    if method == 'balanced':
        predecessors_count = np.empty(n, dtype=np.int32)
        cluster_size = np.empty(len(centers), dtype=np.int32)

    if method == 'standard':
        amg_core.bellman_ford(n, G.indptr, G.indices, G.data, centers,  # IN
                              distances, nearest, predecessors,         # OUT
                              True)
    elif method == 'balanced':
        amg_core.bellman_ford_balanced(n, G.indptr, G.indices, G.data, centers,  # IN
                                       distances, nearest, predecessors,         # OUT
                                       predecessors_count, cluster_size,         # OUT
                                       True)
    else:
        raise ValueError(f'Method {method} is not supported in Bellman-Ford')

    return distances, nearest, predecessors


def lloyd_cluster(G, centers, maxiter=5):
    """Perform Lloyd clustering on graph with weighted edges.

    Parameters
    ----------
    G : csr_matrix, csc_matrix
        A sparse nxn matrix where each nonzero entry G[i,j] is the distance
        between nodes i and j.
    centers : int array
        If centers is an integer, then its value determines the number of
        clusters.  Otherwise, centers is an array of unique integers between 0
        and n-1 that will be used as the initial centers for clustering.

    Returns
    -------
    clusters : int array
        id of each cluster of points
    centers : int array
        index of each center

    Notes
    -----
    If G has complex values, abs(G) is used instead.

    Only positive edge weights may be used
    """
    G = asgraph(G)
    n = G.shape[0]

    # complex dtype
    if G.dtype.kind == 'c':
        G = np.abs(G)

    if G.nnz > 0:
        if G.data.min() < 0:
            raise ValueError('Lloyd Clustering is defined only for positive weights.')

    if np.isscalar(centers):
        centers = np.random.permutation(n)[:centers]
        centers = np.int32(centers)
    else:
        centers = np.asarray(centers, dtype=np.int32)

    num_clusters = len(centers)

    if num_clusters < 1:
        raise ValueError('at least one center is required')

    if centers.min() < 0:
        raise ValueError(f'invalid center index {centers.min()}')

    if centers.max() >= n:
        raise ValueError(f'invalid center index {centers.max()}')

    distances = np.empty(n, dtype=G.dtype)
    clusters = np.empty(n, dtype=np.int32)
    predecessors = np.full(n, -1, dtype=np.int32)

    amg_core.lloyd_cluster(n, G.indptr, G.indices, G.data,     # IN
                           centers,                            # INOUT
                           distances, clusters, predecessors,  # OUT
                           True, maxiter)

    return clusters, centers


def balanced_lloyd_cluster(G, centers, maxiter=5, rebalance_iters=5):
    """Perform Lloyd clustering on graph with weighted edges.

    Parameters
    ----------
    G : csr_matrix, csc_matrix
        A sparse nxn matrix where each nonzero entry G[i,j] is the distance
        between nodes i and j.
    centers : int array
        If centers is an integer, then its value determines the number of
        clusters.  Otherwise, centers is an array of unique integers between 0
        and n-1 that will be used as the initial centers for clustering.
    maxiter : int
        Number of bellman_ford_balanced->center_nodes iterations to run within
        the clustering.
    rebalance_iters : int
        Number of post-Lloyd rebalancing iterations to run.

    Returns
    -------
    clusters : int array
        id of each cluster of points
    centers : int array
        index of each center

    Notes
    -----
    If G has complex values, abs(G) is used instead.

    Only positive edge weights may be used
    """
    G = asgraph(G)
    n = G.shape[0]

    # complex dtype
    if G.dtype.kind == 'c':
        G = np.abs(G)

    if G.nnz > 0:
        if G.data.min() < 0:
            raise ValueError('Lloyd Clustering is defined only for positive weights.')

    if np.isscalar(centers):
        centers = np.random.permutation(n)[:centers]  # same as np.random.choice()
        centers = np.int32(centers)
    else:
        centers = np.asarray(centers, dtype=np.int32)

    num_clusters = len(centers)

    if num_clusters < 1:
        raise ValueError('at least one center is required')

    if centers.min() < 0:
        raise ValueError(f'invalid center index {centers.min()}')

    if centers.max() >= n:
        raise ValueError(f'invalid center index {centers.max()}')

    # create work arrays for C++
    # empty() values are initialized in the kernel
    maxsize = int(8*np.ceil((n / num_clusters)))

    Cptr = np.empty(num_clusters, dtype=np.int32)
    D = np.empty((maxsize, maxsize), dtype=G.dtype)
    P = np.empty((maxsize, maxsize), dtype=np.int32)
    CC = np.empty(n, dtype=np.int32)
    L = np.empty(n, dtype=np.int32)

    q = np.empty(maxsize, dtype=G.dtype)
    d = np.empty(n, dtype=G.dtype)
    m = np.empty(n, dtype=np.int32)
    p = np.empty(n, dtype=np.int32)
    pc = np.empty(n, dtype=np.int32)
    s = np.empty(num_clusters, dtype=np.int32)

    amg_core.lloyd_cluster_balanced(n,
                                    G.indptr, G.indices, G.data,
                                    Cptr, D.ravel(), P.ravel(), CC, L,
                                    q, centers, d, m, p,
                                    pc, s,
                                    True, maxiter)

    for _riter in range(rebalance_iters):

        # don't rebalance a single cluster
        if num_clusters < 2:
            break

        centers = _rebalance(G, centers, m, d, num_clusters, Cptr, CC, L)

        amg_core.lloyd_cluster_balanced(n,
                                        G.indptr, G.indices, G.data,
                                        Cptr, D.ravel(), P.ravel(), CC, L,
                                        q, centers, d, m, p,
                                        pc, s,
                                        True, maxiter)

    return m, centers


def _rebalance(G, c, m, d, num_clusters, Cptr, C, L):
    """
    Parameters
    ----------
    G : sparse matrix
        Sparse graph
    c : array
        List of centers
    d : array
        Distance to cluster center
    m : array
        Cluster membership
    num_clusters : int
        Number of clusters (= number centers)
    Cptr : array
        Work array that points to the start of the nodes in C in a cluster
    C : array
        Work array of clusters, sorted by cluster
    L : array
        Work array of local ids for each cluster

    Return
    ------
    centers : array
        List of new centers
    """
    newc = c.copy()

    AggOp = sparse.coo_matrix((np.ones(len(m)), (np.arange(len(m)), m))).tocsr()
    Agg2Agg = AggOp.T @ G @ AggOp
    Agg2Agg = Agg2Agg.tocsr()

    E = _elimination_penalty(G, m, d, num_clusters, Cptr, C, L)
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

    # show eliminated and split aggregates
    # run one bellman_ford to get new clusters
    gamma = 1.0
    stopsplitting = False
    while E[a_e] < gamma * S[a_s] or stopsplitting:
        newc[a_e] = I[a_s]   # redefine centers
        newc[a_s] = J[a_s]   # redefine centers
        M[Agg2Agg.getrow(a_e).indices] = False  # cannot eliminate neighbors agg
        M[Agg2Agg.getrow(a_s).indices] = False  # cannot split neighbors agg
        if len(np.where(np.logical_not(M))[0]) == num_clusters:
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
            if not M[Slist[-1-i_s]] or pushtie:     # if invalid aggregate or need a new one
                if pushtie:
                    pushtie = False
                while i_s < num_clusters-1:         # increment elimination counter
                    i_s += 1
                    if M[Slist[-1-i_s]]:            # if a valid aggregate is encountered
                        break
            if i_s == num_clusters-1 or i_e == num_clusters-1:
                stopsplitting = True                # if we've looped through, stop
                break
            a_e = Elist[i_e]
            a_s = Slist[-1-i_s]
            if a_e != a_s:                          # if new pair found, then done
                findanother = False
            else:
                pushtie = True                      # otherwise push the tie breaker
    return newc


def _elimination_penalty(A, m, d, num_clusters, Cptr, C, L):
    """
    see _rebalance()
    """
    # pylint: disable=too-many-nested-blocks
    E = np.inf * np.ones(num_clusters)
    for a in range(num_clusters):
        E[a] = 0
        Va = np.int32(np.where(m == a)[0])

        N = len(Va)
        D = np.zeros((N, N))
        P = np.zeros((N, N), dtype=np.int32)
        _N = Cptr[a]+N
        if _N >= A.shape[0]:
            _N = None
        amg_core.floyd_warshall(A.shape[0], A.indptr, A.indices, A.data,
                                D.ravel(), P.ravel(), C[Cptr[a]:_N], L,
                                m, a, N)
        for _i, i in enumerate(Va):  # pylint: disable=unused-variable
            dmin = np.inf
            for _j, j in enumerate(Va):
                for k in A.getrow(j).indices:
                    if m[k] != m[j]:
                        if (d[k] + D[_i, _j] + A[j, k]) < dmin:
                            dmin = d[k] + D[_i, _j] + A[j, k]
            E[a] += dmin**2
        E[a] -= np.sum(d[Va]**2)
    return E


def _split_improvement(A, m, d, num_clusters, Cptr, C, L):
    """
    see _rebalance()
    """
    S = np.inf * np.ones(num_clusters)
    I = -1 * np.ones(num_clusters)  # better cluster centers if split
    J = -1 * np.ones(num_clusters)  # better cluster centers if split
    for a in range(num_clusters):
        S[a] = np.inf
        Va = np.int32(np.where(m == a)[0])

        N = len(Va)
        D = np.zeros((N, N))
        P = np.zeros((N, N), dtype=np.int32)
        _N = Cptr[a]+N
        if _N >= A.shape[0]:
            _N = None
        amg_core.floyd_warshall(A.shape[0], A.indptr, A.indices, A.data,
                                D.ravel(), P.ravel(), C[Cptr[a]:_N], L,
                                m, a, N)
        for _i, i in enumerate(Va):
            for _j, j in enumerate(Va):
                Snew = 0
                for _k, k in enumerate(Va):  # pylint: disable=unused-variable
                    if D[_k, _i] < D[_k, _j]:
                        Snew = Snew + D[_k, _i]**2
                    else:
                        Snew = Snew + D[_k, _j]**2
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
    warn("kmeanspp_seed is O(n^3) -- use only for testing")

    n = G.shape[0]
    C = np.random.choice(n, 1, replace=False)
    for _ in range(nseeds-1):
        d = sparse.csgraph.bellman_ford(G, directed=False, indices=C)
        d = d.min(axis=0)   # shortest path from a seed
        d = d**2            # distance squared
        p = d / np.sum(d)   # probability
        # newC = np.random.choice(n, 1, p=p, replace=False) # <- does not work properly
        newC = _choice(p)
        C = np.append(C, newC)
    return C


def breadth_first_search(G, seed):
    """Breadth First search of a graph.

    Parameters
    ----------
    G : csr_matrix, csc_matrix
        A sparse NxN matrix where each nonzero entry G[i,j] is the distance
        between nodes i and j.
    seed : int
        Index of the seed location

    Returns
    -------
    order : int array
        Breadth first order
    level : int array
        Final levels

    Examples
    --------
    0---2
    |  /
    | /
    1---4---7---8---9
    |  /|  /
    | / | /
    3/  6/
    |
    |
    5
    >>> import numpy as np
    >>> import pyamg
    >>> import scipy.sparse as sparse
    >>> edges = np.array([[0,1],[0,2],[1,2],[1,3],[1,4],[3,4],[3,5],
                          [4,6], [4,7], [6,7], [7,8], [8,9]])
    >>> N = np.max(edges.ravel())+1
    >>> data = np.ones((edges.shape[0],))
    >>> A = sparse.coo_matrix((data, (edges[:,0], edges[:,1])), shape=(N,N))
    >>> c, l = pyamg.graph.breadth_first_search(A, 0)
    >>> print(l)
    >>> print(c)
    [0 1 1 2 2 3 3 3 4 5]
    [0 1 2 3 4 5 6 7 8 9]

    """
    G = asgraph(G)
    N = G.shape[0]

    order = np.empty(N, G.indptr.dtype)
    level = np.empty(N, G.indptr.dtype)
    level[:] = -1

    BFS = amg_core.breadth_first_search
    BFS(G.indptr, G.indices, int(seed), order, level)

    return order, level


def connected_components(G):
    """Compute the connected components of a graph.

    The connected components of a graph G, which is represented by a
    symmetric sparse matrix, are labeled with the integers 0,1,..(K-1) where
    K is the number of components.

    Parameters
    ----------
    G : symmetric matrix, preferably in sparse CSR or CSC format
        The nonzeros of G represent the edges of an undirected graph.

    Returns
    -------
    components : ndarray
        An array of component labels for each vertex of the graph.

    Notes
    -----
    If the nonzero structure of G is not symmetric, then the
    result is undefined.

    Examples
    --------
    >>> from pyamg.graph import connected_components
    >>> print connected_components( [[0,1,0],[1,0,1],[0,1,0]] )
    [0 0 0]
    >>> print connected_components( [[0,1,0],[1,0,0],[0,0,0]] )
    [0 0 1]
    >>> print connected_components( [[0,0,0],[0,0,0],[0,0,0]] )
    [0 1 2]
    >>> print connected_components( [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]] )
    [0 0 1 1]

    """
    G = asgraph(G)
    N = G.shape[0]

    components = np.empty(N, G.indptr.dtype)

    fn = amg_core.connected_components
    fn(N, G.indptr, G.indices, components)

    return components


def symmetric_rcm(A):
    """Symmetric Reverse Cutthill-McKee.

    Parameters
    ----------
    A : sparse matrix
        Sparse matrix

    Returns
    -------
    B : sparse matrix
        Permuted matrix with reordering

    Notes
    -----
    Get a pseudo-peripheral node, then call BFS

    Examples
    --------
    >>> from pyamg import gallery
    >>> from pyamg.graph import symmetric_rcm
    >>> n = 200
    >>> density = 1.0/n
    >>> A = gallery.sprand(n, n, density, format='csr')
    >>> S = A + A.T
    >>> # try the visualizations
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(121)
    >>> plt.spy(S,marker='.')
    >>> plt.subplot(122)
    >>> plt.spy(symmetric_rcm(S),marker='.')

    See Also
    --------
    pseudo_peripheral_node

    """

    dummy_root, order, dummy_level = pseudo_peripheral_node(A)

    p = order[::-1]

    return A[p, :][:, p]


def pseudo_peripheral_node(A):
    """Find a pseudo peripheral node.

    Parameters
    ----------
    A : sparse matrix
        Sparse matrix

    Returns
    -------
    x : int
        Locaiton of the node
    order : array
        BFS ordering
    level : array
        BFS levels

    Notes
    -----
    Algorithm in Saad

    """
    n = A.shape[0]

    valence = np.diff(A.indptr)

    # select an initial node x, set delta = 0
    x = int(np.random.rand() * n)
    delta = 0

    while True:
        # do a level-set traversal from x
        order, level = breadth_first_search(A, x)

        # select a node y in the last level with min degree
        maxlevel = level.max()
        lastnodes = np.where(level == maxlevel)[0]
        lastnodesvalence = valence[lastnodes]
        minlastnodesvalence = lastnodesvalence.min()
        y = np.where(lastnodesvalence == minlastnodesvalence)[0][0]
        y = lastnodes[y]

        # if d(x,y)>delta, set, and go to bfs above
        if level[y] > delta:
            x = y
            delta = level[y]
        else:
            return x, order, level
