"""Algorithms related to graphs."""

from warnings import warn
import numpy as np
from scipy import sparse

from . import amg_core


def asgraph(G):
    """Return (square) array as sparse.

    Parameters
    ----------
    G : sparray
        Sparse matrix.

    Returns
    -------
    csr_array or csc_array
        Converted array.

    """
    if not sparse.issparse(G) or G.format not in ('csc', 'csr'):
        G = sparse.csr_array(G)

    if G.shape[0] != G.shape[1]:
        raise ValueError('expected square matrix')

    return G


def maximal_independent_set(G, algo='serial', k=None):
    """Compute a maximal independent vertex set for a graph.

    Parameters
    ----------
    G : sparray
        Symmetric matrix, preferably in sparse CSR or CSC format.
        The nonzeros of G represent the edges of an undirected graph.
    algo : {'serial', 'parallel'}
        Algorithm used to compute the MIS
            * serial   : greedy serial algorithm
            * parallel : variant of Luby's parallel MIS algorithm
    k : int
        Minimum separation between MIS vertices.

    Returns
    -------
    array
        - ``S[i] = 1`` if vertex i is in the MIS.
        - ``S[i] = 0`` otherwise.

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
            raise ValueError(f'Unknown algorithm ({algo})')
    else:
        fn = amg_core.maximal_independent_set_k_parallel
        fn(N, G.indptr, G.indices, k, mis, np.random.rand(N), -1)

    return mis


def vertex_coloring(G, method='MIS'):
    """Compute a vertex coloring of a graph.

    Parameters
    ----------
    G : sparray
        Symmetric matrix, preferably in sparse CSR or CSC format
        The nonzeros of G represent the edges of an undirected graph.
    method : str
        Algorithm used to compute the vertex coloring:

            * 'MIS' - Maximal Independent Set
            * 'JP'  - Jones-Plassmann (parallel)
            * 'LDF' - Largest-Degree-First (parallel)

    Returns
    -------
    array
        An array of vertex colors (integers beginning at 0).

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
        raise ValueError(f'Unknown method ({method})')

    return coloring


def bellman_ford(G, centers, method='standard', tiebreaking=True):
    """Bellman-Ford iteration.

    Parameters
    ----------
    G : sparray
        Directed graph with positive weights.
    centers : list
        Starting centers or source nodes.
    method : string
        - 'standard': base implementation of Bellman-Ford.
        - 'balanced': a balanced version of Bellman-Ford.
    tiebreaking : bool
        Tie break flag if ``method='balanced'``.

    Returns
    -------
    array
        Distance of each point to the nearest center.
    array
        Index of the nearest center.
    array
        Predecessors in the array.

    See Also
    --------
    pyamg.amg_core.bellman_ford
    scipy.sparse.csgraph.bellman_ford

    Notes
    -----
    This should be viewed as the transpose of Bellman-Ford in
    scipy.sparse.csgraph. Here, bellman_ford is used to find the shortest path
    from any point *to* the seeds. In csgraph, bellman_ford is used to find
    "the shortest distance from point i to point j".  So csgraph.bellman_ford
    could be run `for seed in seeds`.  Also note that ``test_graph.py`` tests
    against ``csgraph.bellman_ford(G.T)``.

    """
    G = asgraph(G)
    n = G.shape[0]

    if G.nnz > 0:
        if G.data.min() < 0:
            raise ValueError('Bellman-Ford is defined only for positive weights.')
    if G.dtype == complex:
        raise ValueError('Bellman-Ford is defined only for real weights.')

    centers = np.asarray(centers, dtype=np.int32)
    num_clusters = len(centers)

    # allocate space for returns and working arrays
    distances = np.full(n, np.inf, dtype=G.dtype)  # distance to cluster center (inf)
    nearest = np.full(n, -1, dtype=np.int32)       # nearest center, cluster membership (-1)
    predecessors = np.full(n, -1, dtype=np.int32)  # predecessor on the shortest path (-1)
    distances[centers] = 0                         # distance = 0 at centers
    nearest[centers] = np.arange(num_clusters)     # number the membership

    if method == 'standard':
        amg_core.bellman_ford(n, G.indptr, G.indices, G.data, centers,  # IN
                              distances, nearest, predecessors)         # OUT
    elif method == 'balanced':
        predecessors_count = np.full(n, 0, dtype=np.int32)
        cluster_size = np.full(num_clusters, 1, dtype=np.int32)
        amg_core.bellman_ford_balanced(n, G.indptr, G.indices, G.data, centers,  # IN
                                       distances, nearest, predecessors,         # OUT
                                       predecessors_count, cluster_size,         # OUT
                                       tiebreaking)
    else:
        raise ValueError(f'Method {method} is not supported in Bellman-Ford')

    return distances, nearest, predecessors


def lloyd_cluster(G, centers, maxiter=5):
    """Perform Lloyd clustering on graph with weighted edges.

    Parameters
    ----------
    G : csr_array, csc_array
        A sparse matrix of size (n,n) where each nonzero entry G[i,j] is the distance
        between nodes i and j.
    centers : int array
        If centers is an integer, then its value determines the number of
        clusters.  Otherwise, centers is an array of unique integers between 0
        and n-1 that will be used as the initial centers for clustering.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    int array
        Id of each cluster of points.
    int array
        Index of each seed.

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

    distances = np.full(n, np.inf, dtype=G.dtype)
    clusters = np.full(n, -1, dtype=np.int32)
    predecessors = np.full(n, -1, dtype=np.int32)
    distances[centers] = 0
    clusters[centers] = np.arange(num_clusters)
    changed = True
    it = 0

    _dist, _near, _pred = bellman_ford(G, centers, method='standard')

    while changed and it < maxiter:
        if it > 0:
            distances.fill(np.inf)
            clusters.fill(-1)
            predecessors.fill(-1)
            distances[centers] = 0
            clusters[centers] = np.arange(num_clusters)

        amg_core.bellman_ford(n, G.indptr, G.indices, G.data, centers,  # IN
                              distances, clusters, predecessors)        # OUT

        changed = amg_core.most_interior_nodes(n, G.indptr, G.indices, G.data, centers,
                                               distances, clusters, predecessors)

        it += 1

    return clusters, centers


def balanced_lloyd_cluster(G, centers, maxiter=5, rebalance_iters=5, tiebreaking=True):
    """Perform Lloyd clustering on graph with weighted edges.

    Parameters
    ----------
    G : csr_array, csc_array
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
    tiebreaking : bool, default True
        Flag for triggering tiebreaking.

    Returns
    -------
    clusters : int array
        id of each cluster of points
    centers : int array
        index of each center

    Notes
    -----
    - If G has complex values, abs(G) is used instead.
    - Only positive edge weights may be used
    - This version computes improved cluster centers with Floyd-Warshall and
      also uses a balanced version of Bellman-Ford to try and find
      nearly-equal-sized clusters.
    - Repeated calls to bellman_ford_balanced() in the rebalance loop can result
      in different centers.  This is due to the tie-breaker based on aggregate
      size in bellman_ford_balanced().  Alternatively, the graph can be seeded
      with a small random number to make the edge lengths (and distances) unique.

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
    maxsize = int(12*np.ceil(n / num_clusters))

    d = np.full(n, np.inf, dtype=G.dtype)         # distance to cluster center (inf)
    m = np.full(n, -1, dtype=np.int32)            # cluster membership or index (-1)
    p = np.full(n, -1, dtype=np.int32)            # predecessor on the shortest path (-1)
    pc = np.full(n, 0, dtype=np.int32)            # predecessor count (0)
    s = np.full(num_clusters, 1, dtype=np.int32)  # cluster size (1)
    d[centers] = 0                                # distance = 0 at centers
    m[centers] = np.arange(num_clusters)          # number the membership

    Cptr = np.empty(num_clusters, dtype=np.int32)    # ptr to start in C for each cluster
    CC = np.empty(n, dtype=np.int32)                 # FW global index for current cluster

    D = np.empty(maxsize*maxsize, dtype=G.dtype)     # FW distance array
    P = np.empty(maxsize*maxsize, dtype=np.int32)    # FW predecessor array
    L = np.empty(n, dtype=np.int32)                  # FW local index for current cluster
    q = np.empty(maxsize, dtype=G.dtype)             # FW work array for d**2

    # global work array for distances
    dist_all = np.empty((num_clusters, maxsize*maxsize), dtype=G.dtype, order='C')

    for riter in range(rebalance_iters+1):

        # lloyd cluster balanced
        it = 0
        changed1 = True
        changed2 = True
        # reinitialize
        d.fill(np.inf)
        m.fill(-1)
        p.fill(-1)
        p[centers] = centers
        pc.fill(0)
        pc[centers] = 1
        s.fill(1)
        d[centers] = 0
        m[centers] = np.arange(num_clusters)
        while (changed1 or changed2) and (it < maxiter):
            changed1 = amg_core.bellman_ford_balanced(n, G.indptr, G.indices, G.data,
                                                      centers, d, m, p, pc, s, tiebreaking)

            if s.max() > maxsize:
                raise ValueError('maxsize (maximum cluster size) is too small')

            if m.min() < 0 or d.min() < 0:
                raise ValueError('Encountered a disconnected nodes from m or d:  '
                                 f'{m.min()=} {d.min()=})')

            changed2 = amg_core.center_nodes(n, G.indptr, G.indices, G.data,
                                             Cptr, D, P, CC, L, q,
                                             centers, d, m, p, pc, s)

            it += 1

        # slow list version: [len(np.where(p==i)[0]) for i in range(len(p))]
        truepc = np.bincount(p[p > -1], minlength=len(p))
        if np.count_nonzero(truepc - pc):
            raise ValueError('Predecessor count is incorrect.')

        # don't rebalance on last pass
        if riter == rebalance_iters:
            break

        # don't rebalance a single cluster
        if num_clusters < 2:
            break

        # calculate distances
        dist_all.fill(np.inf)
        for a in range(num_clusters):
            N = s[a]  # cluster size
            Nloc = Cptr[a]+N
            if Nloc >= G.shape[0]:
                Nloc = None
            P.fill(-1)
            amg_core.floyd_warshall(G.shape[0], G.indptr, G.indices, G.data,
                                    dist_all[a, :].ravel(), P, CC[Cptr[a]:Nloc], L,
                                    m, a, N)
        # rebalance
        centers, rebalance_change = _rebalance(G, centers, m, d, dist_all, num_clusters)

        # rebalance did nothing
        if not rebalance_change:
            break

    return m, centers


def _rebalance(G, c, m, d, dist_all, num_clusters):
    """Rebalance clusters.

    Parameters
    ----------
    G : sparray
        Sparse graph.
    c : array
        List of centers.
    m : array
        Cluster membership.
    d : array
        Distance to cluster center.
    dist_all : array
        Node-to-node distance for every node in each cluster.
    num_clusters : int
        Number of clusters (= number centers).

    Return
    ------
    array
        List of new centers.
    bool
        Indicate whether centers has changed.

    """
    newc = c.copy()

    # aggregate-to-aggregate neighbors
    AggOp = sparse.coo_array((np.ones(len(m)), (np.arange(len(m)), m))).tocsr()
    Agg2Agg = AggOp.T @ G @ AggOp
    Agg2Agg = Agg2Agg.tocsr()

    # calculate elimination and split measures
    E = _elimination_penalty(G, m, d, dist_all, num_clusters)
    S, c1, c2 = _split_improvement(m, d, dist_all, num_clusters)

    # sort both ascending
    M = np.ones(num_clusters, dtype=bool)
    Esortidx = np.argsort(E)
    Ssortidx = np.argsort(S)

    i_e = 0               # elimination index
    i_s = num_clusters-1  # splitting index

    rebalance_change = False
    # 0, 1, ..., num_clusters-1
    while i_e <= (num_clusters-1) and i_s >= 0:
        a_e = Esortidx[i_e]
        a_s = Ssortidx[i_s]

        if not M[a_e] or a_e == a_s:  # is cluster a_e modifiable and distinct from a_s?
            i_e += 1
            continue

        if not M[a_s]:                # is cluster a_s modifiable?
            i_s -= 1
            continue

        if E[a_e] > S[a_s]:
            break

        M[Agg2Agg[[a_e], :].indices] = False  # neighbors of a_e
        M[Agg2Agg[[a_s], :].indices] = False  # neighbors of a_s
        newc[a_e] = c1[a_s]   # redefine centers
        newc[a_s] = c2[a_s]   # redefine centers
        rebalance_change = True

    return newc, rebalance_change


def _elimination_penalty(A, m, d, dist_all, num_clusters):
    """Calculate elimination penalty.

    see _rebalance()
    """
    Acol = A.tocsc()

    # pylint: disable=too-many-nested-blocks
    E = np.inf * np.ones(num_clusters)
    for a in range(num_clusters):
        E[a] = 0
        Va = np.int32(np.where(m == a)[0])
        N = len(Va)

        for iloc, _ in enumerate(Va):
            dmin = np.inf
            for jloc, j in enumerate(Va):
                for k in Acol[:, [j]].indices:
                    if m[k] != m[j]:
                        jiloc = jloc * N + iloc
                        dmin = min(d[k] + A[k, j] + dist_all[a, jiloc], dmin)
            E[a] += dmin**2
        E[a] -= np.sum(d[Va]**2)
    return E


def _split_improvement(m, d, dist_all, num_clusters):
    """Calculate split improvement.

    see _rebalance()
    """
    S = np.inf * np.ones(num_clusters)
    I = -1 * np.ones(num_clusters, dtype=np.int32)  # better cluster centers if split
    J = -1 * np.ones(num_clusters, dtype=np.int32)  # better cluster centers if split
    for a in range(num_clusters):
        S[a] = np.inf
        Va = np.int32(np.where(m == a)[0])
        N = len(Va)

        for iloc, i in enumerate(Va):
            for jloc, j in enumerate(Va):
                Snew = 0
                for kloc, _ in enumerate(Va):
                    ikloc = iloc * N + kloc
                    jkloc = jloc * N + kloc
                    if dist_all[a, ikloc] < dist_all[a, jkloc]:
                        Snew = Snew + dist_all[a, ikloc]**2
                    else:
                        Snew = Snew + dist_all[a, jkloc]**2
                if Snew < S[a]:
                    S[a] = Snew
                    I[a] = i
                    J[a] = j
        S[a] = np.sum(d[Va]**2) - S[a]
    return S, I, J


def _choice(p):
    """Random selection based on a distribution.

    Parameters
    ----------
    p : array
        Probabilities [0,1], with sum(p) == 1.

    Return
    ------
    int
        Index to a selected integer based on the distribution of p.

    Notes
    -----
    For efficiency, there are no checks.

    """
    a = p / np.max(p)
    i = -1
    while True:
        i = np.random.randint(len(a))
        if np.random.rand() < a[i]:
            break
    return i


def kmeanspp_seed(G, nseeds):
    """K-means++ seed.

    Parameters
    ----------
    G : sparray
        Sparse graph on which to seed.

    nseeds : int
        Number of seeds.

    Return
    ------
    array
        List of seeds.

    Notes
    -----
    This is a reference algorithms, at O(n^3).

    TODO - needs testing

    """
    warn('kmeanspp_seed is O(n^3) -- use only for testing')

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
    G : csr_array, csc_array
        A sparse NxN matrix where each nonzero entry G[i,j] is the distance
        between nodes i and j.
    seed : int
        Index of the seed location.

    Returns
    -------
    order : int array
        Breadth first order.
    level : int array
        Final levels.

    Examples
    --------
    >>> # 0---2
    >>> # |  /
    >>> # | /
    >>> # 1---4---7---8---9
    >>> # |  /|  /
    >>> # | / | /
    >>> # 3/  6/
    >>> # |
    >>> # |
    >>> # 5
    >>> import numpy as np
    >>> import pyamg
    >>> import scipy.sparse as sparse
    >>> edges = np.array([[0,1],[0,2],[1,2],[1,3],[1,4],[3,4],[3,5],
    ...                   [4,6], [4,7], [6,7], [7,8], [8,9]], dtype=np.int32)
    >>> N = np.max(edges.ravel())+1
    >>> data = np.ones((edges.shape[0],))
    >>> A = sparse.coo_array((data, (edges[:,0], edges[:,1])), shape=(N,N))
    >>> c, l = pyamg.graph.breadth_first_search(A, 0)
    >>> print(l)
    [0 1 1 2 2 3 3 3 4 5]
    >>> print(c)
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
    ndarray
        An array of component labels for each vertex of the graph.

    Notes
    -----
    If the nonzero structure of G is not symmetric, then the
    result is undefined.

    Examples
    --------
    >>> from pyamg.graph import connected_components
    >>> print(connected_components( [[0,1,0],[1,0,1],[0,1,0]] ))
    [0 0 0]
    >>> print(connected_components( [[0,1,0],[1,0,0],[0,0,0]] ))
    [0 0 1]
    >>> print(connected_components( [[0,0,0],[0,0,0],[0,0,0]] ))
    [0 1 2]
    >>> print(connected_components( [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]] ))
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
    A : sparray
        Sparse matrix.

    Returns
    -------
    sparray
        Permuted matrix with reordering.

    See Also
    --------
    pseudo_peripheral_node

    Notes
    -----
    Get a pseudo-peripheral node, then call BFS.

    Examples
    --------
    >>> from pyamg import gallery
    >>> from pyamg.graph import symmetric_rcm
    >>> n = 200
    >>> density = 1.0/n
    >>> A = gallery.sprand(n, n, density, format='csr')
    >>> S = A + A.T
    >>> # try the visualizations
    >>> # import matplotlib.pyplot as plt
    >>> # plt.figure()
    >>> # plt.subplot(121)
    >>> # plt.spy(S,marker='.')
    >>> # plt.subplot(122)
    >>> # plt.spy(symmetric_rcm(S),marker='.')

    """
    _dummy_root, order, _dummy_level = pseudo_peripheral_node(A)

    p = order[::-1]

    return A[p, :][:, p]


def pseudo_peripheral_node(A):
    """Find a pseudo peripheral node.

    Parameters
    ----------
    A : sparray
        Sparse matrix.

    Returns
    -------
    x : int
        Location of the node.
    order : array
        BFS ordering.
    level : array
        BFS levels.

    Notes
    -----
    Algorithm in Saad.

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


def metis_partition(G, nparts=5, seed=None):
    """Perform partitioning of graph with weighted edges using METIS.

    Parameters
    ----------
    G : sparray
        A sparse n x n matrix where each nonzero entry G[i,j] is the distance
        between nodes i and j.  G[i,j] is required to be integer.
    nparts : int
        Number of parts in the resulting partition.
    seed : int
        Random seed for METIS.

    Returns
    -------
    array
        Array of n x 1 indices from 0 ... nparts-1.

    """
    G = sparse.csr_array(G)

    if G.dtype.kind != 'i':
        raise ValueError('METIS partitioning requires integer weights')

    if G.nnz > 0:
        if G.data.min() < 0:
            raise ValueError('METIS partitioning requires positive integer weights.')

    if not isinstance(nparts, int) or nparts < 1:
        raise ValueError('nparts should be a positive integer')

    try:
        import pymetis  # noqa: PLC0415
    except ImportError as expt:
        raise ImportError('pymetis required for METIS partitioning') from expt

    # set diagonal to zero and force reallocation
    G = G.tocoo()
    G.setdiag(0)
    G = G.tocsr()

    # metis options
    opt = pymetis.Options()
    opt.contig = 1
    if seed:
        opt.seed = seed
    _, parts = pymetis.part_graph(nparts, xadj=G.indptr, adjncy=G.indices, eweights=G.data,
                                  options=opt)

    return np.array(parts)
