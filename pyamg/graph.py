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


def bellman_ford(G, seeds):
    """Bellman-Ford iteration.

    Parameters
    ----------
    G : sparray
        Directed graph with positive weights.
    seeds : list
        Starting seeds.

    Returns
    -------
    array
        Distance of each point to the nearest seed.
    array
        Index of the nearest seed.

    See Also
    --------
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
    N = G.shape[0]

    if G.nnz > 0:
        if G.data.min() < 0:
            raise ValueError('Bellman-Ford is defined only for positive weights.')
    if G.dtype == complex:
        raise ValueError('Bellman-Ford is defined only for real weights.')

    seeds = np.asarray(seeds, dtype='intc')

    distances = np.full(N, np.inf, dtype=G.dtype)
    distances[seeds] = 0

    nearest_seed = np.full(N, -1, dtype='intc')
    nearest_seed[seeds] = seeds

    amg_core.bellman_ford(N, G.indptr, G.indices, G.data, distances, nearest_seed)

    return (distances, nearest_seed)


def lloyd_cluster(G, seeds, maxiter=10):
    """Perform Lloyd clustering on graph with weighted edges.

    Parameters
    ----------
    G : csr_array, csc_array
        A sparse matrix of size (n,n) where each nonzero entry G[i,j] is the distance
        between nodes i and j.
    seeds : int array
        If seeds is an integer, then its value determines the number of
        clusters.  Otherwise, seeds is an array of unique integers between 0
        and N-1 that will be used as the initial seeds for clustering.
    maxiter : int
        The maximum number of iterations to perform.

    Returns
    -------
    array
        Final distances.
    int array
        Id of each cluster of points.
    int array
        Index of each seed.

    Notes
    -----
    If G has complex values, abs(G) is used instead.

    """
    G = asgraph(G)
    N = G.shape[0]

    if G.dtype.kind == 'c':
        # complex dtype
        G = np.abs(G)

    # interpret seeds argument
    if np.isscalar(seeds):
        seeds = np.random.permutation(N)[:seeds]
        seeds = seeds.astype('intc')
    else:
        seeds = np.array(seeds, dtype='intc')

    if len(seeds) < 1:
        raise ValueError('at least one seed is required')

    if seeds.min() < 0:
        raise ValueError(f'Invalid seed index ({seeds.min()})')
    if seeds.max() >= N:
        raise ValueError(f'Invalid seed index ({seeds.max()})')

    clusters = np.empty(N, dtype='intc')
    distances = np.empty(N, dtype=G.dtype)

    for _it in range(1, maxiter+1):
        last_seeds = seeds.copy()

        amg_core.lloyd_cluster(N, G.indptr, G.indices, G.data,
                               len(seeds), distances, clusters, seeds)

        if (seeds == last_seeds).all():
            break

    if _it == maxiter:
        warn('Lloyd clustering reached maxiter (did not converge)')

    return (distances, clusters, seeds)


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
