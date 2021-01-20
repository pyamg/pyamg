"""Algorithms related to graphs."""
from __future__ import absolute_import


import numpy as np
import scipy as sp
from scipy import sparse

from . import amg_core

__all__ = ['maximal_independent_set', 'vertex_coloring',
           'bellman_ford',
           'lloyd_cluster', 'connected_components']

from pyamg.graph_ref import bellman_ford_reference

__all__ += ['bellman_ford_reference', 'bellman_ford_balanced_reference']


def asgraph(G):
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
            raise ValueError('unknown algorithm (%s)' % algo)
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
        raise ValueError('unknown method (%s)' % method)

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
        amg_core.bellman_ford(n, G.indptr, G.indices, G.data, centers, # IN
                              distances, nearest, predecessors,        # OUT
                              True)
    elif method == 'balanced':
        amg_core.bellman_ford(n, G.indptr, G.indices, G.data, centers, # IN
                              distances, nearest, predecessors,        # OUT
                              predecessors_count, cluster_size,        # OUT
                              True)
    else:
        raise ValueError(f'method {method} is not supported in Bellman-Ford')

    return distances, nearest, predecessors


def lloyd_cluster(G, centers):
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
    distances : array
        final distances
    clusters : int array
        id of each cluster of points
    centers : int array
        index of each center

    Notes
    -----
    If G has complex values, abs(G) is used instead.

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
        centers = centers.astype('intc')
    else:
        centers = np.asarray(centers, dtype=np.int32)

    if len(centers) < 1:
        raise ValueError('at least one center is required')

    if centers.min() < 0:
        raise ValueError(f'invalid center index {centers.min()}')
    if centers.max() >= n:
        raise ValueError(f'invalid center index {centers.max()}')

    centers = np.asarray(centers, dtype=np.int32)

    distances = np.empty(n, dtype=G.dtype)
    olddistances = np.empty(n, dtype=G.dtype)
    clusters = np.empty(n, dtype=np.int32)
    predecessors = np.full(n, -1, dtype=np.int32)

    amg_core.lloyd_cluster(n, G.indptr, G.indices, G.data,                  # IN
                           centers,                                         # INOUT
                           distances, olddistances, clusters, predecessors, # OUT
                           True)

    return distances, clusters, centers


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
    n = A.shape[0]

    root, order, level = pseudo_peripheral_node(A)

    Perm = sparse.identity(n, format='csr')
    p = level.argsort()
    Perm = Perm[p, :]

    return Perm * A * Perm.T


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
    from pyamg.graph import breadth_first_search
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
