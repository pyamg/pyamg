"""Algorithms related to graphs"""

__docformat__ = "restructuredtext en"

import numpy
from warnings import warn
from numpy import zeros, empty, asarray, empty_like, isscalar, abs
from scipy import rand
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_csc

import multigridtools

__all__ = ['maximal_independent_set', 'vertex_coloring', 'bellman_ford', \
           'lloyd_cluster', 'connected_components']


def max_value(datatype):
    try:
        return numpy.iinfo(datatype).max
    except:
        return numpy.finfo(datatype).max


def asgraph(G):
    if not ( isspmatrix_csr(G) or isspmatrix_csc(G) ):
        G = csr_matrix(G)

    if G.shape[0] != G.shape[1]:
        raise ValueError('expected square matrix')

    return G


def maximal_independent_set(G, algo='serial', k=None):
    """Compute a maximal independent vertex set for a graph

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
    An array S where 
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

    mis = empty(N, dtype='intc')
    mis[:] = -1

    if k is None:
        if algo == 'serial':
            fn = multigridtools.maximal_independent_set_serial
            fn(N, G.indptr, G.indices, -1, 1, 0, mis)
        elif algo == 'parallel':
            fn = multigridtools.maximal_independent_set_parallel
            fn(N, G.indptr, G.indices, -1, 1, 0, mis, rand(N))
        else:
            raise ValueError('unknown algorithm (%s)' % algo)
    else:
        fn = multigridtools.maximal_independent_set_k_parallel
        fn(N, G.indptr, G.indices, k, mis, rand(N)) 


    return mis


def vertex_coloring(G, method='MIS'):
    """Compute a vertex coloring of a graph 

    Parameters
    ----------
    G : sparse matrix
        Symmetric matrix, preferably in sparse CSR or CSC format
        The nonzeros of G represent the edges of an undirected graph.
    method : {string}
        Algorithm used to compute the vertex coloring:
            * 'MIS' - Maximal Independent Set
            * 'JP'  - Jones-Plassmann (parallel)
            * 'LDF' - Largest-Degree-First (parallel)

    Returns
    -------
    An array of vertex colors (integers beginning at 0)

    Notes
    -----
    Diagonal entries in the G (self loops) will be ignored.

    """

    G = asgraph(G)
    N = G.shape[0]
    
    coloring    = empty(N, dtype='intc')

    if method == 'MIS':
        fn = multigridtools.vertex_coloring_mis
        fn(N, G.indptr, G.indices, coloring)
    elif method == 'JP':
        fn = multigridtools.vertex_coloring_jones_plassmann
        fn(N, G.indptr, G.indices, coloring, rand(N) )
    elif method == 'LDF':
        fn = multigridtools.vertex_coloring_LDF
        fn(N, G.indptr, G.indices, coloring, rand(N) )
    else:
        raise ValueError('unknown method (%s)' % method)

    return coloring

def bellman_ford(G, seeds, maxiter=None):
    """
    Bellman-Ford iteration

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    References
    ----------
    CLR

    Examples
    --------
    """
    G = asgraph(G)
    N = G.shape[0]
    
    if maxiter is not None and maxiter < 0:
        raise ValueError('maxiter must be positive')
    if G.dtype == complex:
        raise ValueError('Bellman-Ford algorithm only defined for real weights')

    seeds = asarray(seeds, dtype='intc')

    distances        = empty( N, dtype=G.dtype )
    distances[:]     = max_value(G.dtype)
    distances[seeds] = 0

    nearest_seed        = empty(N, dtype='intc')
    nearest_seed[:]     = -1
    nearest_seed[seeds] = seeds
   
    old_distances = empty_like(distances)

    iter = 0
    while maxiter is None or iter < maxiter:
        old_distances[:] = distances

        multigridtools.bellman_ford( N, G.indptr, G.indices, G.data,
                                    distances, nearest_seed)
        
        if (old_distances == distances).all():
            break

    return (distances,nearest_seed)

def lloyd_cluster(G, seeds, maxiter=10):
    """Perform Lloyd clustering on graph with weighted edges

    Parameters
    ----------
    G : csr_matrix or csc_matrix
        A sparse NxN matrix where each nonzero entry G[i,j] is the distance 
        between nodes i and j
    seeds : {int, array}
        If seeds is an integer, then its value determines the number of clusters.
        Otherwise, seeds is an array of unique integers between 0 and N-1 that
        will be used as the initial seeds for clustering.
    maxiter : int
        The maximum number of iterations to perform. 

    """
    G = asgraph(G)
    N = G.shape[0]
    
    if G.dtype == complex:
        warn("Converting complex to real for lloyd_cluster")
        G = G.copy()
        G.data = abs(G.data)
        G = G.astype(float)
    
    #interpret seeds argument
    if isscalar(seeds):
        seeds = numpy.random.permutation(N)[:seeds]
        seeds = seeds.astype('intc')
    else: 
        seeds = asarray(seeds, dtype='intc', copy=True)

    if len(seeds) < 1:
        raise ValueError('at least one seed is required')
    
    if seeds.min() < 0:
        raise ValueError('invalid seed index (%d)' % seeds.min())
    if seeds.max() >= N:
        raise ValueError('invalid seed index (%d)' % seeds.max())

    clusters  = empty( N, dtype='intc')
    distances = empty( N, dtype=G.dtype)
    
    for i in range(maxiter):
        last_seeds = seeds.copy()

        multigridtools.lloyd_cluster(N, G.indptr, G.indices, G.data, \
                len(seeds), distances, clusters, seeds)

        if (seeds == last_seeds).all():
            break
    
    return (distances, clusters, seeds)


def breadth_first_search(G, seed):
    """
    Breadth First search of a graph

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    References
    ----------
    CLR

    Examples
    --------
    """
    G = asgraph(G)
    N = G.shape[0]

    #Check symmetry?

    order = empty(N, G.indptr.dtype)
    level = empty(N, G.indptr.dtype)
    level[:] = -1

    BFS = multigridtools.breadth_first_search
    BFS(G.indptr, G.indices, int(seed), order, level)

    return order,level

def connected_components(G):
    """Compute the connected components of a graph

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
    >>> connected_components( [[0,1,0],[1,0,1],[0,1,0]] )
    array([0, 0, 0])
    >>> connected_components( [[0,1,0],[1,0,0],[0,0,0]] )
    array([0, 0, 1])
    >>> connected_components( [[0,0,0],[0,0,0],[0,0,0]] )
    array([0, 1, 2])
    >>> connected_components( [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]] )
    array([0, 0, 1, 1])

    """    
    G = asgraph(G)
    N = G.shape[0]

    #Check symmetry?
    components = empty(N, G.indptr.dtype)
    
    fn = multigridtools.connected_components
    fn(N, G.indptr, G.indices, components)

    return components

