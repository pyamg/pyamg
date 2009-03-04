"""Algorithms related to graphs"""

__docformat__ = "restructuredtext en"

import numpy
import scipy
from scipy import sparse

import amg_core

__all__ = ['maximal_independent_set', 'vertex_coloring', 'bellman_ford', \
           'lloyd_cluster', 'connected_components']


def max_value(datatype):
    try:
        return numpy.iinfo(datatype).max
    except:
        return numpy.finfo(datatype).max


def asgraph(G):
    if not ( sparse.isspmatrix_csr(G) or sparse.isspmatrix_csc(G) ):
        G = sparse.csr_matrix(G)

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

    mis = numpy.empty(N, dtype='intc')
    mis[:] = -1

    if k is None:
        if algo == 'serial':
            fn = amg_core.maximal_independent_set_serial
            fn(N, G.indptr, G.indices, -1, 1, 0, mis)
        elif algo == 'parallel':
            fn = amg_core.maximal_independent_set_parallel
            fn(N, G.indptr, G.indices, -1, 1, 0, mis, scipy.rand(N))
        else:
            raise ValueError('unknown algorithm (%s)' % algo)
    else:
        fn = amg_core.maximal_independent_set_k_parallel
        fn(N, G.indptr, G.indices, k, mis, scipy.rand(N)) 


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
    
    coloring = numpy.empty(N, dtype='intc')

    if method == 'MIS':
        fn = amg_core.vertex_coloring_mis
        fn(N, G.indptr, G.indices, coloring)
    elif method == 'JP':
        fn = amg_core.vertex_coloring_jones_plassmann
        fn(N, G.indptr, G.indices, coloring, scipy.rand(N) )
    elif method == 'LDF':
        fn = amg_core.vertex_coloring_LDF
        fn(N, G.indptr, G.indices, coloring, scipy.rand(N) )
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

    seeds = numpy.asarray(seeds, dtype='intc')

    distances        = numpy.empty(N, dtype=G.dtype)
    distances[:]     = max_value(G.dtype)
    distances[seeds] = 0

    nearest_seed        = numpy.empty(N, dtype='intc')
    nearest_seed[:]     = -1
    nearest_seed[seeds] = seeds
   
    old_distances = numpy.empty_like(distances)

    iter = 0
    while maxiter is None or iter < maxiter:
        old_distances[:] = distances

        amg_core.bellman_ford( N, G.indptr, G.indices, G.data,
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
        between nodes i and j.
    seeds : {int, array}
        If seeds is an integer, then its value determines the number of clusters.
        Otherwise, seeds is an array of unique integers between 0 and N-1 that
        will be used as the initial seeds for clustering.
    maxiter : int
        The maximum number of iterations to perform. 

    Notes
    -----
    If G has complex values, abs(G) is used instead.

    """
    G = asgraph(G)
    N = G.shape[0]
    
    if G.dtype.kind == 'c':
        # complex dtype
        G = numpy.abs(G)
    
    #interpret seeds argument
    if numpy.isscalar(seeds):
        seeds = numpy.random.permutation(N)[:seeds]
        seeds = seeds.astype('intc')
    else: 
        seeds = numpy.asarray(seeds, dtype='intc', copy=True)

    if len(seeds) < 1:
        raise ValueError('at least one seed is required')
    
    if seeds.min() < 0:
        raise ValueError('invalid seed index (%d)' % seeds.min())
    if seeds.max() >= N:
        raise ValueError('invalid seed index (%d)' % seeds.max())

    clusters  = numpy.empty(N, dtype='intc')
    distances = numpy.empty(N, dtype=G.dtype)
    
    for i in range(maxiter):
        last_seeds = seeds.copy()

        amg_core.lloyd_cluster(N, G.indptr, G.indices, G.data, \
                len(seeds), distances, clusters, seeds)

        if (seeds == last_seeds).all():
            break
    
    return (distances, clusters, seeds)


def breadth_first_search(G, seed):
    """Breadth First search of a graph

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
    #TODO document

    G = asgraph(G)
    N = G.shape[0]

    #Check symmetry?

    order = numpy.empty(N, G.indptr.dtype)
    level = numpy.empty(N, G.indptr.dtype)
    level[:] = -1

    BFS = amg_core.breadth_first_search
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

    #Check symmetry?
    components = numpy.empty(N, G.indptr.dtype)
    
    fn = amg_core.connected_components
    fn(N, G.indptr, G.indices, components)

    return components

