"""Algorithms related to Graphs"""

from numpy import zeros, empty
from scipy import rand
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_csc

import multigridtools

__all__ = ['maximal_independent_set','vertex_coloring']

def asgraph(G):
    if not ( isspmatrix_csr(G) or isspmatrix_csc(G) ):
        G = csr_matrix(G)

    if G.shape[0] != G.shape[1]:
        raise ValueError('expected square matrix')

    return G


def maximal_independent_set(G, algo='serial'):
    """Compute a maximal independent vertex set for a graph

    Parameters
    ==========
        G    - symmetric matrix (e.g. csr_matrix or csc_matrix)
        algo - {'serial', 'parallel'}
                Algorithm used to compute the MIS:
                    serial   - greedy serial algorithm
                    parallel - variant of Luby's parallel MIS algorithm

    Returns
    =======
        An array S where 
            S[i] = 1 if vertex i is in the MIS
            S[i] = 0 otherwise

    Notes
    =====
        Diagonal entries in the G (self loops) will be ignored.
        
        Luby's algorithm is significantly more expensive than the 
        greedy serial algorithm.

    """

    G = asgraph(G)
    N = G.shape[0]

    mis = empty(N, dtype='intc')
    mis[:] = -1

    if algo == 'serial':
        fn = multigridtools.maximal_independent_set_serial
        fn(N, G.indptr, G.indices, -1, 1, 0, mis)
    elif algo == 'parallel':
        fn = multigridtools.maximal_independent_set_parallel
        fn(N, G.indptr, G.indices, -1, 1, 0, mis, rand(N))
    else:
        raise ValueError('unknown algorithm (%s)' % algo)

    return mis


def vertex_coloring(G, algo='serial'):
    """Compute a vertex coloring of a graph 

    Parameters
    ==========
        G    - symmetric matrix (e.g. csr_matrix or csc_matrix)
        algo - {'serial', 'parallel'}
                Algorithm used to compute the MIS:
                    serial   - greedy serial algorithm
                    parallel - variant of Luby's parallel MIS algorithm

    Returns
    =======
        An array of vertex colors

    Notes
    =====
        Diagonal entries in the G (self loops) will be ignored.

    """

    G = asgraph(G)
    N = G.shape[0]
    
    coloring    = empty(N, dtype='intc')
    coloring[:] = -1

    if algo == 'serial':
        def mis(K):
            return multigridtools.maximal_independent_set_serial(
                    N, G.indptr, G.indices, -1 - K, K, -2 - K, coloring)
    elif algo == 'parallel':
        rand_vals = rand(N)
        def mis(K):
            return multigridtools.maximal_independent_set_parallel(
                    N, G.indptr, G.indices, 
                    -1 - K, K, -2 - K, 
                    coloring, rand_vals)
    else:
        raise ValueError('unknown algorithm (%s)' % algo)

    count = 0  # number of colored vertices 
    gamma = 0  # number of colors

    # color each MIS with a different color 
    # until all vertices are colored
    while count < N:
        count += mis(gamma)
        gamma += 1

    return coloring




    ## multigridtools method
    #fn = multigridtools.vertex_coloring_mis
    #coloring = empty(N, dtype='intc')
    #fn(N, G.indptr, G.indices, coloring)
    #return coloring
    
