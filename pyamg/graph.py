"""Algorithms related to Graphs"""

from numpy import zeros, empty
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix_csc

import multigridtools

__all__ = ['maximal_independent_set','vertex_coloring']

def maximal_independent_set(G):
    """Compute a maximal independent vertex set for a graph

    Parameters
    ==========
        G - symmetric matrix (e.g. csr_matrix or csc_matrix)

    Returns
    =======
        An array S where 
            S[i] = 1 if vertex i is in the MIS
            S[i] = 0 otherwise

    Notes
    =====
        Diagonal entries in the G are acceptable (will be ignored).

    """

    if not ( isspmatrix_csr(G) or isspmatrix_csc(G) ):
        G = csr_matrix(G)

    if G.shape[0] != G.shape[1]:
        raise ValueError('expected square matrix')

    fn = multigridtools.maximal_independent_set

    mis = zeros(G.shape[0], dtype='intc')

    fn(G.shape[0], G.indptr, G.indices, 1, mis)

    return mis


def vertex_coloring(G):
    """Compute a vertex coloring of a graph 

    Parameters
    ==========
        G - symmetric matrix (e.g. csr_matrix or csc_matrix)

    Returns
    =======
        An array of vertex colors

    Notes
    =====
        Diagonal entries in the G are acceptable (will be ignored).

    """

    if not ( isspmatrix_csr(G) or isspmatrix_csc(G) ):
        G = csr_matrix(G)

    if G.shape[0] != G.shape[1]:
        raise ValueError('expected square matrix')

    fn = multigridtools.vertex_coloring_mis

    coloring = empty(G.shape[0], dtype='intc')

    fn(G.shape[0], G.indptr, G.indices, coloring)

    return coloring

    
