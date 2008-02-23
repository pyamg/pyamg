"""Functions to compute C/F splittings for use in Classical AMG

A C/F splitting is a partitioning of the nodes of a strength of 
connection matrix into sets of C (coarse) and F (fine) nodes.
The C nodes will be promoted to the coarser grid while the F nodes
are retained on the finer grid.  Ideally, the C nodes, which represent
the coarse-level unknowns, should be far fewer in number than the F nodes.
Furthermore, algebraically smooth error must be well-approximated by
the coarse level degrees of freedom.

TODO describe parallel vs. serial

References
----------
    David M. Alber and Luke N. Olson
    "Parallel coarse-grid selection"
    Numerical Linear Algebra with Applications 2007; 14:611-643

"""

from scipy import ones, empty, rand, ravel
from scipy.sparse import csr_matrix, isspmatrix_csr

from graph import vertex_coloring

import multigridtools

__all__ = ['PMISc', 'CLJPc']

def preprocess(S, use_color):
    if not isspmatrix_csr(S): raise TypeError('expected csr_matrix')

    if S.shape[0] != S.shape[1]:
        raise ValueError('expected square matrix, shape=%s' % (S.shape,) )

    N = S.shape[0]
    S = csr_matrix( (ones(S.nnz,dtype='int8'),S.indices,S.indptr), shape=(N,N))
    T = S.T.tocsr()     #transpose S for efficient column access

    G = S + T           # form graph (must be symmetric)
    G.data[:] = 1

    weights   = ravel(T.sum(axis=1))  # initial weights
    #weights -= T.diagonal()          # discount self loops

    if use_color:
        coloring = vertex_coloring(G)
        num_colors = coloring.max() + 1
        weights  = weights + (rand(len(weights)) + coloring)/num_colors
    else:
        weights  = weights + rand(len(weights))

    return (weights,G,S,T)

def MIS(G, weights, maxiter=None):
    """compute an idependent set in parallel"""

    mis    = empty( G.shape[0], dtype='intc' )
    mis[:] = -1
    
    fn = multigridtools.maximal_independent_set_parallel
        
    if maxiter is None:
        fn(G.shape[0], G.indptr, G.indices, -1, 1, 0, mis, weights)
    else:
        if maxiter < 0:
            raise ValueError('maxiter must be >= 0')

        fn(G.shape[0], G.indptr, G.indices, -1, 1, 0, mis, weights, maxiter)

    return mis



def PMISc(S):
    """C/F splitting using the parallel MIS-c algorithm

    PMIS-c, or PMIS in color, improves PMIS by perturbing the initial 
    random weights with weights determined by a vertex coloring.

    """

    weights,G,S,T = preprocess(S,True)
    return MIS(G, weights)
     
    

def CLJPc(S):
    """Compute a C/F splitting using the parallel CLJP-c algorithm
    
    CLJP-c, or CLJP in color, improves CLJP by perturbing the initial 
    random weights with weights determined by a vertex coloring.

    """
    raise NotImplementedError



common_docstring = \
"""
    Parameters
    ----------
    S : csr_matrix
        strength of connection matrix

    Returns
    -------
    splitting : ndarray
        splitting[i] = 1 if the i-th variable is a C-node        
        splitting[i] = 0 if the i-th variable is a F-node        

"""


