"""Functions to compute C/F splittings for use in Classical AMG

Overview
--------

A C/F splitting is a partitioning of the nodes of a strength of 
connection matrix (denoted S) into sets of C (coarse) and F (fine) nodes.
The C-nodes will be promoted to the coarser grid while the F-nodes
are retained on the finer grid.  Ideally, the C-nodes, which represent
the coarse-level unknowns, should be far fewer in number than the F-nodes.
Furthermore, algebraically smooth error must be well-approximated by
the coarse level degrees of freedom.


Representation
--------------

C/F splitting is represented by an array with ones for all the C-nodes
and zeros for the F-nodes.


C/F Splitting Methods
---------------------

RS : Original Ruge-Stuben method
    - Produces good C/F splittings but is inherently serial.  
    - May produce AMG hierarchies with relatively high operator complexities.
    - See References [1,4]

PMIS: Parallel Modified Independent Set 
    - Very fast construction with low operator complexity.  
    - Convergence can deteriorate with increasing problem 
      size on structured meshes.  
    - Uses method similar to Luby's Maximal Independent Set algorithm.
    - See References [1,3]

PMISc: Parallel Modified Independent Set in Color
    - Fast construction with low operator complexity.  
    - Better scalability than PMIS on structured meshes.
    - Augments random weights with a (graph) vertex coloring
    - See References [1]
      
CLJP: Clearly-Luby-Jones-Plassmann [1,2]
    - Parallel method with cost and complexity comparable to Ruge-Stuben.
    - Convergence can deteriorate with increasing problem 
      size on structured meshes.  
    - See References [1,2]

CLJP: Clearly-Luby-Jones-Plassmann in Color [1]
    - Parallel method with cost and complexity comparable to Ruge-Stuben.
    - Better scalability than CLJP on structured meshes.
    - See References [1]
    

Summary
-------

In general, methods that use a graph coloring perform better on 
structured meshes[1].  Unstructured meshes do not appear to benefit 
substaintially from coloring.

    ========  ========  ========  ==========  
     method   parallel  in color     cost    
    ========  ========  ========  ==========
       RS        no        no      moderate
      PMIS      yes        no      very low
      PMISc     yes       yes        low
      CLJP      yes        no      moderate         
      CLJPc     yes       yes      moderate
    ========  ========  ========  ==========
    

References
----------

    [1] David M. Alber and Luke N. Olson
    "Parallel coarse-grid selection"
    Numerical Linear Algebra with Applications 2007; 14:611-643.

    [2] Cleary AJ, Falgout RD, Henson VE, Jones JE. 
    "Coarse-grid selection for parallel algebraic multigrid"
    Proceedings of the 5th International Symposium on Solving Irregularly 
    Structured Problems in Parallel. Springer: Berlin, 1998; 104–115.

    [3] Hans De Sterck, Ulrike M Yang, and Jeffrey J Heys
    "Reducing complexity in parallel algebraic multigrid preconditioners" 
    SIAM Journal on Matrix Analysis and Applications 2006; 27:1019–1039.

    [4] Ruge JW, Stuben K. 
    "Algebraic multigrid (AMG)"
    In Multigrid Methods, McCormick SF (ed.), Frontiers in Applied Mathematics, vol. 3. 
    SIAM: Philadelphia, PA, 1987; 73–130.
    

"""

from scipy import ones, empty, rand, ravel
from scipy.sparse import csr_matrix, isspmatrix_csr

from graph import vertex_coloring

import multigridtools

__all__ = ['RS', 'PMIS', 'PMISc', 'CLJP', 'CLJPc']

def RS(S):
    """Compute a C/F splitting using Ruge-Stuben coarsening
    """
    raise NotImplementedError

def PMIS(S):
    """C/F splitting using the Parallel Modified Independent Set method

    """
    weights,G,S,T = preprocess(S,False)
    return MIS(G, weights)

def PMISc(S):
    """C/F splitting using Parallel Modified Independent Set (in color)

    PMIS-c, or PMIS in color, improves PMIS by perturbing the initial 
    random weights with weights determined by a vertex coloring.

    """
    weights,G,S,T = preprocess(S,True)
    return MIS(G, weights)
     

def CLJP(S):
    """Compute a C/F splitting using the parallel CLJP algorithm
    """
    raise NotImplementedError


def CLJPc(S):
    """Compute a C/F splitting using the parallel CLJP-c algorithm
    
    CLJP-c, or CLJP in color, improves CLJP by perturbing the initial 
    random weights with weights determined by a vertex coloring.

    """
    raise NotImplementedError


#############################
## Helper functions
#############################

def CLJP_update_weights(weights,G,S,T,D):
    raise NotImplementedError

def preprocess(S, use_color):
    """Common preprocess for splitting functions
    
    Performs the following operations:
        - Checks input strength of connection matrix S
        - Replaces S.data with ones
        - Creates T = S.T in CSR format
        - Creates G = S union T in CSR format
        - Creates random weights 
        - Augments weights with graph coloring (if use_color == True)

    Returns:
        (weights,G,S,T)

    """

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
    """compute an idependent set in parallel with given weights"""

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


