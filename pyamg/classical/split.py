"""Functions to compute C/F splittings for use in Classical AMG

Overview
--------

A C/F splitting is a partitioning of the nodes in the graph of as connection
matrix (denoted S for strength) into sets of C (coarse) and F (fine) nodes.
The C-nodes are promoted to the coarser grid while the F-nodes are retained
on the finer grid.  Ideally, the C-nodes, which represent the coarse-level
unknowns, should be far fewer in number than the F-nodes.  Furthermore,
algebraically smooth error must be well-approximated by the coarse level
degrees of freedom.


Representation
--------------

C/F splitting is represented by an array with ones for all the C-nodes
and zeros for the F-nodes.


C/F Splitting Methods
---------------------

RS : Original Ruge-Stuben method
    - Produces good C/F splittings but is inherently serial.
    - May produce AMG hierarchies with relatively high operator complexities.
    - See References [1] and [4]

PMIS: Parallel Modified Independent Set
    - Very fast construction with low operator complexity.
    - Convergence can deteriorate with increasing problem
      size on structured meshes.
    - Uses method similar to Luby's Maximal Independent Set algorithm.
    - See References [1] and [3]

PMISc: Parallel Modified Independent Set in Color
    - Fast construction with low operator complexity.
    - Better scalability than PMIS on structured meshes.
    - Augments random weights with a (graph) vertex coloring
    - See References [1]

CLJP: Clearly-Luby-Jones-Plassmann
    - Parallel method with cost and complexity comparable to Ruge-Stuben.
    - Convergence can deteriorate with increasing problem
      size on structured meshes.
    - See References [1] and [2]

CLJP-c: Clearly-Luby-Jones-Plassmann in Color
    - Parallel method with cost and complexity comparable to Ruge-Stuben.
    - Better scalability than CLJP on structured meshes.
    - See References [1]


Summary
-------

In general, methods that use a graph coloring perform better on structured
meshes [1].  Unstructured meshes do not appear to benefit substantially
from coloring.

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

..  [1] Cleary AJ, Falgout RD, Henson VE, Jones JE.
    "Coarse-grid selection for parallel algebraic multigrid"
    Proceedings of the 5th International Symposium on Solving Irregularly
    Structured Problems in Parallel. Springer: Berlin, 1998; 104-115.

..  [2] David M. Alber and Luke N. Olson
    "Parallel coarse-grid selection"
    Numerical Linear Algebra with Applications 2007; 14:611-643.

..  [3] Hans De Sterck, Ulrike M Yang, and Jeffrey J Heys
    "Reducing complexity in parallel algebraic multigrid preconditioners"
    SIAM Journal on Matrix Analysis and Applications 2006; 27:1019-1039.

..  [4] Ruge JW, Stuben K.
    "Algebraic multigrid (AMG)"
    In Multigrid Methods, McCormick SF (ed.),
    Frontiers in Applied Mathematics, vol. 3.
    SIAM: Philadelphia, PA, 1987; 73-130.


"""

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, isspmatrix_csr

from pyamg.graph import vertex_coloring
from pyamg import amg_core
from pyamg.util.utils import remove_diagonal

__all__ = ['RS', 'PMIS', 'PMISc', 'MIS']


def RS(S):
    """Compute a C/F splitting using Ruge-Stuben coarsening

    Parameters
    ----------
    S : csr_matrix
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)

    Returns
    -------
    splitting : ndarray
        Array of length of S of ones (coarse) and zeros (fine)

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import RS
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = RS(S)

    See Also
    --------
    amg_core.rs_cf_splitting

    References
    ----------
    .. [5] Ruge JW, Stuben K.  "Algebraic multigrid (AMG)"
       In Multigrid Methods, McCormick SF (ed.),
       Frontiers in Applied Mathematics, vol. 3.
       SIAM: Philadelphia, PA, 1987; 73-130.

    """
    if not isspmatrix_csr(S):
        raise TypeError('expected csr_matrix')
    S = remove_diagonal(S)

    T = S.T.tocsr()  # transpose S for efficient column access

    splitting = np.empty(S.shape[0], dtype='intc')

    amg_core.rs_cf_splitting(S.shape[0],
                             S.indptr, S.indices,
                             T.indptr, T.indices,
                             splitting)

    return splitting


def PMIS(S):
    """C/F splitting using the Parallel Modified Independent Set method

    Parameters
    ----------
    S : csr_matrix
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)

    Returns
    -------
    splitting : ndarray
        Array of length of S of ones (coarse) and zeros (fine)

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import PMIS
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = PMIS(S)

    See Also
    --------
    MIS

    References
    ----------
    .. [6] Hans De Sterck, Ulrike M Yang, and Jeffrey J Heys
       "Reducing complexity in parallel algebraic multigrid preconditioners"
       SIAM Journal on Matrix Analysis and Applications 2006; 27:1019-1039.

    """
    S = remove_diagonal(S)
    weights, G, S, T = preprocess(S)
    return MIS(G, weights)


def PMISc(S, method='JP'):
    """C/F splitting using Parallel Modified Independent Set (in color)

    PMIS-c, or PMIS in color, improves PMIS by perturbing the initial
    random weights with weights determined by a vertex coloring.

    Parameters
    ----------
    S : csr_matrix
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)
    method : string
        Algorithm used to compute the initial vertex coloring:
            * 'MIS' - Maximal Independent Set
            * 'JP'  - Jones-Plassmann (parallel)
            * 'LDF' - Largest-Degree-First (parallel)

    Returns
    -------
    splitting : array
        Array of length of S of ones (coarse) and zeros (fine)

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import PMISc
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = PMISc(S)

    See Also
    --------
    MIS

    References
    ----------
    .. [7] David M. Alber and Luke N. Olson
       "Parallel coarse-grid selection"
       Numerical Linear Algebra with Applications 2007; 14:611-643.

    """
    S = remove_diagonal(S)
    weights, G, S, T = preprocess(S, coloring_method=method)
    return MIS(G, weights)


def CLJP(S, color=False):
    """Compute a C/F splitting using the parallel CLJP algorithm

    Parameters
    ----------
    S : csr_matrix
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)
    color : bool
        use the CLJP coloring approach

    Returns
    -------
    splitting : array
        Array of length of S of ones (coarse) and zeros (fine)

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.split import CLJP
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = CLJP(S)

    See Also
    --------
    MIS, PMIS, CLJPc

    References
    ----------
    .. [8] David M. Alber and Luke N. Olson
       "Parallel coarse-grid selection"
       Numerical Linear Algebra with Applications 2007; 14:611-643.

    """
    if not isspmatrix_csr(S):
        raise TypeError('expected csr_matrix')
    S = remove_diagonal(S)

    colorid = 0
    if color:
        colorid = 1

    T = S.T.tocsr()  # transpose S for efficient column access
    splitting = np.empty(S.shape[0], dtype='intc')

    amg_core.cljp_naive_splitting(S.shape[0],
                                  S.indptr, S.indices,
                                  T.indptr, T.indices,
                                  splitting,
                                  colorid)

    return splitting


def CLJPc(S):
    """Compute a C/F splitting using the parallel CLJP-c algorithm

    CLJP-c, or CLJP in color, improves CLJP by perturbing the initial
    random weights with weights determined by a vertex coloring.

    Parameters
    ----------
    S : csr_matrix
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)

    Returns
    -------
    splitting : array
        Array of length of S of ones (coarse) and zeros (fine)

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.split import CLJPc
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = CLJPc(S)

    See Also
    --------
    MIS, PMIS, CLJP

    References
    ----------
    .. [1] David M. Alber and Luke N. Olson
       "Parallel coarse-grid selection"
       Numerical Linear Algebra with Applications 2007; 14:611-643.

    """
    S = remove_diagonal(S)
    return CLJP(S, color=True)


def MIS(G, weights, maxiter=None):
    """Compute a maximal independent set of a graph in parallel

    Parameters
    ----------
    G : csr_matrix
        Matrix graph, G[i,j] != 0 indicates an edge
    weights : ndarray
        Array of weights for each vertex in the graph G
    maxiter : int
        Maximum number of iterations (default: None)

    Returns
    -------
    mis : array
        Array of length of G of zeros/ones indicating the independent set

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical import MIS
    >>> import numpy as np
    >>> G = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> w = np.ones((G.shape[0],1)).ravel()
    >>> mis = MIS(G,w)

    See Also
    --------
    fn = amg_core.maximal_independent_set_parallel

    """

    if not isspmatrix_csr(G):
        raise TypeError('expected csr_matrix')
    G = remove_diagonal(G)

    mis = np.empty(G.shape[0], dtype='intc')
    mis[:] = -1

    fn = amg_core.maximal_independent_set_parallel

    if maxiter is None:
        fn(G.shape[0], G.indptr, G.indices, -1, 1, 0, mis, weights, -1)
    else:
        if maxiter < 0:
            raise ValueError('maxiter must be >= 0')

        fn(G.shape[0], G.indptr, G.indices, -1, 1, 0, mis, weights, maxiter)

    return mis


# internal function
def preprocess(S, coloring_method=None):
    """Common preprocess for splitting functions

    Parameters
    ----------
    S : csr_matrix
        Strength of connection matrix
    method : {string}
        Algorithm used to compute the vertex coloring:
            * 'MIS' - Maximal Independent Set
            * 'JP'  - Jones-Plassmann (parallel)
            * 'LDF' - Largest-Degree-First (parallel)

    Returns
    -------
    weights: ndarray
        Weights from a graph coloring of G
    S : csr_matrix
        Strength matrix with ones
    T : csr_matrix
        transpose of S
    G : csr_matrix
        union of S and T

    Notes
    -----
    Performs the following operations:
        - Checks input strength of connection matrix S
        - Replaces S.data with ones
        - Creates T = S.T in CSR format
        - Creates G = S union T in CSR format
        - Creates random weights
        - Augments weights with graph coloring (if use_color == True)

    """

    if not isspmatrix_csr(S):
        raise TypeError('expected csr_matrix')

    if S.shape[0] != S.shape[1]:
        raise ValueError('expected square matrix, shape=%s' % (S.shape,))

    N = S.shape[0]
    S = csr_matrix((np.ones(S.nnz, dtype='int8'), S.indices, S.indptr),
                   shape=(N, N))
    T = S.T.tocsr()  # transpose S for efficient column access

    G = S + T  # form graph (must be symmetric)
    G.data[:] = 1

    weights = np.ravel(T.sum(axis=1))  # initial weights
    # weights -= T.diagonal()          # discount self loops

    if coloring_method is None:
        weights = weights + sp.rand(len(weights))
    else:
        coloring = vertex_coloring(G, coloring_method)
        num_colors = coloring.max() + 1
        weights = weights + (sp.rand(len(weights)) + coloring)/num_colors

    return (weights, G, S, T)
