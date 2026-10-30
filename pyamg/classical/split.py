"""Functions to compute C/F splittings for use in Classical AMG.

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
    - Produces good C/F splittings.
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

PMISR: PMIS for Reduction
    - Variant of PMIS that selects the F-set as the independent set with the
      smallest measure, keeping off-diagonal entries in A_ff small.
    - Intended for reduction-based AMG (e.g., AIR) on nonsymmetric problems.
    - See Reference [5]

PMISR-DDC: PMISR with Diagonal Dominance Cleanup
    - PMISR followed by one or more cleanup passes that convert the most
      diagonally-dominated F-points to C-points.
    - Weakly enforces diagonally-dominant A_ff for reduction-based AMG.
    - See Reference [5]    

CLJP: Cleary-Luby-Jones-Plassmann
    - Parallel method with cost and complexity comparable to Ruge-Stuben.
    - Convergence can deteriorate with increasing problem
      size on structured meshes.
    - See References [1] and [2]

CLJP-c: Cleary-Luby-Jones-Plassmann in Color
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
      PMISR     yes        no      very low
      PMISR-DDC yes        no        low
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

..  [5] S. Dargaville, R. P. Smedley-Stevenson, P. N. Smith, C. C. Pain.
    "Coarsening and parallelism with reduction multigrids for hyperbolic
    Boltzmann transport"
    The International Journal of High Performance Computing Applications
    2025; 39(3):364-384.

"""
import numpy as np
from scipy.sparse import csr_array, issparse

from pyamg.graph import vertex_coloring
from pyamg import amg_core
from pyamg.util.utils import remove_diagonal


def RS(S, second_pass=False):
    """Compute a C/F splitting using Ruge-Stuben coarsening.

    Parameters
    ----------
    S : csr_array
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)
    second_pass : bool, default False
        Perform second pass of classical AMG coarsening. Can be important for
        classical AMG interpolation. Typically not done in parallel (e.g. Hypre).

    Returns
    -------
    splitting : ndarray
        Array of length of S of ones (coarse) and zeros (fine)

    See Also
    --------
    amg_core.rs_cf_splitting

    References
    ----------
    .. [1] Ruge JW, Stuben K.  "Algebraic multigrid (AMG)"
       In Multigrid Methods, McCormick SF (ed.),
       Frontiers in Applied Mathematics, vol. 3.
       SIAM: Philadelphia, PA, 1987; 73-130.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.split import RS
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = RS(S)

    """
    if not issparse(S) or S.format != 'csr':
        raise TypeError('expected csr_array')
    S = remove_diagonal(S)

    T = S.T.tocsr()  # transpose S for efficient column access
    splitting = np.empty(S.shape[0], dtype=S.indptr.dtype)
    influence = np.zeros((S.shape[0],), dtype=S.indptr.dtype)

    amg_core.rs_cf_splitting(S.shape[0],
                             S.indptr, S.indices,
                             T.indptr, T.indices,
                             influence,
                             splitting)
    if second_pass:
        amg_core.rs_cf_splitting_pass2(S.shape[0], S.indptr,
                                       S.indices, splitting)

    return splitting


def PMIS(S):
    """C/F splitting using the Parallel Modified Independent Set method.

    Parameters
    ----------
    S : csr_array
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)

    Returns
    -------
    splitting : ndarray
        Array of length of S of ones (coarse) and zeros (fine)

    See Also
    --------
    MIS

    References
    ----------
    .. [6] Hans De Sterck, Ulrike M Yang, and Jeffrey J Heys
       "Reducing complexity in parallel algebraic multigrid preconditioners"
       SIAM Journal on Matrix Analysis and Applications 2006; 27:1019-1039.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.split import PMIS
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = PMIS(S)

    """
    S = remove_diagonal(S)
    weights, G, S, T = _preprocess(S)
    del S, T

    splitting = MIS(G, weights)
    _set_dirichlet(G, splitting)

    return splitting


def PMISc(S, method='JP'):
    """C/F splitting using Parallel Modified Independent Set (in color).

    PMIS-c, or PMIS in color, improves PMIS by perturbing the initial
    random weights with weights determined by a vertex coloring.

    Parameters
    ----------
    S : csr_array
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

    See Also
    --------
    MIS

    References
    ----------
    .. [7] David M. Alber and Luke N. Olson
       "Parallel coarse-grid selection"
       Numerical Linear Algebra with Applications 2007; 14:611-643.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.split import PMISc
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = PMISc(S)

    """
    S = remove_diagonal(S)
    weights, G, S, T = _preprocess(S, coloring_method=method)
    del S, T
    return MIS(G, weights)


def PMISR(S):
    """C/F splitting using PMIS for Reduction (PMISR).

    Computes an independent set in S + S^T using a Luby-type parallel
    algorithm, but selects the F-points as the independent set with the
    smallest measure. This keeps off-diagonal entries in A_ff small for
    nonsymmetric problems and is intended for use with reduction-based
    AMG (e.g., AIR).

    Parameters
    ----------
    S : csr_array
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)

    Returns
    -------
    splitting : ndarray
        Array of length of S of ones (coarse) and zeros (fine)

    See Also
    --------
    PMIS, MIS

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.split import PMISR
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = PMISR(S)

    """
    S = remove_diagonal(S)
    weights, G, S, T = _preprocess(S)
    del S, T
    # Negate weights so the Luby-type MIS selects local minima of the
    # original measure. Those vertices become the F-set.
    f_set = MIS(G, -weights)
    splitting = (1 - f_set).astype('intc')
    _set_dirichlet(G, splitting)
    return splitting


def PMISR_DDC(S, A, fraction_swap=0.1, ddc_its=1):
    """C/F splitting using PMISR with Diagonal Dominance Cleanup (DDC).

    First runs PMISR, then performs one or more cleanup passes that convert
    the most diagonally-dominated F-points to C-points. For each F-row, the
    row-wise diagonal dominance ratio is

        theta_i = sum_{j in F, j != i} |A[i, j]| / |A[i, i]|.

    F-points with theta above a threshold are swapped to C-points. This
    gives diagonally dominant A_ff which gives good convergence for
    reduction-based AMG (e.g., AIR) on nonsymmetric problems.

    Parameters
    ----------
    S : csr_array
        Strength of connection matrix.
    A : csr_array
        System matrix used to compute diagonal dominance ratios.
    fraction_swap : float, default 0.1
        If positive, the local fraction of F-points to convert to C-points
        based on diagonal dominance (the threshold is found by binning
        theta values into 1000 bins on [0, 1]).
        If negative, any F-row with diagonal dominance ratio greater than
        or equal to ``-fraction_swap`` is converted to a C-point.
    ddc_its : int, default 1
        Number of DDC cleanup passes to perform. Each pass recomputes
        theta against the current F-set and swaps F-points to C-points
        according to ``fraction_swap``.

    Returns
    -------
    splitting : ndarray
        Array of length of S of ones (coarse) and zeros (fine).

    See Also
    --------
    PMIS, PMISR, MIS

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.split import PMISR_DDC
    >>> A = poisson((7,), format='csr')
    >>> S = A.copy()
    >>> splitting = PMISR_DDC(S, A)

    """
    if not issparse(A) or A.format != 'csr':
        raise TypeError('expected csr_array for A')
    if A.shape != S.shape:
        raise ValueError(f'A and S must have the same shape, '
                         f'got {A.shape} and {S.shape}')
    if ddc_its < 0:
        raise ValueError('ddc_its must be >= 0')

    splitting = PMISR(S)

    A_abs = abs(A).tocsr()
    diag = np.abs(A.diagonal())

    for _ in range(ddc_its):
        # Compute diagonal dominance ratio for F points
        # Row-restricted F-column sum |A[i,:]| @ f_mask, minus the diagonal
        # contribution to enforce j != i, then divide by |A[i,i]|.
        f_mask = (splitting == 0)
        f_mask_f = f_mask.astype(A_abs.dtype)

        masked_sums = A_abs.dot(f_mask_f) - diag * f_mask_f

        theta = np.zeros(A.shape[0])
        valid = f_mask & (diag > 0)
        theta[valid] = masked_sums[valid] / diag[valid]

        F_idx = np.where(f_mask)[0]
        num_F = F_idx.size
        if num_F == 0:
            break

        F_theta = theta[F_idx]

        if fraction_swap < 0:
            swap_threshold = -fraction_swap
        else:
            target = int(num_F * fraction_swap)
            if target <= 0:
                break

            # Bin theta on [0, 1] (overflow lumped into the top bin) and walk
            # bins from the top down to find the lowest bin needed to cover
            # the worst `target` F-points; that bin's lower edge is the threshold.
            n_bins = 1000
            bin_idx = np.floor(F_theta * n_bins).astype(np.int64)
            bin_idx = np.where((bin_idx < 0) | (bin_idx >= n_bins),
                               n_bins - 1, bin_idx)
            counts = np.bincount(bin_idx, minlength=n_bins)

            cumsum_top = np.cumsum(counts[::-1])
            idx = np.searchsorted(cumsum_top, target)
            bin_boundary = n_bins - int(idx)
            swap_threshold = (bin_boundary - 1) / n_bins

        # Skip theta == 0 rows
        swap_mask = (F_theta != 0) & (F_theta >= swap_threshold)
        if not swap_mask.any():
            break
        # Swap the worst F points to C
        splitting[F_idx[swap_mask]] = 1

    return splitting


def CLJP(S, color=False):
    """Compute a C/F splitting using the parallel CLJP algorithm.

    Parameters
    ----------
    S : csr_array
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)
    color : bool
        use the CLJP coloring approach

    Returns
    -------
    splitting : array
        Array of length of S of ones (coarse) and zeros (fine)

    See Also
    --------
    MIS, PMIS, CLJPc

    References
    ----------
    .. [8] David M. Alber and Luke N. Olson
       "Parallel coarse-grid selection"
       Numerical Linear Algebra with Applications 2007; 14:611-643.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.split import CLJP
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = CLJP(S)

    """
    if not issparse(S) or S.format != 'csr':
        raise TypeError('expected csr_array')
    S = remove_diagonal(S)

    colorid = 0
    if color:
        colorid = 1

    T = S.T.tocsr()  # transpose S for efficient column access
    splitting = np.empty(S.shape[0], dtype=S.indptr.dtype)

    amg_core.cljp_naive_splitting(S.shape[0],
                                  S.indptr, S.indices,
                                  T.indptr, T.indices,
                                  splitting,
                                  colorid)

    return splitting


def CLJPc(S):
    """Compute a C/F splitting using the parallel CLJP-c algorithm.

    CLJP-c, or CLJP in color, improves CLJP by perturbing the initial
    random weights with weights determined by a vertex coloring.

    Parameters
    ----------
    S : csr_array
        Strength of connection matrix indicating the strength between nodes i
        and j (S_ij)

    Returns
    -------
    splitting : array
        Array of length of S of ones (coarse) and zeros (fine)

    See Also
    --------
    MIS, PMIS, CLJP

    References
    ----------
    .. [1] David M. Alber and Luke N. Olson
       "Parallel coarse-grid selection"
       Numerical Linear Algebra with Applications 2007; 14:611-643.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.split import CLJPc
    >>> S = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> splitting = CLJPc(S)

    """
    S = remove_diagonal(S)
    return CLJP(S, color=True)


def MIS(G, weights, maxiter=None):
    """Compute a maximal independent set of a graph in parallel.

    Parameters
    ----------
    G : csr_array
        Matrix graph, G[i,j] != 0 indicates an edge
    weights : ndarray
        Array of weights for each vertex in the graph G
    maxiter : int
        Maximum number of iterations (default: None)

    Returns
    -------
    mis : array
        Array of length of G of zeros/ones indicating the independent set

    See Also
    --------
    fn = amg_core.maximal_independent_set_parallel

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.classical.split import MIS
    >>> import numpy as np
    >>> G = poisson((7,), format='csr') # 1D mesh with 7 vertices
    >>> w = np.ones((G.shape[0],1)).ravel()
    >>> mis = MIS(G,w)

    """
    if not issparse(G) or G.format != 'csr':
        raise TypeError('expected csr_array')
    G = remove_diagonal(G)

    mis = np.empty(G.shape[0], dtype=G.indptr.dtype)
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
def _preprocess(S, coloring_method=None):
    """Preprocess splitting functions.

    Parameters
    ----------
    S : csr_array
        Strength of connection matrix
    coloring_method : string
        Algorithm used to compute the vertex coloring:
            * 'MIS' - Maximal Independent Set
            * 'JP'  - Jones-Plassmann (parallel)
            * 'LDF' - Largest-Degree-First (parallel)

    Returns
    -------
    weights: ndarray
        Weights from a graph coloring of G
    S : csr_array
        Strength matrix with ones
    T : csr_array
        transpose of S
    G : csr_array
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
    if not issparse(S) or S.format != 'csr':
        raise TypeError('expected csr_array')

    if S.shape[0] != S.shape[1]:
        raise ValueError(f'expected square matrix, shape={S.shape}')

    N = S.shape[0]
    S = csr_array((np.ones(S.nnz, dtype=np.int32), S.indices, S.indptr),
                   shape=(N, N))
    T = S.T.tocsr()  # transpose S for efficient column access

    G = S + T  # form graph (must be symmetric)
    G.data[:] = 1

    weights = np.ravel(T.sum(axis=1))  # initial weights
    # weights -= T.diagonal()          # discount self loops

    if coloring_method is None:
        weights = weights + np.random.rand(len(weights))
    else:
        coloring = vertex_coloring(G, coloring_method)
        num_colors = coloring.max() + 1
        weights = (weights + (np.random.rand(len(weights)) + coloring)
                   / num_colors)

    return (weights, G, S, T)


def _set_dirichlet(G, splitting):
    # For rows of zero-length (Dirichlet), set as Fine.
    I = np.where(np.diff(G.indptr) == 0)[0]
    splitting[I] = 0
