"""Aggregation methods."""


from warnings import warn
import numpy as np
from scipy import sparse
from .. import amg_core
from ..graph import lloyd_cluster
from ..strength import classical_strength_of_connection


def standard_aggregation(C):
    """Compute the sparsity pattern of the tentative prolongator.

    Parameters
    ----------
    C : csr_array
        Strength of connection matrix.

    Returns
    -------
    csr_array
        Aggregation operator which determines the sparsity pattern
        of the tentative prolongator.
    array
        Array of Cpts, i.e., Cpts[i] = root node of aggregate i.

    See Also
    --------
    amg_core.standard_aggregation

    Examples
    --------
    >>> from scipy.sparse import csr_array
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import standard_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.toarray()
    array([[ 2., -1.,  0.,  0.],
           [-1.,  2., -1.,  0.],
           [ 0., -1.,  2., -1.],
           [ 0.,  0., -1.,  2.]])
    >>> standard_aggregation(A)[0].toarray() # two aggregates
    array([[1, 0],
           [1, 0],
           [0, 1],
           [0, 1]], dtype=int8)
    >>> A = csr_array([[1,0,0],[0,1,1],[0,1,1]])
    >>> A.toarray()                      # first vertex is isolated
    array([[1, 0, 0],
           [0, 1, 1],
           [0, 1, 1]])
    >>> standard_aggregation(A)[0].toarray() # one aggregate
    array([[0],
           [1],
           [1]], dtype=int8)

    """
    if not sparse.issparse(C) or C.format != 'csr':
        raise TypeError('expected csr_array')

    if C.shape[0] != C.shape[1]:
        raise ValueError('expected square matrix')

    index_type = C.indptr.dtype
    num_rows = C.shape[0]

    Tj = np.empty(num_rows, dtype=index_type)  # stores the aggregate #s
    Cpts = np.empty(num_rows, dtype=index_type)  # stores the Cpts

    fn = amg_core.standard_aggregation

    num_aggregates = fn(num_rows, C.indptr, C.indices, Tj, Cpts)
    Cpts = Cpts[:num_aggregates]

    # no nodes aggregated
    if num_aggregates == 0:
        # return all zero matrix and no Cpts
        return sparse.csr_array((num_rows, 1), dtype='int8'), \
            np.array([], dtype=index_type)

    shape = (num_rows, num_aggregates)

    # some nodes not aggregated
    if Tj.min() == -1:
        mask = Tj != -1
        row = np.arange(num_rows, dtype=index_type)[mask]
        col = Tj[mask]
        data = np.ones(len(col), dtype='int8')
        return sparse.coo_array((data, (row, col)), shape=shape).tocsr(), Cpts

    # all nodes aggregated
    Tp = np.arange(num_rows+1, dtype=index_type)
    Tx = np.ones(len(Tj), dtype='int8')
    return sparse.csr_array((Tx, Tj, Tp), shape=shape), Cpts


def naive_aggregation(C):
    """Compute the sparsity pattern of the tentative prolongator.

    Parameters
    ----------
    C : csr_array
        Strength of connection matrix.

    Returns
    -------
    csr_array
        Aggregation operator which determines the sparsity pattern
        of the tentative prolongator.
    array
        Array of Cpts, i.e., Cpts[i] = root node of aggregate i.

    See Also
    --------
    amg_core.naive_aggregation

    Notes
    -----
    Differs from standard aggregation.  Each dof is considered.  If it has been
    aggregated, skip over.  Otherwise, put dof and any unaggregated neighbors
    in an aggregate.  Results in possibly much higher complexities than
    standard aggregation.

    Examples
    --------
    >>> from scipy.sparse import csr_array
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import naive_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.toarray()
    array([[ 2., -1.,  0.,  0.],
           [-1.,  2., -1.,  0.],
           [ 0., -1.,  2., -1.],
           [ 0.,  0., -1.,  2.]])
    >>> naive_aggregation(A)[0].toarray() # two aggregates
    array([[1, 0],
           [1, 0],
           [0, 1],
           [0, 1]], dtype=int8)
    >>> A = csr_array([[1,0,0],[0,1,1],[0,1,1]])
    >>> A.toarray()                      # first vertex is isolated
    array([[1, 0, 0],
           [0, 1, 1],
           [0, 1, 1]])
    >>> naive_aggregation(A)[0].toarray() # two aggregates
    array([[1, 0],
           [0, 1],
           [0, 1]], dtype=int8)

    """
    if not sparse.issparse(C) or C.format != 'csr':
        raise TypeError('expected csr_array')

    if C.shape[0] != C.shape[1]:
        raise ValueError('expected square matrix')

    index_type = C.indptr.dtype
    num_rows = C.shape[0]

    Tj = np.empty(num_rows, dtype=index_type)  # stores the aggregate #s
    Cpts = np.empty(num_rows, dtype=index_type)  # stores the Cpts

    fn = amg_core.naive_aggregation

    num_aggregates = fn(num_rows, C.indptr, C.indices, Tj, Cpts)
    Cpts = Cpts[:num_aggregates]
    Tj = Tj - 1

    if num_aggregates == 0:
        # all zero matrix
        return sparse.csr_array((num_rows, 1), dtype='int8'), Cpts

    shape = (num_rows, num_aggregates)
    # all nodes aggregated
    Tp = np.arange(num_rows+1, dtype=index_type)
    Tx = np.ones(len(Tj), dtype='int8')
    return sparse.csr_array((Tx, Tj, Tp), shape=shape), Cpts


def pairwise_aggregation(A, matchings=2, theta=0.25,
                         norm='min', compute_P=False):
    """Compute the sparsity pattern of the tentative prolongator.

    Parameters
    ----------
    A : csr_array or bsr_array
        level matrix
    matchings : int, default 2
        number of times to perform pairwise aggregation; each
        matching increases coarsening factor by about two.
    theta : float, default 0.25
        Strength tolerance used in computing classical SOC.
    norm : string, default 'min'
        Norm type used in computing classical SOC.
    compute_P : bool; default False
        Compute pairwise interpolation directly; if False, return
        integer aggregation matrix for smoothed_aggregation_solver.
        If True, return float interpolation P, converting to BSR
        form with identity of size bsize x bsize on each aggregate
        if A is BSR.

    Examples
    --------
    >>> from scipy.sparse import csr_array
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import pairwise_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.toarray()
    array([[ 2., -1.,  0.,  0.],
           [-1.,  2., -1.,  0.],
           [ 0., -1.,  2., -1.],
           [ 0.,  0., -1.,  2.]])
    >>> pairwise_aggregation(A, matchings=1)[0].toarray() # two aggregates
    array([[1, 0],
           [1, 0],
           [0, 1],
           [0, 1]], dtype=int8)
    >>> pairwise_aggregation(A, matchings=2)[0].toarray() # one aggregate
    array([[1],
           [1],
           [1],
           [1]], dtype=int8)

    See Also
    --------
    amg_core.pairwise_aggregation

    References
    ----------
    [0] Notay, Y. (2010). An aggregation-based algebraic multigrid
    method. Electronic transactions on numerical analysis, 37(6),
    123-146.

    """
    # Get SOC matrix
    if not sparse.issparse(A) or A.format not in ('bsr', 'csr'):
        try:
            A = A.tocsr()
            warn('Implicit conversion of A to csr', sparse.SparseEfficiencyWarning)
        except Exception as e:
            raise TypeError('Invalid matrix type, must be CSR or BSR.') from e

    index_type = A.indptr.dtype
    Ac = A      # Let Ac reference A for loop purposes
    T = None
    Cpts = None

    # Loop over the number of pairwise matchings to be done
    for i in range(0, matchings):

        # Compute SOC matrix for this matching
        if sparse.issparse(A) and A.format == 'bsr':
            C = classical_strength_of_connection(A=Ac, theta=theta, block=True, norm=norm)
        else:
            C = classical_strength_of_connection(A=Ac, theta=theta, block=False, norm=norm)

        # Form pairwise aggregation matrix
        num_rows = C.shape[0]
        Tj = np.empty(num_rows, dtype=index_type)  # stores the aggregate #s
        new_cpts = np.empty(num_rows, dtype=index_type)  # stores the new_cpts
        fn = amg_core.pairwise_aggregation
        num_aggregates = fn(num_rows, C.indptr, C.indices, C.data, Tj, new_cpts)
        if Cpts is None:
            Cpts = new_cpts[:num_aggregates]
        else:
            Cpts = Cpts[new_cpts[:num_aggregates]]
        Tj = Tj - 1

        # Construct sparse T
        if num_aggregates == 0:
            # all zero matrix
            T_temp = sparse.csr_array((num_rows, 1), dtype='int8')
            warn('No pairwise aggregates found, T = 0.')
        else:
            shape = (num_rows, num_aggregates)
            Tp = np.arange(num_rows+1, dtype=index_type)
            # If A is not BSR
            if not sparse.issparse(A) or A.format != 'bsr':
                Tx = np.ones(len(Tj), dtype='int8')
                T_temp = sparse.csr_array((Tx, Tj, Tp), shape=shape)
            else:
                shape = (shape[0]*A.blocksize[0], shape[1]*A.blocksize[1])
                Tx = np.array(len(Tj)*[np.identity(A.blocksize[0])], dtype='int8')
                T_temp = sparse.bsr_array((Tx, Tj, Tp), blocksize=A.blocksize, shape=shape)

        # Form aggregation matrix, need to make sure is CSR/BSR
        if i == 0:
            T = T_temp
        elif sparse.issparse(A) and A.format == 'bsr':
            T = sparse.bsr_array(T @ T_temp)
        else:
            T = sparse.csr_array(T @ T_temp)

        # Break loop if zero aggregates were found
        if num_aggregates == 0:
            break

        # Form coarse grid operator for next matching
        if i < (matchings-1):
            if sparse.issparse(T_temp) and T_temp.format == 'csr':
                Ac = T_temp.T.tocsr() @ Ac @ T_temp
            else:
                Ac = T_temp.T @ Ac @ T_temp

    # Convert T to dtype int if only used for aggregation
    if compute_P:
        T = T.astype(A.dtype, copy=False)

    return T, Cpts


def lloyd_aggregation(C, ratio=0.03, distance='unit', maxiter=10):
    """Aggregate nodes using Lloyd Clustering.

    Parameters
    ----------
    C : csr_array
        Strength of connection matrix.
    ratio : scalar
        Fraction of the nodes which will be seeds.
    distance : ['unit','abs','inv',None]
        Distance assigned to each edge of the graph G used in Lloyd clustering.

        For each nonzero value C[i,j]::

            =======  ===========================
            'unit'   G[i,j] = 1
            'abs'    G[i,j] = abs(C[i,j])
            'inv'    G[i,j] = 1.0/abs(C[i,j])
            'same'   G[i,j] = C[i,j]
            'sub'    G[i,j] = C[i,j] - min(C)
            =======  ===========================

    maxiter : int
        Maximum number of iterations to perform.

    Returns
    -------
    csr_array
        Aggregation operator which determines the sparsity pattern
        of the tentative prolongator.
    array
        Array of Cpts, i.e., Cpts[i] = root node of aggregate i.

    See Also
    --------
    amg_core.standard_aggregation

    Examples
    --------
    >>> from scipy.sparse import csr_array
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import lloyd_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.toarray()
    array([[ 2., -1.,  0.,  0.],
           [-1.,  2., -1.,  0.],
           [ 0., -1.,  2., -1.],
           [ 0.,  0., -1.,  2.]])
    >>> lloyd_aggregation(A)[0].toarray() # one aggregate
    array([[1],
           [1],
           [1],
           [1]], dtype=int8)
    >>> # more seeding for two aggregates
    >>> Agg = lloyd_aggregation(A,ratio=0.5)[0].toarray()

    """
    if ratio <= 0 or ratio > 1:
        raise ValueError('ratio must be > 0.0 and <= 1.0')

    if not sparse.issparse(C) or C.format not in ('csc', 'csr'):
        raise TypeError('expected csr_array or csc_array')

    if distance == 'unit':
        data = np.ones_like(C.data).astype(float)
    elif distance == 'abs':
        data = abs(C.data)
    elif distance == 'inv':
        data = 1.0/abs(C.data)
    elif distance == 'same':
        data = C.data
    elif distance == 'min':
        data = C.data - C.data.min()
    else:
        raise ValueError(f'Unrecognized value distance={distance}')

    if C.dtype == complex:
        data = np.real(data)

    assert data.min() >= 0

    G = C.__class__((data, C.indices, C.indptr), shape=C.shape)

    num_seeds = int(min(max(ratio * G.shape[0], 1), G.shape[0]))

    _, clusters, seeds = lloyd_cluster(G, num_seeds, maxiter=maxiter)

    row = (clusters >= 0).nonzero()[0].astype(C.indices.dtype)
    col = clusters[row]
    data = np.ones(len(row), dtype='int8')
    AggOp = sparse.coo_array((data, (row, col)),
                              shape=(G.shape[0], num_seeds)).tocsr()
    return AggOp, seeds


def balanced_lloyd_aggregation(C, num_clusters=None):
    """Aggregate nodes using Balanced Lloyd Clustering.

    Parameters
    ----------
    C : csr_array
        Strength of connection matrix with positive weights.
    num_clusters : int
        Number of seeds or clusters expected (default: C.shape[0] / 10).

    Returns
    -------
    csr_array
        Aggregation operator which determines the sparsity pattern
        of the tentative prolongator.
    array
        Array of Cpts, i.e., Cpts[i] = root node of aggregate i.

    See Also
    --------
    amg_core.standard_aggregation

    Examples
    --------
    >>> import numpy as np
    >>> import pyamg
    >>> from pyamg.aggregation.aggregate import balanced_lloyd_aggregation
    >>> data = pyamg.gallery.load_example('unit_square')
    >>> G = data['A'].tocsr()
    >>> xy = data['vertices'][:,:2]
    >>> G.data[:] = np.ones(len(G.data))
    >>> np.random.seed(787888)
    >>> AggOp, seeds = balanced_lloyd_aggregation(G)

    """
    if num_clusters is None:
        num_clusters = int(C.shape[0] / 10)

    if num_clusters < 1 or num_clusters > C.shape[0]:
        raise ValueError('num_clusters must be between 1 and n')

    if not sparse.issparse(C) or C.format not in ('csc', 'csr'):
        raise TypeError('expected csr_array or csc_array')

    if C.data.min() <= 0:
        raise ValueError('positive edge weights required')

    if C.dtype == complex:
        data = np.real(C.data)
    else:
        data = C.data

    G = C.__class__((data, C.indices, C.indptr), shape=C.shape)
    num_nodes = G.shape[0]

    seeds = np.random.permutation(num_nodes)[:num_clusters]
    seeds = seeds.astype(np.int32)
    mv = np.finfo(G.dtype).max
    d = mv * np.ones(num_nodes, dtype=G.dtype)
    d[seeds] = 0

    cm = -1 * np.ones(num_nodes, dtype=np.int32)
    cm[seeds] = seeds

    amg_core.lloyd_cluster_exact(num_nodes,
                                 G.indptr, G.indices, G.data,
                                 num_clusters,
                                 d, cm, seeds)

    col = cm
    row = np.arange(len(cm), dtype=np.int32)
    data = np.ones(len(row), dtype=np.int32)
    AggOp = sparse.coo_array((data, (row, col)),
                              shape=(G.shape[0], num_clusters)).tocsr()
    return AggOp, seeds
