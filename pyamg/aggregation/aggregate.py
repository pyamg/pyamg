"""Aggregation methods."""


import warnings
import numpy as np
from scipy import sparse
from .. import amg_core
from ..graph import lloyd_cluster, balanced_lloyd_cluster


def standard_aggregation(C):
    """Compute the sparsity pattern of the tentative prolongator.

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator
    Cpts : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import standard_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.toarray()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> standard_aggregation(A)[0].toarray() # two aggregates
    matrix([[1, 0],
            [1, 0],
            [0, 1],
            [0, 1]], dtype=int8)
    >>> A = csr_matrix([[1,0,0],[0,1,1],[0,1,1]])
    >>> A.toarray()                      # first vertex is isolated
    matrix([[1, 0, 0],
            [0, 1, 1],
            [0, 1, 1]])
    >>> standard_aggregation(A)[0].toarray() # one aggregate
    matrix([[0],
            [1],
            [1]], dtype=int8)

    See Also
    --------
    amg_core.standard_aggregation

    """
    if not sparse.isspmatrix_csr(C):
        raise TypeError('expected csr_matrix')

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
        return sparse.csr_matrix((num_rows, 1), dtype='int8'),\
            np.array([], dtype=index_type)

    shape = (num_rows, num_aggregates)

    # some nodes not aggregated
    if Tj.min() == -1:
        mask = Tj != -1
        row = np.arange(num_rows, dtype=index_type)[mask]
        col = Tj[mask]
        data = np.ones(len(col), dtype='int8')
        return sparse.coo_matrix((data, (row, col)), shape=shape).tocsr(), Cpts

    # all nodes aggregated
    Tp = np.arange(num_rows+1, dtype=index_type)
    Tx = np.ones(len(Tj), dtype='int8')
    return sparse.csr_matrix((Tx, Tj, Tp), shape=shape), Cpts


def naive_aggregation(C):
    """Compute the sparsity pattern of the tentative prolongator.

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator
    Cpts : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import naive_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.toarray()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> naive_aggregation(A)[0].toarray() # two aggregates
    matrix([[1, 0],
            [1, 0],
            [0, 1],
            [0, 1]], dtype=int8)
    >>> A = csr_matrix([[1,0,0],[0,1,1],[0,1,1]])
    >>> A.toarray()                      # first vertex is isolated
    matrix([[1, 0, 0],
            [0, 1, 1],
            [0, 1, 1]])
    >>> naive_aggregation(A)[0].toarray() # two aggregates
    matrix([[1, 0],
            [0, 1],
            [0, 1]], dtype=int8)

    See Also
    --------
    amg_core.naive_aggregation

    Notes
    -----
    Differs from standard aggregation.  Each dof is considered.  If it has been
    aggregated, skip over.  Otherwise, put dof and any unaggregated neighbors
    in an aggregate.  Results in possibly much higher complexities than
    standard aggregation.

    """
    if not sparse.isspmatrix_csr(C):
        raise TypeError('expected csr_matrix')

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
        return sparse.csr_matrix((num_rows, 1), dtype='int8'), Cpts

    shape = (num_rows, num_aggregates)
    # all nodes aggregated
    Tp = np.arange(num_rows+1, dtype=index_type)
    Tx = np.ones(len(Tj), dtype='int8')
    return sparse.csr_matrix((Tx, Tj, Tp), shape=shape), Cpts


def lloyd_aggregation(C, ratio=0.1, measure=None, maxiter=5):
    """Aggregate nodes using Lloyd Clustering.

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix
    ratio : scalar
        Fraction of nodes to be aggregate (centers).  ratio=0.1 is
        a coarsening by 10
    measure : ['unit','abs','inv',None]
        Distance measure to use and assigned to each edge graph.

        For each nonzero value C[i,j]:
        =======  ===========================
        None     G[i,j] = C[i,j]
        'abs'    G[i,j] = abs(C[i,j])
        'inv'    G[i,j] = 1.0/abs(C[i,j])
        'inv+'    G[i,j] = 1.0/abs(C[i,j])
        'unit'   G[i,j] = 1
        'sub'    G[i,j] = C[i,j] - min(C)
        =======  ===========================
    maxiter : int
        Maximum number of iterations to perform

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator.  Node i is in cluster j if AggOp[i,j] = 1.
    centers : array
        array of centers or Cpts, i.e., Cpts[i] = root node of aggregate i

    See Also
    --------
    amg_core.standard_aggregation

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import lloyd_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.toarray()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> lloyd_aggregation(A)[0].toarray() # one aggregate
    matrix([[1],
            [1],
            [1],
            [1]], dtype=int8)

    """
    C = sparse.csr_matrix(C)

    if C.shape[0] != C.shape[1]:
        raise ValueError('graph should be a square matrix.')

    n = C.shape[0]

    if ratio <= 0 or ratio > 1:
        raise ValueError('ratio must be > 0.0 and <= 1.0')

    naggs = int(min(max(ratio * n, 1), n))

    if measure is None:
        data = C.data
    elif measure == 'abs':
        data = np.abs(C.data)
    elif measure == 'inv':
        data = 1.0 / abs(C.data)
    elif measure == 'unit':
        data = np.ones_like(C.data).astype(float)
    elif measure == 'min':
        data = C.data - C.data.min()
    else:
        raise ValueError(f'Unrecognized value measure={measure}')

    if C.dtype == complex:
        data = np.real(data)

    if C.data.min() <= 0:
        raise ValueError('positive edge weights required')

    if len(data) > 0:
        if data.min() < 0:
            raise ValueError('Lloyd aggregation requires a positive measure.')

    G = C.__class__((data, C.indices, C.indptr), shape=C.shape)

    clusters, centers = lloyd_cluster(G, naggs, maxiter=maxiter)

    if np.any(clusters < 0):
        warnings.warn('Lloyd aggregation encountered a point that is unaggregated.')

    if clusters.min() < 0:
        warnings.warn('Lloyd clustering did not cluster every point')

    row = (clusters >= 0).nonzero()[0]
    col = clusters[row]
    data = np.ones(len(row), dtype=np.int32)
    AggOp = sparse.coo_matrix((data, (row, col)), shape=(n, naggs)).tocsr()

    return AggOp, centers


def balanced_lloyd_aggregation(C, ratio=0.1, measure=None, maxiter=5,
                               rebalance_iters=5, pad=None, A=None):
    """Aggregate nodes using Balanced Lloyd Clustering.

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix with positive weights
    ratio : scalar
        Fraction of nodes to be aggregate (centers).  ratio=0.1 is
        a coarsening by 10
    naggs : int
        Number of aggregates or clusters expected (default: C.shape[0] / 10)
    measure : ['unit','abs','inv',None]
        Distance measure to use and assigned to each edge graph.

        For each nonzero value C[i,j]:
        =======  ===========================
        None     G[i,j] = C[i,j]
        'abs'    G[i,j] = abs(C[i,j])
        'inv'    G[i,j] = 1.0/abs(C[i,j])
        'unit'   G[i,j] = 1
        'sub'    G[i,j] = C[i,j] - min(C)
        =======  ===========================
    maxiter : int
        Maximum number of iterations to perform
    rebalance_iters : int
        Number of rebalance iterations to perform in balanced Lloyd clustering
    pad : float
        A pad for the measure with the sparsity of A.
    A : csr_matrix
        Sparse matrix to pad with.

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator.  Node i is in cluster j if AggOp[i,j] = 1.
    centers : array
        array of centers or Cpts, i.e., centers[i] = root node of aggregate i

    Notes
    -----
    If pad is not None, then C will be augmented by
        C = C + E
    where E=A and E.data = pad.  The goal is to "fix up" the connectivity of G.
    As an example, if pad is small and if measure='inv' is used, then C + E
    will have "long" edges for the pad.

    See Also
    --------
    amg_core.standard_aggregation
    amg_core.balanced_lloyd_cluster

    Examples
    --------
    >>> import pyamg
    >>> from pyamg.aggregation.aggregate import balanced_lloyd_aggregation
    >>> data = pyamg.gallery.load_example('unit_square')
    >>> G = data['A']
    >>> xy = data['vertices'][:,:2]
    >>> G.data[:] = np.ones(len(G.data))

    >>> np.random.seed(787888)
    >>> AggOp, seeds = balanced_lloyd_aggregation(G)

    """
    C = sparse.csr_matrix(C)

    if C.shape[0] != C.shape[1]:
        raise ValueError('Graph should be a square matrix.')

    n = C.shape[0]

    if ratio <= 0 or ratio > 1:
        raise ValueError('ratio must be > 0.0 and <= 1.0')

    naggs = int(min(max(ratio * n, 1), n))

    if pad is not None:
        if A is None:
            raise ValueError('Matrix A is required if pad is used')

        A = sparse.csr_matrix(A)

        Epad = A.copy()
        Epad.data[:] = pad

        C += Epad

    if measure is None:
        data = C.data
    elif measure == 'abs':
        data = np.abs(C.data)
    elif measure == 'inv':
        data = 1.0 / abs(C.data)
    elif measure == 'unit':
        data = np.ones_like(C.data).astype(float)
    elif measure == 'min':
        data = C.data - C.data.min()
    else:
        raise ValueError(f'Unrecognized value measure={measure}')

    if C.dtype == complex:
        data = np.real(C.data)

    if len(data) > 0:
        if data.min() < 0:
            raise ValueError('Lloyd aggregation requires a positive measure.')

    G = C.__class__((data, C.indices, C.indptr), shape=C.shape)

    clusters, centers = balanced_lloyd_cluster(G, naggs, maxiter=maxiter,
                                               rebalance_iters=rebalance_iters)

    if np.any(clusters < 0):
        warnings.warn('Lloyd aggregation encountered a point that is unaggregated.')

    if clusters.min() < 0:
        warnings.warn('Lloyd clustering did not cluster every point')

    row = (clusters >= 0).nonzero()[0]
    col = clusters[row]
    data = np.ones(len(row), dtype=np.int32)
    AggOp = sparse.coo_matrix((data, (row, col)), shape=(n, naggs)).tocsr()

    return AggOp, centers
