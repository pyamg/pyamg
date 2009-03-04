"""Aggregation methods"""

__docformat__ = "restructuredtext en"

import numpy
import scipy
from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_csr, isspmatrix_csc
from pyamg import amg_core
from pyamg.graph import lloyd_cluster

__all__ = ['standard_aggregation','lloyd_aggregation']

def standard_aggregation(C):
    """Compute the sparsity pattern of the tentative prolongator

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern 
        of the tentative prolongator

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import standard_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.todense()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> standard_aggregation(A).todense() # two aggregates
    matrix([[1, 0],
            [1, 0],
            [0, 1],
            [0, 1]], dtype=int8)
    >>> A = csr_matrix([[1,0,0],[0,1,1],[0,1,1]])
    >>> A.todense()                      # first vertex is isolated
    matrix([[1, 0, 0],
            [0, 1, 1],
            [0, 1, 1]])
    >>> standard_aggregation(A).todense() # one aggregate
    matrix([[0],
            [1],
            [1]], dtype=int8)

    See Also
    --------
    amg_core.standard_aggregation

    """

    if not isspmatrix_csr(C): 
        raise TypeError('expected csr_matrix') 

    if C.shape[0] != C.shape[1]:
        raise ValueError('expected square matrix')

    index_type = C.indptr.dtype
    num_rows   = C.shape[0]

    Tj = numpy.empty(num_rows, dtype=index_type) #stores the aggregate #s
    
    fn = amg_core.standard_aggregation

    num_aggregates = fn(num_rows, C.indptr, C.indices, Tj)

    if num_aggregates == 0:
        return csr_matrix( (num_rows,1), dtype='int8' ) # all zero matrix
    else:
        shape = (num_rows, num_aggregates)
        if Tj.min() == -1:
            # some nodes not aggregated
            mask = Tj != -1
            row  = numpy.arange( num_rows, dtype=index_type )[mask]
            col  = Tj[mask]
            data = numpy.ones(len(col), dtype='int8')
            return coo_matrix( (data,(row,col)), shape=shape).tocsr()
        else:
            # all nodes aggregated
            Tp = numpy.arange( num_rows+1, dtype=index_type)
            Tx = numpy.ones( len(Tj), dtype='int8')
            return csr_matrix( (Tx,Tj,Tp), shape=shape)


def lloyd_aggregation(C, ratio=0.03, distance='unit', maxiter=10):
    """Aggregated nodes using Lloyd Clustering

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix
    ratio : scalar
        Fraction of the nodes which will be seeds.
    distance : ['unit','abs','inv',None]
        Distance assigned to each edge of the graph G used in Lloyd clustering

        For each nonzero value C[i,j]:

        =======  ===========================
        'unit'   G[i,j] = 1 
        'abs'    G[i,j] = abs(C[i,j])
        'inv'    G[i,j] = 1.0/abs(C[i,j])
        'same'   G[i,j] = C[i,j]
        'sub'    G[i,j] = C[i,j] - min(C)
        =======  ===========================

    maxiter : int
        Maximum number of iterations to perform

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern 
        of the tentative prolongator

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import lloyd_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.todense()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> lloyd_aggregation(A).todense() # one aggregate
    matrix([[1],
            [1],
            [1],
            [1]], dtype=int8)
    >>> lloyd_aggregation(A,ratio=0.5).todense() # more seeding for two aggregates
    matrix([[0, 1],
            [0, 1],
            [1, 0],
            [1, 0]], dtype=int8)

    See Also
    --------
    amg_core.standard_aggregation

    """

    if ratio <= 0 or ratio > 1:
        raise ValueError('ratio must be > 0.0 and <= 1.0')

    if not (isspmatrix_csr(C) or isspmatrix_csc(C)):
        raise TypeError('expected csr_matrix or csc_matrix')

    if distance == 'unit':
        data = numpy.ones_like(C.data).astype(float)
    elif distance == 'abs':
        data = abs(C.data)
    elif distance == 'inv':
        data = 1.0/abs(C.data)
    elif distance is 'same':
        data = C.data
    elif distance is 'min':
        data = C.data - C.data.min()
    else:
        raise ValueError('unrecognized value distance=%s' % distance)
    
    if C.dtype == complex:
        data = scipy.real(data)

    assert(data.min() >= 0)

    G = C.__class__((data,C.indices,C.indptr),shape=C.shape)

    num_seeds = int(min( max(ratio*G.shape[0],1), G.shape[0] ))

    distances, clusters, seeds = lloyd_cluster( G, num_seeds, maxiter=maxiter )

    row  = (clusters >= 0).nonzero()[0]
    col  = clusters[row]
    data = numpy.ones(len(row), dtype='int8')
    AggOp = coo_matrix( (data,(row,col)), shape=(G.shape[0],num_seeds)).tocsr()
    return AggOp
