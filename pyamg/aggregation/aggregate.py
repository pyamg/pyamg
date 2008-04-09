"""Aggregation methods"""

__docformat__ = "restructuredtext en"

from numpy import arange, ones, empty
from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_csr

from pyamg import multigridtools

__all__ = ['standard_aggregation']

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
    >>> from pyamg import poisson
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

    """

    if not isspmatrix_csr(C): 
        raise TypeError('expected csr_matrix') 

    if C.shape[0] != C.shape[1]:
        raise ValueError('expected square matrix')

    index_type = C.indptr.dtype
    num_rows   = C.shape[0]

    Tj = empty(num_rows, dtype=index_type) #stores the aggregate #s
    
    fn = multigridtools.standard_aggregation

    num_aggregates = fn(num_rows, C.indptr, C.indices, Tj)

    if num_aggregates == 0:
        return csr_matrix( (num_rows,1), dtype='int8' ) # all zero matrix
    else:
        shape = (num_rows, num_aggregates)
        if Tj.min() == -1:
            # some nodes not aggregated
            mask = Tj != -1
            row  = arange( num_rows, dtype=index_type )[mask]
            col  = Tj[mask]
            data = ones(len(col), dtype='int8')
            return coo_matrix( (data,(row,col)), shape=shape).tocsr()
        else:
            # all nodes aggregated
            Tp = arange( num_rows+1, dtype=index_type)
            Tx = ones( len(Tj), dtype='int8')
            return csr_matrix( (Tx,Tj,Tp), shape=shape)


from pyamg.utils import dispatcher
name_to_handle = dict([ (fn[:-len('_aggregation')], eval(fn)) for fn in __all__]) 
dispatch = dispatcher( name_to_handle )

