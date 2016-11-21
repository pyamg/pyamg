"""Utility Functions for reading and writing individual rows
   in BSR matrices"""


import numpy as np

__all__ = ['BSR_Get_Row', 'BSR_Row_WriteScalar', 'BSR_Row_WriteVect']


def BSR_Get_Row(A, i):
    """
    Return row i in BSR matrix A.  Only nonzero entries are returned

    Parameters
    ----------
    A : bsr_matrix
        Input matrix
    i : int
        Row number

    Returns
    -------
    z : array
        Actual nonzero values for row i colindx Array of column indices for the
        nonzeros of row i

    Examples
    --------
    >>> from numpy import array
    >>> from scipy.sparse import bsr_matrix
    >>> from pyamg.util.BSR_utils import BSR_Get_Row
    >>> indptr  = array([0,2,3,6])
    >>> indices = array([0,2,2,0,1,2])
    >>> data    = array([1,2,3,4,5,6]).repeat(4).reshape(6,2,2)
    >>> B = bsr_matrix( (data,indices,indptr), shape=(6,6) )
    >>> Brow = BSR_Get_Row(B,2)
    >>> print Brow[1]
    [4 5]

    """

    blocksize = A.blocksize[0]
    BlockIndx = int(i/blocksize)
    rowstart = A.indptr[BlockIndx]
    rowend = A.indptr[BlockIndx+1]
    localRowIndx = i % blocksize

    # Get z
    indys = A.data[rowstart:rowend, localRowIndx, :].nonzero()
    z = A.data[rowstart:rowend, localRowIndx, :][indys[0], indys[1]]

    colindx = np.zeros((1, z.__len__()), dtype=np.int32)
    counter = 0

    for j in range(rowstart, rowend):
        coloffset = blocksize*A.indices[j]
        indys = A.data[j, localRowIndx, :].nonzero()[0]
        increment = indys.shape[0]
        colindx[0, counter:(counter+increment)] = coloffset + indys
        counter += increment

    return np.mat(z).T, colindx[0, :]


def BSR_Row_WriteScalar(A, i, x):
    """
    Write a scalar at each nonzero location in row i of BSR matrix A

    Parameters
    ----------
    A : bsr_matrix
        Input matrix
    i : int
        Row number
    x : float
        Scalar to overwrite nonzeros of row i in A

    Returns
    -------
    A : bsr_matrix
        All nonzeros in row i of A have been overwritten with x.
        If x is a vector, the first length(x) nonzeros in row i
        of A have been overwritten with entries from x

    Examples
    --------
    >>> from numpy import array
    >>> from scipy.sparse import bsr_matrix
    >>> from pyamg.util.BSR_utils import BSR_Row_WriteScalar
    >>> indptr  = array([0,2,3,6])
    >>> indices = array([0,2,2,0,1,2])
    >>> data    = array([1,2,3,4,5,6]).repeat(4).reshape(6,2,2)
    >>> B = bsr_matrix( (data,indices,indptr), shape=(6,6) )
    >>> BSR_Row_WriteScalar(B,5,22)
    """

    blocksize = A.blocksize[0]
    BlockIndx = int(i/blocksize)
    rowstart = A.indptr[BlockIndx]
    rowend = A.indptr[BlockIndx+1]
    localRowIndx = i % blocksize

    # for j in range(rowstart, rowend):
    #   indys = A.data[j,localRowIndx,:].nonzero()[0]
    #   increment = indys.shape[0]
    #   A.data[j,localRowIndx,indys] = x

    indys = A.data[rowstart:rowend, localRowIndx, :].nonzero()
    A.data[rowstart:rowend, localRowIndx, :][indys[0], indys[1]] = x


def BSR_Row_WriteVect(A, i, x):
    """
    Overwrite the nonzeros in row i of BSR matrix A with the vector x.
    length(x) and nnz(A[i,:]) must be equivalent.

    Parameters
    ----------
    A : bsr_matrix
        Matrix assumed to be in BSR format
    i : int
        Row number
    x : array
        Array of values to overwrite nonzeros in row i of A

    Returns
    -------
    A : bsr_matrix
        The nonzeros in row i of A have been
        overwritten with entries from x.  x must be same
        length as nonzeros of row i.  This is guaranteed
        when this routine is used with vectors derived form
        Get_BSR_Row

    Examples
    --------
    >>> from numpy import array
    >>> from scipy.sparse import bsr_matrix
    >>> from pyamg.util.BSR_utils import BSR_Row_WriteVect
    >>> indptr  = array([0,2,3,6])
    >>> indices = array([0,2,2,0,1,2])
    >>> data    = array([1,2,3,4,5,6]).repeat(4).reshape(6,2,2)
    >>> B = bsr_matrix( (data,indices,indptr), shape=(6,6) )
    >>> BSR_Row_WriteVect(B,5,array([11,22,33,44,55,66]))
    """

    blocksize = A.blocksize[0]
    BlockIndx = int(i/blocksize)
    rowstart = A.indptr[BlockIndx]
    rowend = A.indptr[BlockIndx+1]
    localRowIndx = i % blocksize

    # like matlab slicing:
    x = x.__array__().reshape((max(x.shape),))

    # counter = 0
    # for j in range(rowstart, rowend):
    #   indys = A.data[j,localRowIndx,:].nonzero()[0]
    #   increment = min(indys.shape[0], blocksize)
    #   A.data[j,localRowIndx,indys] = x[counter:(counter+increment), 0]
    #   counter += increment

    indys = A.data[rowstart:rowend, localRowIndx, :].nonzero()
    A.data[rowstart:rowend, localRowIndx, :][indys[0], indys[1]] = x
