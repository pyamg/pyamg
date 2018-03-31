"""General utility functions for pyamg"""
from __future__ import print_function

from warnings import warn

import numpy as np
import scipy as sp
from scipy.sparse import isspmatrix, isspmatrix_csr, isspmatrix_csc, \
    isspmatrix_bsr, csr_matrix, csc_matrix, bsr_matrix, coo_matrix, eye
from scipy.sparse.sputils import upcast
from pyamg.util.linalg import norm, cond, pinv_array
from scipy.linalg import eigvals
import pyamg.amg_core

__all__ = ['blocksize', 'diag_sparse', 'profile_solver', 'to_type',
           'type_prep', 'get_diagonal', 'UnAmal', 'Coord2RBM',
           'hierarchy_spectrum', 'print_table', 'get_block_diag', 'amalgamate',
           'scale_rows', 'scale_columns',
           'symmetric_rescaling', 'symmetric_rescaling_sa',
           'relaxation_as_linear_operator', 'filter_operator', 'scale_T',
           'get_Cpt_params', 'compute_BtBinv', 'eliminate_diag_dom_nodes',
           'levelize_strength_or_aggregation',
           'levelize_smooth_or_improve_candidates', 'filter_matrix_columns',
           'filter_matrix_rows', 'truncate_rows']

try:
    from scipy.sparse._sparsetools import csr_scale_rows, bsr_scale_rows
    from scipy.sparse._sparsetools import csr_scale_columns, bsr_scale_columns
except ImportError:
    from scipy.sparse.sparsetools import csr_scale_rows, bsr_scale_rows
    from scipy.sparse.sparsetools import csr_scale_columns, bsr_scale_columns


def blocksize(A):
    # Helper Function: return the blocksize of a matrix
    if isspmatrix_bsr(A):
        return A.blocksize[0]
    else:
        return 1


def profile_solver(ml, accel=None, **kwargs):
    """
    A quick solver to profile a particular multilevel object

    Parameters
    ----------
    ml : multilevel
        Fully constructed multilevel object
    accel : function pointer
        Pointer to a valid Krylov solver (e.g. gmres, cg)

    Returns
    -------
    residuals : array
        Array of residuals for each iteration

    See Also
    --------
    multilevel.psolve, multilevel.solve

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import spdiags, csr_matrix
    >>> from scipy.sparse.linalg import cg
    >>> from pyamg.classical import ruge_stuben_solver
    >>> from pyamg.util.utils import profile_solver
    >>> n=100
    >>> e = np.ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = csr_matrix(spdiags(data,[-1,0,1],n,n))
    >>> b = A*np.ones(A.shape[0])
    >>> ml = ruge_stuben_solver(A, max_coarse=10)
    >>> res = profile_solver(ml,accel=cg)

    """
    A = ml.levels[0].A
    b = A * sp.rand(A.shape[0], 1)
    residuals = []

    if accel is None:
        ml.solve(b, residuals=residuals, **kwargs)
    else:
        def callback(x):
            residuals.append(norm(np.ravel(b) - np.ravel(A*x)))
        M = ml.aspreconditioner(cycle=kwargs.get('cycle', 'V'))
        accel(A, b, M=M, callback=callback, **kwargs)

    return np.asarray(residuals)


def diag_sparse(A):
    """
    If A is a sparse matrix (e.g. csr_matrix or csc_matrix)
       - return the diagonal of A as an array

    Otherwise
       - return a csr_matrix with A on the diagonal

    Parameters
    ----------
    A : sparse matrix or 1d array
        General sparse matrix or array of diagonal entries

    Returns
    -------
    B : array or sparse matrix
        Diagonal sparse is returned as csr if A is dense otherwise return an
        array of the diagonal

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.util.utils import diag_sparse
    >>> d = 2.0*np.ones((3,)).ravel()
    >>> print diag_sparse(d).todense()
    [[ 2.  0.  0.]
     [ 0.  2.  0.]
     [ 0.  0.  2.]]

    """
    if isspmatrix(A):
        return A.diagonal()
    else:
        if(np.ndim(A) != 1):
            raise ValueError('input diagonal array expected to be 1d')
        return csr_matrix((np.asarray(A), np.arange(len(A)),
                          np.arange(len(A)+1)), (len(A), len(A)))


def scale_rows(A, v, copy=True):
    """
    Scale the sparse rows of a matrix

    Parameters
    ----------
    A : sparse matrix
        Sparse matrix with M rows
    v : array_like
        Array of M scales
    copy : {True,False}
        - If copy=True, then the matrix is copied to a new and different return
          matrix (e.g. B=scale_rows(A,v))
        - If copy=False, then the matrix is overwritten deeply (e.g.
          scale_rows(A,v,copy=False) overwrites A)

    Returns
    -------
    A : sparse matrix
        Scaled sparse matrix in original format

    See Also
    --------
    scipy.sparse._sparsetools.csr_scale_rows, scale_columns

    Notes
    -----
    - if A is a csc_matrix, the transpose A.T is passed to scale_columns
    - if A is not csr, csc, or bsr, it is converted to csr and sent
      to scale_rows

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import spdiags
    >>> from pyamg.util.utils import scale_rows
    >>> n=5
    >>> e = np.ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n-1).tocsr()
    >>> B = scale_rows(A,5*np.ones((A.shape[0],1)))
    """

    v = np.ravel(v)

    M, N = A.shape

    if not isspmatrix(A):
        raise ValueError('scale rows needs a sparse matrix')

    if M != len(v):
        raise ValueError('scale vector has incompatible shape')

    if copy:
        A = A.copy()
        A.data = np.asarray(A.data, dtype=upcast(A.dtype, v.dtype))
    else:
        v = np.asarray(v, dtype=A.dtype)

    if isspmatrix_csr(A):
        csr_scale_rows(M, N, A.indptr, A.indices, A.data, v)
    elif isspmatrix_bsr(A):
        R, C = A.blocksize
        bsr_scale_rows(int(M/R), int(N/C), R, C, A.indptr, A.indices,
                       np.ravel(A.data), v)
    elif isspmatrix_csc(A):
        pyamg.amg_core.csc_scale_rows(M, N, A.indptr, A.indices, A.data, v)
    else:
        fmt = A.format
        A = scale_rows(csr_matrix(A), v).asformat(fmt)

    return A

def scale_columns(A, v, copy=True):
    """
    Scale the sparse columns of a matrix

    Parameters
    ----------
    A : sparse matrix
        Sparse matrix with N rows
    v : array_like
        Array of N scales
    copy : {True,False}
        - If copy=True, then the matrix is copied to a new and different return
          matrix (e.g. B=scale_columns(A,v))
        - If copy=False, then the matrix is overwritten deeply (e.g.
          scale_columns(A,v,copy=False) overwrites A)

    Returns
    -------
    A : sparse matrix
        Scaled sparse matrix in original format

    See Also
    --------
    scipy.sparse._sparsetools.csr_scale_columns, scale_rows

    Notes
    -----
    - if A is a csc_matrix, the transpose A.T is passed to scale_rows
    - if A is not csr, csc, or bsr, it is converted to csr and sent to
      scale_rows

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import spdiags
    >>> from pyamg.util.utils import scale_columns
    >>> n=5
    >>> e = np.ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n-1).tocsr()
    >>> print scale_columns(A,5*np.ones((A.shape[1],1))).todense()
    [[ 10.  -5.   0.   0.]
     [ -5.  10.  -5.   0.]
     [  0.  -5.  10.  -5.]
     [  0.   0.  -5.  10.]
     [  0.   0.   0.  -5.]]

    """

    v = np.ravel(v)

    M, N = A.shape

    if not isspmatrix(A):
        raise ValueError('scale columns needs a sparse matrix')

    if N != len(v):
        raise ValueError('scale vector has incompatible shape')

    if copy:
        A = A.copy()
        A.data = np.asarray(A.data, dtype=upcast(A.dtype, v.dtype))
    else:
        v = np.asarray(v, dtype=A.dtype)

    if isspmatrix_csr(A):
        csr_scale_columns(M, N, A.indptr, A.indices, A.data, v)
    elif isspmatrix_bsr(A):
        R, C = A.blocksize
        bsr_scale_columns(int(M/R), int(N/C), R, C, A.indptr, A.indices,
                          np.ravel(A.data), v)
    elif isspmatrix_csc(A):
         pyamg.amg_core.csc_scale_columns(M, N, A.indptr, A.indices, A.data, v)
    else:
        fmt = A.format
        A = scale_columns(csr_matrix(A), v).asformat(fmt)

    return A


def symmetric_rescaling(A, copy=True):
    """
    Scale the matrix symmetrically::

        A = D^{-1/2} A D^{-1/2}

    where D=diag(A).

    The left multiplication is accomplished through scale_rows and the right
    multiplication is done through scale columns.

    Parameters
    ----------
    A : sparse matrix
        Sparse matrix with N rows
    copy : {True,False}
        - If copy=True, then the matrix is copied to a new and different return
          matrix (e.g. B=symmetric_rescaling(A))
        - If copy=False, then the matrix is overwritten deeply (e.g.
          symmetric_rescaling(A,copy=False) overwrites A)

    Returns
    -------
    D_sqrt : array
        Array of sqrt(diag(A))
    D_sqrt_inv : array
        Array of 1/sqrt(diag(A))
    DAD    : csr_matrix
        Symmetrically scaled A

    Notes
    -----
    - if A is not csr, it is converted to csr and sent to scale_rows

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import spdiags
    >>> from pyamg.util.utils import symmetric_rescaling
    >>> n=5
    >>> e = np.ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n).tocsr()
    >>> Ds, Dsi, DAD = symmetric_rescaling(A)
    >>> print DAD.todense()
    [[ 1.  -0.5  0.   0.   0. ]
     [-0.5  1.  -0.5  0.   0. ]
     [ 0.  -0.5  1.  -0.5  0. ]
     [ 0.   0.  -0.5  1.  -0.5]
     [ 0.   0.   0.  -0.5  1. ]]

    """
    if isspmatrix_csr(A) or isspmatrix_csc(A) or isspmatrix_bsr(A):
        if A.shape[0] != A.shape[1]:
            raise ValueError('expected square matrix')

        D = diag_sparse(A)
        mask = (D != 0)

        if A.dtype != complex:
            D_sqrt = np.sqrt(abs(D))
        else:
            # We can take square roots of negative numbers
            D_sqrt = np.sqrt(D)

        D_sqrt_inv = np.zeros_like(D_sqrt)
        D_sqrt_inv[mask] = 1.0/D_sqrt[mask]

        DAD = scale_rows(A, D_sqrt_inv, copy=copy)
        DAD = scale_columns(DAD, D_sqrt_inv, copy=False)

        return D_sqrt, D_sqrt_inv, DAD

    else:
        return symmetric_rescaling(csr_matrix(A))


def symmetric_rescaling_sa(A, B, BH=None):
    """
    Scale the matrix symmetrically::

        A = D^{-1/2} A D^{-1/2}

    where D=diag(A).  The left multiplication is accomplished through
    scale_rows and the right multiplication is done through scale columns.

    The candidates B and BH are scaled accordingly::

        B = D^{1/2} B
        BH = D^{1/2} BH

    Parameters
    ----------
    A : {sparse matrix}
        Sparse matrix with N rows
    B : {array}
        N x m array
    BH : {None, array}
        If A.symmetry == 'nonsymmetric, then BH must be an N x m array.
        Otherwise, BH is ignored.

    Returns
    -------
    Appropriately scaled A, B and BH, i.e.,
    A = D^{-1/2} A D^{-1/2},  B = D^{1/2} B,  and BH = D^{1/2} BH

    Notes
    -----
    - if A is not csr, it is converted to csr and sent to scale_rows

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import spdiags
    >>> from pyamg.util.utils import symmetric_rescaling_sa
    >>> n=5
    >>> e = np.ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n).tocsr()
    >>> B = e.copy().reshape(-1,1)
    >>> [DAD, DB, DBH] = symmetric_rescaling_sa(A,B,BH=None)
    >>> print DAD.todense()
    [[ 1.  -0.5  0.   0.   0. ]
     [-0.5  1.  -0.5  0.   0. ]
     [ 0.  -0.5  1.  -0.5  0. ]
     [ 0.   0.  -0.5  1.  -0.5]
     [ 0.   0.   0.  -0.5  1. ]]
    >>> print DB
    [[ 1.41421356]
     [ 1.41421356]
     [ 1.41421356]
     [ 1.41421356]
     [ 1.41421356]]
    """

    # rescale A
    [D_sqrt, D_sqrt_inv, A] = symmetric_rescaling(A, copy=False)
    # scale candidates
    for i in range(B.shape[1]):
        B[:, i] = np.ravel(B[:, i])*np.ravel(D_sqrt)

    if hasattr(A, 'symmetry'):
        if A.symmetry == 'nonsymmetric':
            if BH is None:
                raise ValueError("BH should be an n x m array")
            else:
                for i in range(BH.shape[1]):
                    BH[:, i] = np.ravel(BH[:, i])*np.ravel(D_sqrt)

    return [A, B, BH]


def type_prep(upcast_type, varlist):
    """
    Loop over all elements of varlist and convert them to upcasttype
    The only difference with pyamg.util.utils.to_type(...), is that scalars
    are wrapped into (1,0) arrays.  This is desirable when passing
    the numpy complex data type to C routines and complex scalars aren't
    handled correctly

    Parameters
    ----------
    upcast_type : data type
        e.g. complex, float64 or complex128
    varlist : list
        list may contain arrays, mat's, sparse matrices, or scalars
        the elements may be float, int or complex

    Returns
    -------
    Returns upcast-ed varlist to upcast_type

    Notes
    -----
    Useful when harmonizing the types of variables, such as
    if A and b are complex, but x,y and z are not.

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.util.utils import type_prep
    >>> from scipy.sparse.sputils import upcast
    >>> x = np.ones((5,1))
    >>> y = 2.0j*np.ones((5,1))
    >>> z = 2.3
    >>> varlist = type_prep(upcast(x.dtype, y.dtype), [x, y, z])

    """
    varlist = to_type(upcast_type, varlist)
    for i in range(len(varlist)):
        if np.isscalar(varlist[i]):
            varlist[i] = np.array([varlist[i]])

    return varlist


def to_type(upcast_type, varlist):
    """
    Loop over all elements of varlist and convert them to upcasttype

    Parameters
    ----------
    upcast_type : data type
        e.g. complex, float64 or complex128
    varlist : list
        list may contain arrays, mat's, sparse matrices, or scalars
        the elements may be float, int or complex

    Returns
    -------
    Returns upcast-ed varlist to upcast_type

    Notes
    -----
    Useful when harmonizing the types of variables, such as
    if A and b are complex, but x,y and z are not.

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.util.utils import to_type
    >>> from scipy.sparse.sputils import upcast
    >>> x = np.ones((5,1))
    >>> y = 2.0j*np.ones((5,1))
    >>> varlist = to_type(upcast(x.dtype, y.dtype), [x, y])

    """

    # convert_type = type(np.array([0], upcast_type)[0])

    for i in range(len(varlist)):

        # convert scalars to complex
        if np.isscalar(varlist[i]):
            varlist[i] = np.array([varlist[i]], upcast_type)[0]
        else:
            # convert sparse and dense mats to complex
            try:
                if varlist[i].dtype != upcast_type:
                    varlist[i] = varlist[i].astype(upcast_type)
            except AttributeError:
                warn('Failed to cast in to_type')
                pass

    return varlist


def get_diagonal(A, norm_eq=False, inv=False):
    """ Return the diagonal or inverse of diagonal for
        A, (A.H A) or (A A.H)

    Parameters
    ----------
    A   : {dense or sparse matrix}
        e.g. array, matrix, csr_matrix, ...
    norm_eq : {0, 1, 2}
        0 ==> D = diag(A)
        1 ==> D = diag(A.H A)
        2 ==> D = diag(A A.H)
    inv : {True, False}
        If True, D = 1.0/D

    Returns
    -------
    diagonal, D, of appropriate system

    Notes
    -----
    This function is especially useful for its fast methods
    of obtaining diag(A A.H) and diag(A.H A).  Dinv is zero
    wherever D is zero

    Examples
    --------
    >>> from pyamg.util.utils import get_diagonal
    >>> from pyamg.gallery import poisson
    >>> A = poisson( (5,), format='csr' )
    >>> D = get_diagonal(A)
    >>> print D
    [ 2.  2.  2.  2.  2.]
    >>> D = get_diagonal(A, norm_eq=1, inv=True)
    >>> print D
    [ 0.2         0.16666667  0.16666667  0.16666667  0.2       ]

    """

    # if not isspmatrix(A):
    if not (isspmatrix_csr(A) or isspmatrix_csc(A) or isspmatrix_bsr(A)):
        warn('Implicit conversion to sparse matrix')
        A = csr_matrix(A)

    # critical to sort the indices of A
    A.sort_indices()
    if norm_eq == 1:
        # This transpose involves almost no work, use csr data structures as
        # csc, or vice versa
        At = A.T
        D = (At.multiply(At.conjugate()))*np.ones((At.shape[0],))
    elif norm_eq == 2:
        D = (A.multiply(A.conjugate()))*np.ones((A.shape[0],))
    else:
        D = A.diagonal()

    if inv:
        Dinv = np.zeros_like(D)
        mask = (D != 0.0)
        Dinv[mask] = 1.0 / D[mask]
        return Dinv
    else:
        return D


def get_block_diag(A, blocksize, inv_flag=True):
    """
    Return the block diagonal of A, in array form

    Parameters
    ----------
    A : csr_matrix
        assumed to be square
    blocksize : int
        square block size for the diagonal
    inv_flag : bool
        if True, return the inverse of the block diagonal

    Returns
    -------
    block_diag : array
        block diagonal of A in array form,
        array size is (A.shape[0]/blocksize, blocksize, blocksize)

    Examples
    --------
    >>> from scipy import arange
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.util import get_block_diag
    >>> A = csr_matrix(arange(36).reshape(6,6))
    >>> block_diag_inv = get_block_diag(A, blocksize=2, inv_flag=False)
    >>> print block_diag_inv
    [[[  0.   1.]
      [  6.   7.]]
    <BLANKLINE>
     [[ 14.  15.]
      [ 20.  21.]]
    <BLANKLINE>
     [[ 28.  29.]
      [ 34.  35.]]]
    >>> block_diag_inv = get_block_diag(A, blocksize=2, inv_flag=True)

    """

    if not isspmatrix(A):
        raise TypeError('Expected sparse matrix')
    if A.shape[0] != A.shape[1]:
        raise ValueError("Expected square matrix")
    if sp.mod(A.shape[0], blocksize) != 0:
        raise ValueError("blocksize and A.shape must be compatible")

    # If the block diagonal of A already exists, return that
    if hasattr(A, 'block_D_inv') and inv_flag:
        if (A.block_D_inv.shape[1] == blocksize) and\
           (A.block_D_inv.shape[2] == blocksize) and \
           (A.block_D_inv.shape[0] == int(A.shape[0]/blocksize)):
            return A.block_D_inv
    elif hasattr(A, 'block_D') and (not inv_flag):
        if (A.block_D.shape[1] == blocksize) and\
           (A.block_D.shape[2] == blocksize) and \
           (A.block_D.shape[0] == int(A.shape[0]/blocksize)):
            return A.block_D

    # Convert to BSR
    if not isspmatrix_bsr(A):
        A = bsr_matrix(A, blocksize=(blocksize, blocksize))
    if A.blocksize != (blocksize, blocksize):
        A = A.tobsr(blocksize=(blocksize, blocksize))

    # Peel off block diagonal by extracting block entries from the now BSR
    # matrix A
    A = A.asfptype()
    block_diag = sp.zeros((int(A.shape[0]/blocksize), blocksize, blocksize),
                          dtype=A.dtype)

    AAIJ = (sp.arange(1, A.indices.shape[0]+1), A.indices, A.indptr)
    shape = (int(A.shape[0]/blocksize), int(A.shape[0]/blocksize))
    diag_entries = csr_matrix(AAIJ, shape=shape).diagonal()
    diag_entries -= 1
    nonzero_mask = (diag_entries != -1)
    diag_entries = diag_entries[nonzero_mask]
    if diag_entries.shape != (0,):
        block_diag[nonzero_mask, :, :] = A.data[diag_entries, :, :]

    if inv_flag:
        # Invert each block
        if block_diag.shape[1] < 7:
            # This specialized routine lacks robustness for large matrices
            pyamg.amg_core.pinv_array(block_diag.ravel(), block_diag.shape[0],
                                      block_diag.shape[1], 'T')
        else:
            pinv_array(block_diag)
        A.block_D_inv = block_diag
    else:
        A.block_D = block_diag

    return block_diag


def amalgamate(A, blocksize):
    """
    Amalgamate matrix A

    Parameters
    ----------
    A : csr_matrix
        Matrix to amalgamate
    blocksize : int
        blocksize to use while amalgamating

    Returns
    -------
    A_amal : csr_matrix
        Amalgamated  matrix A, first, convert A to BSR with square blocksize
        and then return a CSR matrix of ones using the resulting BSR indptr and
        indices

    Notes
    -----
    inverse operation of UnAmal for square matrices

    Examples
    --------
    >>> from numpy import array
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.util.utils import amalgamate
    >>> row = array([0,0,1])
    >>> col = array([0,2,1])
    >>> data = array([1,2,3])
    >>> A = csr_matrix( (data,(row,col)), shape=(4,4) )
    >>> A.todense()
    matrix([[1, 0, 2, 0],
            [0, 3, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])
    >>> amalgamate(A,2).todense()
    matrix([[ 1.,  1.],
            [ 0.,  0.]])


    """

    if blocksize == 1:
        return A
    elif sp.mod(A.shape[0], blocksize) != 0:
        raise ValueError("Incompatible blocksize")

    A = A.tobsr(blocksize=(blocksize, blocksize))
    A.sort_indices()
    subI = (np.ones(A.indices.shape), A.indices, A.indptr)
    shape = (int(A.shape[0]/A.blocksize[0]),
             int(A.shape[1]/A.blocksize[1]))
    return csr_matrix(subI, shape=shape)


def UnAmal(A, RowsPerBlock, ColsPerBlock):
    """

    Unamalgamate a CSR A with blocks of 1's.  This operation is equivalent to
    replacing each entry of A with ones(RowsPerBlock, ColsPerBlock), i.e., this
    is equivalent to setting all of A's nonzeros to 1 and then doing a
    Kronecker product between A and ones(RowsPerBlock, ColsPerBlock).

    Parameters
    ----------
    A : csr_matrix
        Amalgamted matrix
    RowsPerBlock : int
        Give A blocks of size (RowsPerBlock, ColsPerBlock)
    ColsPerBlock : int
        Give A blocks of size (RowsPerBlock, ColsPerBlock)

    Returns
    -------
    A : bsr_matrix
        Returns A.data[:] = 1, followed by a Kronecker product of A and
        ones(RowsPerBlock, ColsPerBlock)

    Examples
    --------
    >>> from numpy import array
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.util.utils import UnAmal
    >>> row = array([0,0,1,2,2,2])
    >>> col = array([0,2,2,0,1,2])
    >>> data = array([1,2,3,4,5,6])
    >>> A = csr_matrix( (data,(row,col)), shape=(3,3) )
    >>> A.todense()
    matrix([[1, 0, 2],
            [0, 0, 3],
            [4, 5, 6]])
    >>> UnAmal(A,2,2).todense()
    matrix([[ 1.,  1.,  0.,  0.,  1.,  1.],
            [ 1.,  1.,  0.,  0.,  1.,  1.],
            [ 0.,  0.,  0.,  0.,  1.,  1.],
            [ 0.,  0.,  0.,  0.,  1.,  1.],
            [ 1.,  1.,  1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.,  1.,  1.]])

    """
    data = np.ones((A.indices.shape[0], RowsPerBlock, ColsPerBlock))
    blockI = (data, A.indices, A.indptr)
    shape = (RowsPerBlock*A.shape[0], ColsPerBlock*A.shape[1])
    return bsr_matrix(blockI, shape=shape)


def print_table(table, title='', delim='|', centering='center', col_padding=2,
                header=True, headerchar='-'):
    """
    Print a table from a list of lists representing the rows of a table


    Parameters
    ----------
    table : list
        list of lists, e.g. a table with 3 columns and 2 rows could be
        [ ['0,0', '0,1', '0,2'], ['1,0', '1,1', '1,2'] ]
    title : string
        Printed centered above the table
    delim : string
        character to delimit columns
    centering : {'left', 'right', 'center'}
        chooses justification for columns
    col_padding : int
        number of blank spaces to add to each column
    header : {True, False}
        Does the first entry of table contain column headers?
    headerchar : {string}
        character to separate column headers from rest of table

    Returns
    -------
    string representing table that's ready to be printed

    Notes
    -----
    The string for the table will have correctly justified columns
    with extra padding added into each column entry to ensure columns align.
    The characters to delimit the columns can be user defined.  This
    should be useful for printing convergence data from tests.


    Examples
    --------
    >>> from pyamg.util.utils import print_table
    >>> table = [ ['cos(0)', 'cos(pi/2)', 'cos(pi)'], ['0.0', '1.0', '0.0'] ]
    >>> table1 = print_table(table)                 # string to print
    >>> table2 = print_table(table, delim='||')
    >>> table3 = print_table(table, headerchar='*')
    >>> table4 = print_table(table, col_padding=6, centering='left')

    """

    table_str = '\n'

    # sometimes, the table will be passed in as (title, table)
    if isinstance(table, tuple):
        title = table[0]
        table = table[1]

    # Calculate each column's width
    colwidths = []
    for i in range(len(table)):
        # extend colwidths for row i
        for k in range(len(table[i]) - len(colwidths)):
            colwidths.append(-1)

        # Update colwidths if table[i][j] is wider than colwidth[j]
        for j in range(len(table[i])):
            if len(table[i][j]) > colwidths[j]:
                colwidths[j] = len(table[i][j])

    # Factor in extra column padding
    for i in range(len(colwidths)):
        colwidths[i] += col_padding

    # Total table width
    ttwidth = sum(colwidths) + len(delim)*(len(colwidths)-1)

    # Print Title
    if len(title) > 0:
        title = title.split("\n")
        for i in range(len(title)):
            table_str += str.center(title[i], ttwidth) + '\n'
        table_str += "\n"

    # Choose centering scheme
    centering = centering.lower()
    if centering == 'center':
        centering = str.center
    if centering == 'right':
        centering = str.rjust
    if centering == 'left':
        centering = str.ljust

    if header:
        # Append Column Headers
        for elmt, elmtwidth in zip(table[0], colwidths):
            table_str += centering(str(elmt), elmtwidth) + delim
        if table[0] != []:
            table_str = table_str[:-len(delim)] + '\n'

        # Append Header Separator
        #              Total Column Width            Total Col Delimiter Widths
        if len(headerchar) == 0:
            headerchar = ' '
        table_str += headerchar *\
            int(sp.ceil(float(ttwidth)/float(len(headerchar)))) + '\n'

        table = table[1:]

    for row in table:
        for elmt, elmtwidth in zip(row, colwidths):
            table_str += centering(str(elmt), elmtwidth) + delim
        if row != []:
            table_str = table_str[:-len(delim)] + '\n'
        else:
            table_str += '\n'

    return table_str


def hierarchy_spectrum(mg, filter=True, plot=False):
    """
    Examine a multilevel hierarchy's spectrum

    Parameters
    ----------
    mg { pyamg multilevel hierarchy }
        e.g. generated with smoothed_aggregation_solver(...) or
        ruge_stuben_solver(...)

    Returns
    -------
    (1) table to standard out detailing the spectrum of each level in mg
    (2) if plot==True, a sequence of plots in the complex plane of the
        spectrum at each level

    Notes
    -----
    This can be useful for troubleshooting and when examining how your
    problem's nature changes from level to level

    Examples
    --------
    >>> from pyamg import smoothed_aggregation_solver
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.utils import hierarchy_spectrum
    >>> A = poisson( (1,), format='csr' )
    >>> ml = smoothed_aggregation_solver(A)
    >>> hierarchy_spectrum(ml)
    <BLANKLINE>
     Level min(re(eig)) max(re(eig)) num re(eig) < 0 num re(eig) > 0 cond_2(A)
    ---------------------------------------------------------------------------
       0      2.000        2.000            0               1         1.00e+00
    <BLANKLINE>
    <BLANKLINE>
     Level min(im(eig)) max(im(eig)) num im(eig) < 0 num im(eig) > 0 cond_2(A)
    ---------------------------------------------------------------------------
       0      0.000        0.000            0               0         1.00e+00
    <BLANKLINE>


    """

    real_table = [['Level', 'min(re(eig))', 'max(re(eig))', 'num re(eig) < 0',
                   'num re(eig) > 0', 'cond_2(A)']]
    imag_table = [['Level', 'min(im(eig))', 'max(im(eig))', 'num im(eig) < 0',
                   'num im(eig) > 0', 'cond_2(A)']]

    for i in range(len(mg.levels)):
        A = mg.levels[i].A.tocsr()

        if filter is True:
            # Filter out any zero rows and columns of A
            A.eliminate_zeros()
            nnz_per_row = A.indptr[0:-1] - A.indptr[1:]
            nonzero_rows = (nnz_per_row != 0).nonzero()[0]
            A = A.tocsc()
            nnz_per_col = A.indptr[0:-1] - A.indptr[1:]
            nonzero_cols = (nnz_per_col != 0).nonzero()[0]
            nonzero_rowcols = sp.union1d(nonzero_rows, nonzero_cols)
            A = np.mat(A.todense())
            A = A[nonzero_rowcols, :][:, nonzero_rowcols]
        else:
            A = np.mat(A.todense())

        e = eigvals(A)
        c = cond(A)
        lambda_min = min(sp.real(e))
        lambda_max = max(sp.real(e))
        num_neg = max(e[sp.real(e) < 0.0].shape)
        num_pos = max(e[sp.real(e) > 0.0].shape)
        real_table.append([str(i), ('%1.3f' % lambda_min),
                          ('%1.3f' % lambda_max),
                          str(num_neg), str(num_pos), ('%1.2e' % c)])

        lambda_min = min(sp.imag(e))
        lambda_max = max(sp.imag(e))
        num_neg = max(e[sp.imag(e) < 0.0].shape)
        num_pos = max(e[sp.imag(e) > 0.0].shape)
        imag_table.append([str(i), ('%1.3f' % lambda_min),
                          ('%1.3f' % lambda_max),
                          str(num_neg), str(num_pos), ('%1.2e' % c)])

        if plot:
            import pylab
            pylab.figure(i+1)
            pylab.plot(sp.real(e), sp.imag(e), 'kx')
            handle = pylab.title('Level %d Spectrum' % i)
            handle.set_fontsize(19)
            handle = pylab.xlabel('real(eig)')
            handle.set_fontsize(17)
            handle = pylab.ylabel('imag(eig)')
            handle.set_fontsize(17)

    print(print_table(real_table))
    print(print_table(imag_table))

    if plot:
        pylab.show()


def Coord2RBM(numNodes, numPDEs, x, y, z):
    """
    Convert 2D or 3D coordinates into Rigid body modes for use as near
    nullspace modes in elasticity AMG solvers

    Parameters
    ----------
    numNodes : int
        Number of nodes
    numPDEs :
        Number of dofs per node
    x,y,z : array_like
        Coordinate vectors

    Returns
    -------
    rbm : matrix
        A matrix of size (numNodes*numPDEs) x (1 | 6) containing the 6 rigid
        body modes

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.util.utils import Coord2RBM
    >>> a = np.array([0,1,2])
    >>> Coord2RBM(3,6,a,a,a)
    matrix([[ 1.,  0.,  0.,  0.,  0., -0.],
            [ 0.,  1.,  0., -0.,  0.,  0.],
            [ 0.,  0.,  1.,  0., -0.,  0.],
            [ 0.,  0.,  0.,  1.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  1.],
            [ 1.,  0.,  0.,  0.,  1., -1.],
            [ 0.,  1.,  0., -1.,  0.,  1.],
            [ 0.,  0.,  1.,  1., -1.,  0.],
            [ 0.,  0.,  0.,  1.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  1.],
            [ 1.,  0.,  0.,  0.,  2., -2.],
            [ 0.,  1.,  0., -2.,  0.,  2.],
            [ 0.,  0.,  1.,  2., -2.,  0.],
            [ 0.,  0.,  0.,  1.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  1.]])
    """

    # check inputs
    if(numPDEs == 1):
        numcols = 1
    elif((numPDEs == 3) or (numPDEs == 6)):
        numcols = 6
    else:
        raise ValueError("Coord2RBM(...) only supports 1, 3 or 6 PDEs per\
                          spatial location,i.e. numPDEs = [1 | 3 | 6].\
                          You've entered " + str(numPDEs) + ".")

    if((max(x.shape) != numNodes) or
       (max(y.shape) != numNodes) or
       (max(z.shape) != numNodes)):
        raise ValueError("Coord2RBM(...) requires coordinate vectors of equal\
                          length.  Length must be numNodes = " + str(numNodes))

    # if( (min(x.shape) != 1) or (min(y.shape) != 1) or (min(z.shape) != 1) ):
    #    raise ValueError("Coord2RBM(...) requires coordinate vectors that are
    #    (numNodes x 1) or (1 x numNodes).")

    # preallocate rbm
    rbm = np.mat(np.zeros((numNodes*numPDEs, numcols)))

    for node in range(numNodes):
        dof = node*numPDEs

        if(numPDEs == 1):
            rbm[node] = 1.0

        if(numPDEs == 6):
            for ii in range(3, 6):  # lower half = [ 0 I ]
                for jj in range(0, 6):
                    if(ii == jj):
                        rbm[dof+ii, jj] = 1.0
                    else:
                        rbm[dof+ii, jj] = 0.0

        if((numPDEs == 3) or (numPDEs == 6)):
            for ii in range(0, 3):  # upper left = [ I ]
                for jj in range(0, 3):
                    if(ii == jj):
                        rbm[dof+ii, jj] = 1.0
                    else:
                        rbm[dof+ii, jj] = 0.0

            for ii in range(0, 3):  # upper right = [ Q ]
                for jj in range(3, 6):
                    if(ii == (jj-3)):
                        rbm[dof+ii, jj] = 0.0
                    else:
                        if((ii+jj) == 4):
                            rbm[dof+ii, jj] = z[node]
                        elif((ii+jj) == 5):
                            rbm[dof+ii, jj] = y[node]
                        elif((ii+jj) == 6):
                            rbm[dof+ii, jj] = x[node]
                        else:
                            rbm[dof+ii, jj] = 0.0

            ii = 0
            jj = 5
            rbm[dof+ii, jj] *= -1.0

            ii = 1
            jj = 3
            rbm[dof+ii, jj] *= -1.0

            ii = 2
            jj = 4
            rbm[dof+ii, jj] *= -1.0

    return rbm


def relaxation_as_linear_operator(method, A, b):
    """
    Create a linear operator that applies a relaxation method for the
    given right-hand-side

    Parameters
    ----------
    methods : {tuple or string}
        Relaxation descriptor: Each tuple must be of the form ('method','opts')
        where 'method' is the name of a supported smoother, e.g., gauss_seidel,
        and 'opts' a dict of keyword arguments to the smoother, e.g., opts =
        {'sweep':symmetric}.  If string, must be that of a supported smoother,
        e.g., gauss_seidel.

    Returns
    -------
    linear operator that applies the relaxation method to a vector for a
    fixed right-hand-side, b.

    Notes
    -----

    This method is primarily used to improve B during the aggregation setup
    phase.  Here b = 0, and each relaxation call can improve the quality of B,
    especially near the boundaries.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.utils import relaxation_as_linear_operator
    >>> import numpy as np
    >>> A = poisson((100,100), format='csr')           # matrix
    >>> B = np.ones((A.shape[0],1))                 # Candidate vector
    >>> b = np.zeros((A.shape[0]))                  # RHS
    >>> relax = relaxation_as_linear_operator('gauss_seidel', A, b)
    >>> B = relax*B

    """
    from pyamg import relaxation
    from scipy.sparse.linalg.interface import LinearOperator
    import pyamg.multilevel

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        else:
            return v, {}

    # setup variables
    accepted_methods = ['gauss_seidel', 'block_gauss_seidel', 'sor',
                        'gauss_seidel_ne', 'gauss_seidel_nr', 'jacobi',
                        'block_jacobi', 'richardson', 'schwarz',
                        'strength_based_schwarz', 'jacobi_ne']

    b = np.array(b, dtype=A.dtype)
    fn, kwargs = unpack_arg(method)
    lvl = pyamg.multilevel_solver.level()
    lvl.A = A

    # Retrieve setup call from relaxation.smoothing for this relaxation method
    if not accepted_methods.__contains__(fn):
        raise NameError("invalid relaxation method: ", fn)
    try:
        setup_smoother = getattr(relaxation.smoothing, 'setup_' + fn)
    except NameError:
        raise NameError("invalid presmoother method: ", fn)
    # Get relaxation routine that takes only (A, x, b) as parameters
    relax = setup_smoother(lvl, **kwargs)

    # Define matvec
    def matvec(x):
        xcopy = x.copy()
        relax(A, xcopy, b)
        return xcopy

    return LinearOperator(A.shape, matvec, dtype=A.dtype)


def filter_operator(A, C, B, Bf, BtBinv=None):
    """
    Filter the matrix A according to the matrix graph of C,
    while ensuring that the new, filtered A satisfies:  A_new*B = Bf.

    A : {csr_matrix, bsr_matrix}
        n x m matrix to filter
    C : {csr_matrix, bsr_matrix}
        n x m matrix representing the couplings in A to keep
    B : {array}
        m x k array of near nullspace vectors
    Bf : {array}
        n x k array of near nullspace vectors to place in span(A)
    BtBinv : {None, array}
        3 dimensional array such that,
        BtBinv[i] = pinv(B_i.H Bi), and B_i is B restricted
        to the neighborhood (with respect to the matrix graph
        of C) of dof of i.  If None is passed in, this array is
        computed internally.

    Returns
    -------
    A : sparse matrix updated such that sparsity structure of A now matches
    that of C, and that the relationship A*B = Bf holds.

    Notes
    -----
    This procedure allows for certain important modes (i.e., Bf) to be placed
    in span(A) by way of row-wise l2-projections that enforce the relationship
    A*B = Bf.  This is useful for maintaining certain modes (e.g., the
    constant) in the span of prolongation.

    Examples
    --------
    >>> from numpy import ones, array
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.util.utils import filter_operator
    >>> A = array([ [1.,1,1],[1,1,1],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])
    >>> C = array([ [1.,1,0],[1,1,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])
    >>> B = ones((3,1))
    >>> Bf = ones((6,1))
    >>> filter_operator(csr_matrix(A), csr_matrix(C), B, Bf).todense()
    matrix([[ 0.5,  0.5,  0. ],
            [ 0.5,  0.5,  0. ],
            [ 0. ,  1. ,  0. ],
            [ 0. ,  1. ,  0. ],
            [ 0. ,  0. ,  1. ],
            [ 0. ,  0. ,  1. ]])

    Notes
    -----

    This routine is primarily used in
    pyamg.aggregation.smooth.energy_prolongation_smoother, where it is used to
    generate a suitable initial guess for the energy-minimization process, when
    root-node style SA is used.  Essentially, the tentative prolongator, T, is
    processed by this routine to produce fine-grid nullspace vectors when
    multiplying coarse-grid nullspace vectors, i.e., T*B = Bf.  This is
    possible for any arbitrary vectors B and Bf, so long as the sparsity
    structure of T is rich enough.

    When generating initial guesses for root-node style prolongation operators,
    this function is usually called before pyamg.uti.utils.scale_T

    """

    # First preprocess the parameters
    Nfine = A.shape[0]
    if A.shape[0] != C.shape[0]:
        raise ValueError('A and C must be the same size')
    if A.shape[1] != C.shape[1]:
        raise ValueError('A and C must be the same size')

    if isspmatrix_bsr(C):
        isBSR = True
        ColsPerBlock = C.blocksize[1]
        RowsPerBlock = C.blocksize[0]
        Nnodes = int(Nfine/RowsPerBlock)
        if not isspmatrix_bsr(A):
            raise ValueError('A and C must either both be CSR or BSR')
        elif (ColsPerBlock != A.blocksize[1]) or\
             (RowsPerBlock != A.blocksize[0]):
            raise ValueError('A and C must have same BSR blocksizes')
    elif isspmatrix_csr(C):
        isBSR = False
        ColsPerBlock = 1
        RowsPerBlock = 1
        Nnodes = int(Nfine/RowsPerBlock)
        if not isspmatrix_csr(A):
            raise ValueError('A and C must either both be CSR or BSR')
    else:
        raise ValueError('A and C must either both be CSR or BSR')

    if len(Bf.shape) == 1:
        Bf = Bf.reshape(-1, 1)
    if Bf.shape[0] != A.shape[0]:
        raise ValueError('A and Bf must have the same first dimension')

    if len(B.shape) == 1:
        B = B.reshape(-1, 1)
    if B.shape[0] != A.shape[1]:
        raise ValueError('A and B must have matching dimensions such\
                          that A*B is computable')

    if B.shape[1] != Bf.shape[1]:
        raise ValueError('B and Bf must have the same second\
                          dimension')
    else:
        NullDim = B.shape[1]

    if A.dtype == int:
        A.data = np.array(A.data, dtype=float)
    if B.dtype == int:
        B.data = np.array(B.data, dtype=float)
    if Bf.dtype == int:
        Bf.data = np.array(Bf.data, dtype=float)
    if (A.dtype != B.dtype) or (A.dtype != Bf.dtype):
        raise TypeError('A, B and Bf must of the same dtype')

    # First, preprocess some values for filtering.  Construct array of
    # inv(Bi'Bi), where Bi is B restricted to row i's sparsity pattern in
    # C. This array is used multiple times in Satisfy_Constraints(...).
    if BtBinv is None:
        BtBinv = compute_BtBinv(B, C)

    # Filter A according to C's matrix graph
    C = C.copy()
    C.data[:] = 1
    A = A.multiply(C)
    # add explicit zeros to A wherever C is nonzero, but A is zero
    A = A.tocoo()
    C = C.tocoo()
    A.data = np.hstack((np.zeros(C.data.shape, dtype=A.dtype), A.data))
    A.row = np.hstack((C.row, A.row))
    A.col = np.hstack((C.col, A.col))
    if isBSR:
        A = A.tobsr((RowsPerBlock, ColsPerBlock))
    else:
        A = A.tocsr()

    # Calculate difference between A*B and Bf
    diff = A*B - Bf

    # Right multiply each row i of A with
    # A_i <--- A_i - diff_i*inv(B_i.T B_i)*Bi.T
    # where A_i, and diff_i denote restriction to just row i, and B_i denotes
    # restriction to multiple rows corresponding to the the allowed nz's for
    # row i in A_i.  A_i also represents just the nonzeros for row i.
    pyamg.amg_core.satisfy_constraints_helper(RowsPerBlock, ColsPerBlock,
                                              Nnodes, NullDim,
                                              np.conjugate(np.ravel(B)),
                                              np.ravel(diff),
                                              np.ravel(BtBinv), A.indptr,
                                              A.indices, np.ravel(A.data))

    A.eliminate_zeros()
    return A


def scale_T(T, P_I, I_F):
    '''
    Helper function that scales T with a right multiplication by a block
    diagonal inverse, so that T is the identity at C-node rows.

    Parameters
    ----------
    T : {bsr_matrix}
        Tentative prolongator, with square blocks in the BSR data structure,
        and a non-overlapping block-diagonal structure
    P_I : {bsr_matrix}
        Interpolation operator that carries out only simple injection from the
        coarse grid to fine grid Cpts nodes
    I_F : {bsr_matrix}
        Identity operator on Fpts, i.e., the action of this matrix zeros
        out entries in a vector at all Cpts, leaving Fpts untouched

    Returns
    -------
    T : {bsr_matrix}
        Tentative prolongator scaled to be identity at C-pt nodes

    Examples
    --------
    >>> from scipy.sparse import csr_matrix, bsr_matrix
    >>> from scipy import matrix, array
    >>> from pyamg.util.utils import scale_T
    >>> T = matrix([[ 1.0,  0.,   0. ],
    ...             [ 0.5,  0.,   0. ],
    ...             [ 0. ,  1.,   0. ],
    ...             [ 0. ,  0.5,  0. ],
    ...             [ 0. ,  0.,   1. ],
    ...             [ 0. ,  0.,   0.25 ]])
    >>> P_I = matrix([[ 0.,  0.,   0. ],
    ...               [ 1.,  0.,   0. ],
    ...               [ 0.,  1.,   0. ],
    ...               [ 0.,  0.,   0. ],
    ...               [ 0.,  0.,   0. ],
    ...               [ 0.,  0.,   1. ]])
    >>> I_F = matrix([[ 1.,  0.,  0.,  0.,  0.,  0.],
    ...               [ 0.,  0.,  0.,  0.,  0.,  0.],
    ...               [ 0.,  0.,  0.,  0.,  0.,  0.],
    ...               [ 0.,  0.,  0.,  1.,  0.,  0.],
    ...               [ 0.,  0.,  0.,  0.,  1.,  0.],
    ...               [ 0.,  0.,  0.,  0.,  0.,  0.]])
    >>> scale_T(bsr_matrix(T), bsr_matrix(P_I), bsr_matrix(I_F)).todense()
    matrix([[ 2. ,  0. ,  0. ],
            [ 1. ,  0. ,  0. ],
            [ 0. ,  1. ,  0. ],
            [ 0. ,  0.5,  0. ],
            [ 0. ,  0. ,  4. ],
            [ 0. ,  0. ,  1. ]])

    Notes
    -----
    This routine is primarily used in
    pyamg.aggregation.smooth.energy_prolongation_smoother, where it is used to
    generate a suitable initial guess for the energy-minimization process, when
    root-node style SA is used.  This function, scale_T, takes an existing
    tentative prolongator and ensures that it injects from the coarse-grid to
    fine-grid root-nodes.

    When generating initial guesses for root-node style prolongation operators,
    this function is usually called after pyamg.uti.utils.filter_operator

    This function assumes that the eventual coarse-grid nullspace vectors
    equal coarse-grid injection applied to the fine-grid nullspace vectors.

    '''

    if not isspmatrix_bsr(T):
        raise TypeError('Expected BSR matrix T')
    elif T.blocksize[0] != T.blocksize[1]:
        raise TypeError('Expected BSR matrix T with square blocks')
    if not isspmatrix_bsr(P_I):
        raise TypeError('Expected BSR matrix P_I')
    elif P_I.blocksize[0] != P_I.blocksize[1]:
        raise TypeError('Expected BSR matrix P_I with square blocks')
    if not isspmatrix_bsr(I_F):
        raise TypeError('Expected BSR matrix I_F')
    elif I_F.blocksize[0] != I_F.blocksize[1]:
        raise TypeError('Expected BSR matrix I_F with square blocks')
    if (I_F.blocksize[0] != P_I.blocksize[0]) or\
       (I_F.blocksize[0] != T.blocksize[0]):
        raise TypeError('Expected identical blocksize in I_F, P_I and T')

    # Only do if we have a non-trivial coarse-grid
    if P_I.nnz > 0:
        # Construct block diagonal inverse D
        D = P_I.T*T
        if D.nnz > 0:
            # changes D in place
            pinv_array(D.data)

        # Scale T to be identity at root-nodes
        T = T*D

        # Ensure coarse-grid injection
        T = I_F*T + P_I

    return T


def get_Cpt_params(A, Cnodes, AggOp, T):
    ''' Helper function that returns a dictionary of sparse matrices and arrays
        which allow us to easily operate on Cpts and Fpts separately.

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Operator
    Cnodes : {array}
        Array of all root node indices.  This is an array of nodal indices,
        not degree-of-freedom indices.  If the blocksize of T is 1, then
        nodal indices and degree-of-freedom indices coincide.
    AggOp : {csr_matrix}
        Aggregation operator corresponding to A
    T : {bsr_matrix}
        Tentative prolongator based on AggOp

    Returns
    -------
    Dictionary containing these parameters:

    P_I : {bsr_matrix}
        Interpolation operator that carries out only simple injection from the
        coarse grid to fine grid Cpts nodes
    I_F : {bsr_matrix}
        Identity operator on Fpts, i.e., the action of this matrix zeros
        out entries in a vector at all Cpts, leaving Fpts untouched
    I_C : {bsr_matrix}
        Identity operator on Cpts nodes, i.e., the action of this matrix zeros
        out entries in a vector at all Fpts, leaving Cpts untouched
    Cpts : {array}
        An array of all root node dofs, corresponding to the F/C splitting
    Fpts : {array}
        An array of all non root node dofs, corresponding to the F/C splitting

    Example
    -------
    >>> from numpy import array
    >>> from pyamg.util.utils import get_Cpt_params
    >>> from pyamg.gallery import poisson
    >>> from scipy.sparse import csr_matrix, bsr_matrix
    >>> A = poisson((10,), format='csr')
    >>> Cpts = array([3, 7])
    >>> AggOp = ([[ 1., 0.], [ 1., 0.],
    ...           [ 1., 0.], [ 1., 0.],
    ...           [ 1., 0.], [ 0., 1.],
    ...           [ 0., 1.], [ 0., 1.],
    ...           [ 0., 1.], [ 0., 1.]])
    >>> AggOp = csr_matrix(AggOp)
    >>> T = AggOp.copy().tobsr()
    >>> params = get_Cpt_params(A, Cpts, AggOp, T)
    >>> params['P_I'].todense()
    matrix([[ 0.,  0.],
            [ 0.,  0.],
            [ 0.,  0.],
            [ 1.,  0.],
            [ 0.,  0.],
            [ 0.,  0.],
            [ 0.,  0.],
            [ 0.,  1.],
            [ 0.,  0.],
            [ 0.,  0.]])

    Notes
    -----
    The principal calling routine is
    aggregation.smooth.energy_prolongation_smoother,
    which uses the Cpt_param dictionary for root-node style
    prolongation smoothing

    '''

    if not isspmatrix_bsr(A) and not isspmatrix_csr(A):
        raise TypeError('Expected BSR or CSR matrix A')
    if not isspmatrix_csr(AggOp):
        raise TypeError('Expected CSR matrix AggOp')
    if not isspmatrix_bsr(T):
        raise TypeError('Expected BSR matrix T')
    if T.blocksize[0] != T.blocksize[1]:
        raise TypeError('Expected square blocksize for BSR matrix T')
    if A.shape[0] != A.shape[1]:
        raise TypeError('Expected square matrix A')
    if T.shape[0] != A.shape[0]:
        raise TypeError('Expected compatible dimensions for T and A,\
                         T.shape[0] = A.shape[0]')
    if Cnodes.shape[0] != AggOp.shape[1]:
        if AggOp.shape[1] > 1:
            raise TypeError('Number of columns in AggOp must equal number\
                             of Cnodes')

    if isspmatrix_bsr(A) and A.blocksize[0] > 1:
        # Expand the list of Cpt nodes to a list of Cpt dofs
        blocksize = A.blocksize[0]
        Cpts = np.repeat(blocksize*Cnodes, blocksize)
        for k in range(1, blocksize):
            Cpts[list(range(k, Cpts.shape[0], blocksize))] += k
    else:
        blocksize = 1
        Cpts = Cnodes
    Cpts = np.array(Cpts, dtype=int)

    # More input checking
    if Cpts.shape[0] != T.shape[1]:
        if T.shape[1] > blocksize:
            raise ValueError('Expected number of Cpts to match T.shape[1]')
    if blocksize != T.blocksize[0]:
        raise ValueError('Expected identical blocksize in A and T')
    if AggOp.shape[0] != int(T.shape[0]/blocksize):
        raise ValueError('Number of rows in AggOp must equal number of\
                          fine-grid nodes')

    # Create two maps, one for F points and one for C points
    ncoarse = T.shape[1]
    I_C = eye(A.shape[0], A.shape[1], format='csr')
    I_F = I_C.copy()
    I_F.data[Cpts] = 0.0
    I_F.eliminate_zeros()
    I_C = I_C - I_F
    I_C.eliminate_zeros()

    # Find Fpts, the complement of Cpts
    Fpts = I_F.indices.copy()

    # P_I only injects from Cpts on the coarse grid to the fine grid, but
    # because of it's later uses, it must have the CSC indices ordered as
    # in Cpts
    if I_C.nnz > 0:
        indices = Cpts.copy()
        indptr = np.arange(indices.shape[0]+1)
    else:
        indices = np.zeros((0,), dtype=T.indices.dtype)
        indptr = np.zeros((ncoarse+1,), dtype=T.indptr.dtype)

    P_I = csc_matrix((I_C.data.copy(), indices, indptr),
                     shape=(I_C.shape[0], ncoarse))
    P_I = P_I.tobsr(T.blocksize)

    # Use same blocksize as A
    if isspmatrix_bsr(A):
        I_C = I_C.tobsr(A.blocksize)
        I_F = I_F.tobsr(A.blocksize)
    else:
        I_C = I_C.tobsr(blocksize=(1, 1))
        I_F = I_F.tobsr(blocksize=(1, 1))

    return {'P_I': P_I, 'I_F': I_F, 'I_C': I_C, 'Cpts': Cpts, 'Fpts': Fpts}


def compute_BtBinv(B, C):
    ''' Helper function that creates inv(B_i.T B_i) for each block row i in C,
        where B_i is B restricted to the sparsity pattern of block row i.

    Parameters
    ----------
    B : {array}
        (M,k) array, typically near-nullspace modes for coarse grid, i.e., B_c.
    C : {csr_matrix, bsr_matrix}
        Sparse NxM matrix, whose sparsity structure (i.e., matrix graph)
        is used to determine BtBinv.

    Returns
    -------
    BtBinv : {array}
        BtBinv[i] = inv(B_i.T B_i), where B_i is B restricted to the nonzero
        pattern of block row i in C.

    Example
    -------
    >>> from numpy import array
    >>> from scipy.sparse import bsr_matrix
    >>> from pyamg.util.utils import compute_BtBinv
    >>> T = array([[ 1.,  0.],
    ...            [ 1.,  0.],
    ...            [ 0.,  .5],
    ...            [ 0.,  .25]])
    >>> T = bsr_matrix(T)
    >>> B = array([[1.],[2.]])
    >>> compute_BtBinv(B, T)
    array([[[ 1.  ]],
    <BLANKLINE>
           [[ 1.  ]],
    <BLANKLINE>
           [[ 0.25]],
    <BLANKLINE>
           [[ 0.25]]])

    Notes
    -----
    The principal calling routines are
    aggregation.smooth.energy_prolongation_smoother, and
    util.utils.filter_operator.

    BtBinv is used in the prolongation smoothing process that incorporates B
    into the span of prolongation with row-wise projection operators.  It is
    these projection operators that BtBinv is part of.

    '''

    if not isspmatrix_bsr(C) and not isspmatrix_csr(C):
        raise TypeError('Expected bsr_matrix or csr_matrix for C')
    if C.shape[1] != B.shape[0]:
        raise TypeError('Expected matching dimensions such that C*B')

    # Problem parameters
    if isspmatrix_bsr(C):
        ColsPerBlock = C.blocksize[1]
        RowsPerBlock = C.blocksize[0]
    else:
        ColsPerBlock = 1
        RowsPerBlock = 1
    Ncoarse = C.shape[1]
    Nfine = C.shape[0]
    NullDim = B.shape[1]
    Nnodes = int(Nfine/RowsPerBlock)

    # Construct BtB
    BtBinv = np.zeros((Nnodes, NullDim, NullDim), dtype=B.dtype)
    BsqCols = sum(range(NullDim+1))
    Bsq = np.zeros((Ncoarse, BsqCols), dtype=B.dtype)
    counter = 0
    for i in range(NullDim):
        for j in range(i, NullDim):
            Bsq[:, counter] = np.conjugate(np.ravel(np.asarray(B[:, i]))) * \
                np.ravel(np.asarray(B[:, j]))
            counter = counter + 1
    # This specialized C-routine calculates (B.T B) for each row using Bsq
    pyamg.amg_core.calc_BtB(NullDim, Nnodes, ColsPerBlock,
                            np.ravel(np.asarray(Bsq)),
                            BsqCols, np.ravel(np.asarray(BtBinv)),
                            C.indptr, C.indices)

    # Invert each block of BtBinv, noting that amg_core.calc_BtB(...) returns
    # values in column-major form, thus necessitating the deep transpose
    #   This is the old call to a specialized routine, but lacks robustness
    #   pyamg.amg_core.pinv_array(np.ravel(BtBinv), Nnodes, NullDim, 'F')
    BtBinv = BtBinv.transpose((0, 2, 1)).copy()
    pinv_array(BtBinv)

    return BtBinv


def eliminate_diag_dom_nodes(A, C, theta=1.02):
    ''' Helper function that eliminates diagonally dominant rows and cols from A
    in the separate matrix C.  This is useful because it eliminates nodes in C
    which we don't want coarsened.  These eliminated nodes in C just become
    the rows and columns of the identity.

    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix
    C : {csr_matrix}
        Sparse MxM matrix, where M is the number of nodes in A.  M=N if A
        is CSR or is BSR with blocksize 1.  Otherwise M = N/blocksize.
    theta : {float}
        determines diagonal dominance threshhold

    Returns
    -------
    C : {csr_matrix}
        C updated such that the rows and columns corresponding to diagonally
        dominant rows in A have been eliminated and replaced with rows and
        columns of the identity.

    Notes
    -----
    Diagonal dominance is defined as
     || (e_i, A) - a_ii ||_1  <  theta a_ii
    that is, the 1-norm of the off diagonal elements in row i must be less than
    theta times the diagonal element.


    Example
    -------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.utils import eliminate_diag_dom_nodes
    >>> A = poisson( (4,), format='csr' )
    >>> C = eliminate_diag_dom_nodes(A, A.copy(), 1.1)
    >>> C.todense()
    matrix([[ 1.,  0.,  0.,  0.],
            [ 0.,  2., -1.,  0.],
            [ 0., -1.,  2.,  0.],
            [ 0.,  0.,  0.,  1.]])

    '''

    # Find the diagonally dominant rows in A.
    A_abs = A.copy()
    A_abs.data = np.abs(A_abs.data)
    D_abs = get_diagonal(A_abs, norm_eq=0, inv=False)
    diag_dom_rows = (D_abs > (theta*(A_abs*np.ones((A_abs.shape[0],),
                     dtype=A_abs) - D_abs)))

    # Account for BSR matrices and translate diag_dom_rows from dofs to nodes
    bsize = blocksize(A_abs)
    if bsize > 1:
        diag_dom_rows = np.array(diag_dom_rows, dtype=int)
        diag_dom_rows = diag_dom_rows.reshape(-1, bsize)
        diag_dom_rows = np.sum(diag_dom_rows, axis=1)
        diag_dom_rows = (diag_dom_rows == bsize)

    # Replace these rows/cols in # C with rows/cols of the identity.
    Id = eye(C.shape[0], C.shape[1], format='csr')
    Id.data[diag_dom_rows] = 0.0
    C = Id * C * Id
    Id.data[diag_dom_rows] = 1.0
    Id.data[np.where(diag_dom_rows == 0)[0]] = 0.0
    C = C + Id

    del A_abs
    return C


def remove_diagonal(S):
    """ Removes the diagonal of the matrix S

    Parameters
    ----------
    S : csr_matrix
        Square matrix

    Returns
    -------
    S : csr_matrix
        Strength matrix with the diagonal removed

    Notes
    -----
    This is needed by all the splitting routines which operate on matrix graphs
    with an assumed zero diagonal


    Example
    -------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.utils import remove_diagonal
    >>> A = poisson( (4,), format='csr' )
    >>> C = remove_diagonal(A)
    >>> C.todense()
    matrix([[ 0., -1.,  0.,  0.],
            [-1.,  0., -1.,  0.],
            [ 0., -1.,  0., -1.],
            [ 0.,  0., -1.,  0.]])

    """

    if not isspmatrix_csr(S):
        raise TypeError('expected csr_matrix')

    if S.shape[0] != S.shape[1]:
        raise ValueError('expected square matrix, shape=%s' % (S.shape,))

    S = coo_matrix(S)
    mask = S.row != S.col
    S.row = S.row[mask]
    S.col = S.col[mask]
    S.data = S.data[mask]

    return S.tocsr()


def scale_rows_by_largest_entry(S):
    """ Scale each row in S by it's largest in magnitude entry

    Parameters
    ----------
    S : csr_matrix

    Returns
    -------
    S : csr_matrix
        Each row has been scaled by it's largest in magnitude entry

    Example
    -------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.utils import scale_rows_by_largest_entry
    >>> A = poisson( (4,), format='csr' )
    >>> A.data[1] = 5.0
    >>> A = scale_rows_by_largest_entry(A)
    >>> A.todense()
    matrix([[ 0.4,  1. ,  0. ,  0. ],
            [-0.5,  1. , -0.5,  0. ],
            [ 0. , -0.5,  1. , -0.5],
            [ 0. ,  0. , -0.5,  1. ]])

    """

    if not isspmatrix_csr(S):
        raise TypeError('expected csr_matrix')

    # Scale S by the largest magnitude entry in each row
    largest_row_entry = np.zeros((S.shape[0],), dtype=S.dtype)
    pyamg.amg_core.maximum_row_value(S.shape[0], largest_row_entry,
                                     S.indptr, S.indices, S.data)

    largest_row_entry[largest_row_entry != 0] =\
        1.0 / largest_row_entry[largest_row_entry != 0]
    S = scale_rows(S, largest_row_entry, copy=True)

    return S


def levelize_strength_or_aggregation(to_levelize, max_levels, max_coarse):
    """
    Helper function to preprocess the strength and aggregation parameters
    passed to smoothed_aggregation_solver and rootnode_solver.

    Parameters
    ----------
    to_levelize : {string, tuple, list}
        Parameter to preprocess, i.e., levelize and convert to a level-by-level
        list such that entry i specifies the parameter at level i
    max_levels : int
        Defines the maximum number of levels considered
    max_coarse : int
        Defines the maximum coarse grid size allowed

    Returns
    -------
    (max_levels, max_coarse, to_levelize) : tuple
        New max_levels and max_coarse values and then the parameter list
        to_levelize, such that entry i specifies the parameter choice at level
        i.  max_levels and max_coarse are returned, because they may be updated
        if strength or aggregation set a predefined coarsening and possibly
        change these values.

    Notes
    --------
    This routine is needed because the user will pass in a parameter option
    such as smooth='jacobi', or smooth=['jacobi', None], and this option must
    be "levelized", or converted to a list of length max_levels such that entry
    [i] in that list is the parameter choice for level i.

    The parameter choice in to_levelize can be a string, tuple or list.  If
    it is a string or tuple, then that option is assumed to be the
    parameter setting at every level.  If to_levelize is inititally a list,
    if the length of the list is less than max_levels, the last entry in the
    list defines that parameter for all subsequent levels.


    Examples
    --------
    >>> from pyamg.util.utils import levelize_strength_or_aggregation
    >>> strength = ['evolution', 'classical']
    >>> levelize_strength_or_aggregation(strength, 4, 10)
    (4, 10, ['evolution', 'classical', 'classical'])

    """

    if isinstance(to_levelize, tuple):
        if to_levelize[0] == 'predefined':
            to_levelize = [to_levelize]
            max_levels = 2
            max_coarse = 0
        else:
            to_levelize = [to_levelize for i in range(max_levels-1)]

    elif isinstance(to_levelize, str):
        if to_levelize == 'predefined':
            raise ValueError('predefined to_levelize requires a user-provided\
                              CSR matrix representing strength or aggregation\
                              i.e., (\'predefined\', {\'C\' : CSR_MAT}).')
        else:
            to_levelize = [to_levelize for i in range(max_levels-1)]

    elif isinstance(to_levelize, list):
        if isinstance(to_levelize[-1], tuple) and\
           (to_levelize[-1][0] == 'predefined'):
            # to_levelize is a list that ends with a predefined operator
            max_levels = len(to_levelize) + 1
            max_coarse = 0
        else:
            # to_levelize a list that __doesn't__ end with 'predefined'
            if len(to_levelize) < max_levels-1:
                mlz = max_levels - 1 - len(to_levelize)
                toext = [to_levelize[-1] for i in range(mlz)]
                to_levelize.extend(toext)

    elif to_levelize is None:
        to_levelize = [(None, {}) for i in range(max_levels-1)]
    else:
        raise ValueError('invalid to_levelize')

    return max_levels, max_coarse, to_levelize


def levelize_smooth_or_improve_candidates(to_levelize, max_levels):
    """
    Helper function to preprocess the smooth and improve_candidates
    parameters passed to smoothed_aggregation_solver and rootnode_solver.

    Parameters
    ----------
    to_levelize : {string, tuple, list}
        Parameter to preprocess, i.e., levelize and convert to a level-by-level
        list such that entry i specifies the parameter at level i
    max_levels : int
        Defines the maximum number of levels considered

    Returns
    -------
    to_levelize : list
        The parameter list such that entry i specifies the parameter choice
        at level i.

    Notes
    --------
    This routine is needed because the user will pass in a parameter option
    such as smooth='jacobi', or smooth=['jacobi', None], and this option must
    be "levelized", or converted to a list of length max_levels such that entry
    [i] in that list is the parameter choice for level i.

    The parameter choice in to_levelize can be a string, tuple or list.  If
    it is a string or tuple, then that option is assumed to be the
    parameter setting at every level.  If to_levelize is inititally a list,
    if the length of the list is less than max_levels, the last entry in the
    list defines that parameter for all subsequent levels.

    Examples
    --------
    >>> from pyamg.util.utils import levelize_smooth_or_improve_candidates
    >>> improve_candidates = ['gauss_seidel', None]
    >>> levelize_smooth_or_improve_candidates(improve_candidates, 4)
    ['gauss_seidel', None, None, None]
    """

    if isinstance(to_levelize, tuple) or isinstance(to_levelize, str):
        to_levelize = [to_levelize for i in range(max_levels)]
    elif isinstance(to_levelize, list):
        if len(to_levelize) < max_levels:
            mlz = max_levels - len(to_levelize)
            toext = [to_levelize[-1] for i in range(mlz)]
            to_levelize.extend(toext)
    elif to_levelize is None:
        to_levelize = [(None, {}) for i in range(max_levels)]

    return to_levelize


def filter_matrix_columns(A, theta):
    """
    Filter each column of A with tol, i.e., drop all entries in column k where
        abs(A[i,k]) < tol max( abs(A[:,k]) )

    Parameters
    ----------
    A : sparse_matrix

    theta : float
        In range [0,1) and defines drop-tolerance used to filter the columns
        of A

    Returns
    -------
    A_filter : sparse_matrix
        Each column has been filtered by dropping all entries where
        abs(A[i,k]) < tol max( abs(A[:,k]) )

    Example
    -------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.utils import filter_matrix_columns
    >>> from scipy import array
    >>> from scipy.sparse import csr_matrix
    >>> A = csr_matrix( array([[ 0.24,  1.  ,  0.  ],
    ...                        [-0.5 ,  1.  , -0.5 ],
    ...                        [ 0.  ,  0.49,  1.  ],
    ...                        [ 0.  ,  0.  , -0.5 ]]) )
    >>> filter_matrix_columns(A, 0.5).todense()
    matrix([[ 0. ,  1. ,  0. ],
            [-0.5,  1. , -0.5],
            [ 0. ,  0. ,  1. ],
            [ 0. ,  0. , -0.5]])

    """
    if not isspmatrix(A):
        raise ValueError("Sparse matrix input needed")
    if isspmatrix_bsr(A):
        blocksize = A.blocksize
    Aformat = A.format

    if (theta < 0) or (theta >= 1.0):
        raise ValueError("theta must be in [0,1)")

    # Apply drop-tolerance to each column of A, which is most easily
    # accessed by converting to CSC.  We apply the drop-tolerance with
    # amg_core.classical_strength_of_connection(), which ignores
    # diagonal entries, thus necessitating the trick where we add
    # A.shape[1] to each of the column indices
    A = A.copy().tocsc()
    A_filter = A.copy()
    A.indices += A.shape[1]
    A_filter.indices += A.shape[1]
    # classical_strength_of_connection takes an absolute value internally
    pyamg.amg_core.classical_strength_of_connection_abs(A.shape[1], theta,
                                                        A.indptr, A.indices,
                                                        A.data, A_filter.indptr,
                                                        A_filter.indices,
                                                        A_filter.data)
    A_filter.indices[:A_filter.indptr[-1]] -= A_filter.shape[1]
    A_filter = csc_matrix((A_filter.data[:A_filter.indptr[-1]],
                           A_filter.indices[:A_filter.indptr[-1]],
                           A_filter.indptr), shape=A_filter.shape)
    del A

    if Aformat == 'bsr':
        A_filter = A_filter.tobsr(blocksize)
    else:
        A_filter = A_filter.asformat(Aformat)

    return A_filter


def filter_matrix_rows(A, theta):
    """
    Filter each row of A with tol, i.e., drop all entries in row k where
        abs(A[i,k]) < tol max( abs(A[:,k]) )

    Parameters
    ----------
    A : sparse_matrix

    theta : float
        In range [0,1) and defines drop-tolerance used to filter the row of A

    Returns
    -------
    A_filter : sparse_matrix
        Each row has been filtered by dropping all entries where
        abs(A[i,k]) < tol max( abs(A[:,k]) )

    Example
    -------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.utils import filter_matrix_rows
    >>> from scipy import array
    >>> from scipy.sparse import csr_matrix
    >>> A = csr_matrix( array([[ 0.24, -0.5 ,  0.  ,  0.  ],
    ...                        [ 1.  ,  1.  ,  0.49,  0.  ],
    ...                        [ 0.  , -0.5 ,  1.  , -0.5 ]])  )
    >>> filter_matrix_rows(A, 0.5).todense()
    matrix([[ 0. , -0.5,  0. ,  0. ],
            [ 1. ,  1. ,  0. ,  0. ],
            [ 0. , -0.5,  1. , -0.5]])

    """
    if not isspmatrix(A):
        raise ValueError("Sparse matrix input needed")
    if isspmatrix_bsr(A):
        blocksize = A.blocksize
    Aformat = A.format
    A = A.tocsr()

    if (theta < 0) or (theta >= 1.0):
        raise ValueError("theta must be in [0,1)")

    # Apply drop-tolerance to each row of A.  We apply the drop-tolerance with
    # amg_core.classical_strength_of_connection(), which ignores diagonal
    # entries, thus necessitating the trick where we add A.shape[0] to each of
    # the row indices
    A_filter = A.copy()
    A.indices += A.shape[0]
    A_filter.indices += A.shape[0]
    # classical_strength_of_connection takes an absolute value internally
    pyamg.amg_core.classical_strength_of_connection_abs(A.shape[0], theta,
                                                        A.indptr, A.indices,
                                                        A.data, A_filter.indptr,
                                                        A_filter.indices,
                                                        A_filter.data)
    A_filter.indices[:A_filter.indptr[-1]] -= A_filter.shape[0]
    A_filter = csr_matrix((A_filter.data[:A_filter.indptr[-1]],
                           A_filter.indices[:A_filter.indptr[-1]],
                           A_filter.indptr), shape=A_filter.shape)

    if Aformat == 'bsr':
        A_filter = A_filter.tobsr(blocksize)
    else:
        A_filter = A_filter.asformat(Aformat)

    A.indices -= A.shape[0]
    return A_filter


def truncate_rows(A, nz_per_row):
    """
    Truncate the rows of A by keeping only the largest in magnitude entries in
    each row.

    Parameters
    ----------
    A : sparse_matrix

    nz_per_row : int
        Determines how many entries in each row to keep

    Returns
    -------
    A : sparse_matrix
        Each row has been truncated to at most nz_per_row entries

    Example
    -------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.utils import truncate_rows
    >>> from scipy import array
    >>> from scipy.sparse import csr_matrix
    >>> A = csr_matrix( array([[-0.24, -0.5 ,  0.  ,  0.  ],
    ...                        [ 1.  , -1.1 ,  0.49,  0.1 ],
    ...                        [ 0.  ,  0.4 ,  1.  ,  0.5 ]])  )
    >>> truncate_rows(A, 2).todense()
    matrix([[-0.24, -0.5 ,  0.  ,  0.  ],
            [ 1.  , -1.1 ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  1.  ,  0.5 ]])

    """
    if not isspmatrix(A):
        raise ValueError("Sparse matrix input needed")
    if isspmatrix_bsr(A):
        blocksize = A.blocksize
    if isspmatrix_csr(A):
        A = A.copy()    # don't modify A in-place
    Aformat = A.format
    A = A.tocsr()
    nz_per_row = int(nz_per_row)

    # Truncate rows of A, and then convert A back to original format
    pyamg.amg_core.truncate_rows_csr(A.shape[0], nz_per_row, A.indptr,
                                     A.indices, A.data)

    A.eliminate_zeros()
    if Aformat == 'bsr':
        A = A.tobsr(blocksize)
    else:
        A = A.asformat(Aformat)

    return A


# from functools import partial, update_wrapper
# def dispatcher(name_to_handle):
#    def dispatcher(arg):
#        if isinstance(arg,tuple):
#            fn,opts = arg[0],arg[1]
#        else:
#            fn,opts = arg,{}
#
#        if fn in name_to_handle:
#            # convert string into function handle
#            fn = name_to_handle[fn]
#        #elif isinstance(fn, type(numpy.ones)):
#        #    pass
#        elif callable(fn):
#            # if fn is itself a function handle
#            pass
#        else:
#            raise TypeError('Expected function')
#
#        wrapped = partial(fn, **opts)
#        update_wrapper(wrapped, fn)
#
#        return wrapped
#
#    return dispatcher
