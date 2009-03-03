"""General utility functions for pyamg"""

__docformat__ = "restructuredtext en"

from warnings import warn

import numpy
import scipy
from scipy.sparse import isspmatrix, isspmatrix_csr, isspmatrix_csc, \
        isspmatrix_bsr, csr_matrix, csc_matrix, bsr_matrix, coo_matrix
from scipy.sparse.sputils import upcast
from pyamg.util.linalg import norm, cond
from scipy.linalg import eigvals

__all__ = ['diag_sparse', 'profile_solver', 'to_type', 'type_prep', 
           'get_diagonal', 'UnAmal', 'Coord2RBM', 'hierarchy_spectrum',
           'print_table']

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
    >>> import numpy
    >>> from scipy.sparse import spdiags
    >>> from scipy.sparse.linalg import cg
    >>> from pyamg.classical import ruge_stuben_solver
    >>> from pyamg.util.utils import profile_solver
    >>> n=100
    >>> e = numpy.ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n)
    >>> b = A*numpy.ones(A.shape[0])
    >>> ml = ruge_stuben_solver(A, max_coarse=10)
    >>> res = profile_solver(ml,accel=cg)
 
    """
    A = ml.levels[0].A
    b = A * scipy.rand(A.shape[0],1)
    residuals = []

    if accel is None:
        x_sol = ml.solve(b, residuals=residuals, **kwargs)
    else:
        def callback(x):
            residuals.append( norm(numpy.ravel(b) - numpy.ravel(A*x)) )
        M = ml.aspreconditioner(cycle=kwargs.get('cycle','V'))
        accel(A, b, M=M, callback=callback, **kwargs)

    return numpy.asarray(residuals)

def diag_sparse(A):
    """
    If A is a sparse matrix (e.g. csr_matrix or csc_matrix)
       - return the diagonal of A as an array

    Otherwise
       - return a csr_matrix with A on the diagonal

    Parameters
    ----------
    A : sparse matrix or rank 1 array
        General sparse matrix or array of diagonal entries

    Returns
    -------
    B : array or sparse matrix
        Diagonal sparse is returned as csr if A is dense otherwise return an
        array of the diagonal

    Examples
    --------
    >>> import numpy
    >>> from pyamg.util.utils import diag_sparse
    >>> d = 2.0*numpy.ones((3,)).ravel()
    >>> print diag_sparse(d).todense()
    [[ 2.  0.  0.]
     [ 0.  2.  0.]
     [ 0.  0.  2.]]

    """
    if isspmatrix(A):
        return A.diagonal()
    else:
        if(numpy.rank(A)!=1):
            raise ValueError,'input diagonal array expected to be rank 1'
        return csr_matrix((numpy.asarray(A),numpy.arange(len(A)),numpy.arange(len(A)+1)),(len(A),len(A)))

def scale_rows(A,v,copy=True):
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
    scipy.sparse.sparsetools.csr_scale_rows, scale_columns

    Notes
    -----
    - if A is a csc_matrix, the transpose A.T is passed to scale_columns
    - if A is not csr, csc, or bsr, it is converted to csr and sent to scale_rows

    Examples
    --------
    >>> import numpy
    >>> from scipy.sparse import spdiags
    >>> from pyamg.util.utils import scale_rows
    >>> n=5
    >>> e = numpy.ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n-1).tocsr()
    >>> B = scale_rows(A,5*numpy.ones((A.shape[0],1)))
    """
    from scipy.sparse.sparsetools import csr_scale_rows, bsr_scale_rows

    v = numpy.ravel(v)

    if isspmatrix_csr(A) or isspmatrix_bsr(A):
        M,N = A.shape
        if M != len(v):
            raise ValueError,'scale vector has incompatible shape'

        if copy:
            A = A.copy()
            A.data = numpy.asarray(A.data,dtype=upcast(A.dtype,v.dtype))
        else:
            v = numpy.asarray(v,dtype=A.dtype)

        if isspmatrix_csr(A):
            csr_scale_rows(M, N, A.indptr, A.indices, A.data, v)
        else:
            R,C = A.blocksize
            bsr_scale_rows(M/R, N/C, R, C, A.indptr, A.indices, numpy.ravel(A.data), v)

        return A
    elif isspmatrix_csc(A):
        return scale_columns(A.T,v)
    else:
        return scale_rows(csr_matrix(A),v)
        
def scale_columns(A,v,copy=True):
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
    scipy.sparse.sparsetools.csr_scale_columns, scale_rows

    Notes
    -----
    - if A is a csc_matrix, the transpose A.T is passed to scale_rows
    - if A is not csr, csc, or bsr, it is converted to csr and sent to scale_rows

    Examples
    --------
    >>> import numpy
    >>> from scipy.sparse import spdiags
    >>> from pyamg.util.utils import scale_columns
    >>> n=5
    >>> e = numpy.ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n-1).tocsr()
    >>> print scale_columns(A,5*numpy.ones((A.shape[1],1))).todense()
    [[ 10.  -5.   0.   0.]
     [ -5.  10.  -5.   0.]
     [  0.  -5.  10.  -5.]
     [  0.   0.  -5.  10.]
     [  0.   0.   0.  -5.]]

    """
    from scipy.sparse.sparsetools import csr_scale_columns, bsr_scale_columns

    v = numpy.ravel(v)

    if isspmatrix_csr(A) or isspmatrix_bsr(A):
        M,N = A.shape
        if N != len(v):
            raise ValueError,'scale vector has incompatible shape'

        if copy:
            A = A.copy()
            A.data = numpy.asarray(A.data,dtype=upcast(A.dtype,v.dtype))
        else:
            v = numpy.asarray(v,dtype=A.dtype)

        if isspmatrix_csr(A):
            csr_scale_columns(M, N, A.indptr, A.indices, A.data, v)
        else:
            R,C = A.blocksize
            bsr_scale_columns(M/R, N/C, R, C, A.indptr, A.indices, numpy.ravel(A.data), v)

        return A
    elif isspmatrix_csc(A):
        return scale_rows(A.T,v)
    else:
        return scale_rows(csr_matrix(A),v)

def symmetric_rescaling(A,copy=True):
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
    >>> import numpy
    >>> from scipy.sparse import spdiags
    >>> from pyamg.util.utils import symmetric_rescaling
    >>> n=5
    >>> e = numpy.ones((n,1)).ravel()
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
            raise ValueError,'expected square matrix'

        D = diag_sparse(A)
        mask = D == 0

        if A.dtype != complex:
            D_sqrt = numpy.sqrt(abs(D))
        else:
            # We can take square roots of negative numbers
            D_sqrt = numpy.sqrt(D)
        
        D_sqrt_inv = 1.0/D_sqrt
        D_sqrt_inv[mask] = 0

        DAD = scale_rows(A,D_sqrt_inv,copy=copy)
        DAD = scale_columns(DAD,D_sqrt_inv,copy=False)

        return D_sqrt,D_sqrt_inv,DAD

    else:
        return symmetric_rescaling(csr_matrix(A))


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
    Returns upcasted varlist to upcast_type

    Notes
    -----
    Useful when harmonizing the types of variables, such as 
    if A and b are complex, but x,y and z are not.

    Examples
    --------
    >>> import numpy
    >>> from pyamg.util.utils import type_prep 
    >>> from scipy.sparse.sputils import upcast
    >>> x = numpy.ones((5,1))
    >>> y = 2.0j*numpy.ones((5,1))
    >>> z = 2.3
    >>> varlist = type_prep(upcast(x.dtype, y.dtype), [x, y, z])

    """
    varlist = to_type(upcast_type, varlist)
    for i in range(len(varlist)):
        if numpy.isscalar(varlist[i]):
            varlist[i] = numpy.array([varlist[i]])
    
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
    Returns upcasted varlist to upcast_type

    Notes
    -----
    Useful when harmonizing the types of variables, such as 
    if A and b are complex, but x,y and z are not.

    Examples
    --------
    >>> import numpy
    >>> from pyamg.util.utils import to_type  
    >>> from scipy.sparse.sputils import upcast
    >>> x = numpy.ones((5,1))
    >>> y = 2.0j*numpy.ones((5,1))
    >>> varlist = to_type(upcast(x.dtype, y.dtype), [x, y])

    """

    #convert_type = type(numpy.array([0], upcast_type)[0])
        
    for i in range(len(varlist)):

        # convert scalars to complex
        if numpy.isscalar(varlist[i]):
            varlist[i] = numpy.array([varlist[i]], upcast_type)[0]
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
    
    #if not isspmatrix(A):
    if not (isspmatrix_csr(A) or isspmatrix_csc(A) or isspmatrix_bsr(A)):
        warn('Implicit conversion to sparse matrix')
        A = csr_matrix(A)
    
    # critical to sort the indices of A
    A.sort_indices()
    if norm_eq == 1:
        # This transpose involves almost no work, use csr data structures as csc, or vice versa
        At = A.T    
        D = (At.multiply(At.conjugate()))*numpy.ones((At.shape[0],))
    elif norm_eq == 2:    
        D = (A.multiply(A.conjugate()))*numpy.ones((A.shape[0],))
    else:
        D = A.diagonal()
        
    if inv:
        Dinv = 1.0 / D
        Dinv[D == 0] = 0.0
        return Dinv
    else:
        return D

def UnAmal(A, RowsPerBlock, ColsPerBlock):
    """
    Unamalgamate a CSR A with blocks of 1's.  

    Equivalent to Kronecker_Product(A, ones(RowsPerBlock, ColsPerBlock))

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
    A_UnAmal : bsr_matrix 
        Similar to a Kronecker product of A and ones(RowsPerBlock, ColsPerBlock)

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
    data = numpy.ones( (A.indices.shape[0], RowsPerBlock, ColsPerBlock) )
    return bsr_matrix((data, A.indices, A.indptr), shape=(RowsPerBlock*A.shape[0], ColsPerBlock*A.shape[1]) )

def print_table(table, title='', delim='|', centering='center', col_padding=2, header=True, headerchar='-'):
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
    with extra paddding added into each column entry to ensure columns align.
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
    if type(table) == type( (2,2) ):
        title = table[0]
        table = table[1]

    # Calculate each column's width
    colwidths=[]
    for i in range(len(table)):
        # extend colwidths for row i
        for k in range( len(table[i]) - len(colwidths) ):
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
        table_str += str.center(title, ttwidth) + '\n'

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
        for elmt,elmtwidth in zip(table[0],colwidths):
            table_str += centering(str(elmt), elmtwidth) + delim 
        if table[0] != []:
            table_str = table_str[:-len(delim)] + '\n'

        # Append Header Separator
        #                Total Column Width            Total Col Delimiter Widths
        if len(headerchar) == 0:
            headerchard = ' '
        table_str += headerchar*int(scipy.ceil(float(ttwidth)/float(len(headerchar)))) + '\n'
        
        table = table[1:]

    for row in table:
        for elmt,elmtwidth in zip(row,colwidths):
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
        e.g. generated with smoothed_aggregation_solver(...) or ruge_stuben_solver(...)

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
     Level | min(re(eig)) | max(re(eig)) | num re(eig) < 0 | num re(eig) > 0 | cond_2(A) 
    -------------------------------------------------------------------------------------
       0   |    2.000     |    2.000     |        0        |        1        |  1.00e+00 
    <BLANKLINE>
    <BLANKLINE>
     Level | min(im(eig)) | max(im(eig)) | num im(eig) < 0 | num im(eig) > 0 | cond_2(A) 
    -------------------------------------------------------------------------------------
       0   |    0.000     |    0.000     |        0        |        0        |  1.00e+00 
    <BLANKLINE>


    """
    
    real_table = [ ['Level', 'min(re(eig))', 'max(re(eig))', 'num re(eig) < 0', 'num re(eig) > 0', 'cond_2(A)'] ]
    imag_table = [ ['Level', 'min(im(eig))', 'max(im(eig))', 'num im(eig) < 0', 'num im(eig) > 0', 'cond_2(A)'] ]

    for i in range(len(mg.levels)):
        A = mg.levels[i].A.tocsr()

        if filter == True:
            # Filter out any zero rows and columns of A
            A.eliminate_zeros()
            nnz_per_row = A.indptr[0:-1] - A.indptr[1:]
            nonzero_rows = (nnz_per_row != 0).nonzero()[0]
            A = A.tocsc()
            nnz_per_col = A.indptr[0:-1] - A.indptr[1:]
            nonzero_cols = (nnz_per_col != 0).nonzero()[0]
            nonzero_rowcols = scipy.union1d(nonzero_rows, nonzero_cols)
            A = numpy.mat(A.todense())
            A = A[nonzero_rowcols,:][:,nonzero_rowcols]
        else:
            A = numpy.mat(A.todense())

        e = eigvals(A)
        c = cond(A)
        lambda_min = min(scipy.real(e))
        lambda_max = max(scipy.real(e))
        num_neg = max(e[scipy.real(e) < 0.0].shape)
        num_pos = max(e[scipy.real(e) > 0.0].shape)
        real_table.append([str(i), ('%1.3f' % lambda_min), ('%1.3f' % lambda_max), str(num_neg), str(num_pos), ('%1.2e' % c)])
        
        lambda_min = min(scipy.imag(e))
        lambda_max = max(scipy.imag(e))
        num_neg = max(e[scipy.imag(e) < 0.0].shape)
        num_pos = max(e[scipy.imag(e) > 0.0].shape)
        imag_table.append([str(i), ('%1.3f' % lambda_min), ('%1.3f' % lambda_max), str(num_neg), str(num_pos), ('%1.2e' % c)])

        if plot:
            import pylab
            pylab.figure(i+1)
            pylab.plot(scipy.real(e), scipy.imag(e), 'kx')
            handle = pylab.title('Level %d Spectrum' % i)
            handle.set_fontsize(19)
            handle = pylab.xlabel('real(eig)')
            handle.set_fontsize(17)
            handle = pylab.ylabel('imag(eig)')
            handle.set_fontsize(17)

    print print_table(real_table)
    print print_table(imag_table)

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
    >>> import numpy
    >>> from pyamg.util.utils import Coord2RBM
    >>> a = numpy.array([0,1,2]) 
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

    #check inputs
    if(numPDEs == 1):
        numcols = 1
    elif( (numPDEs == 3) or (numPDEs == 6) ):
        numcols = 6
    else:
        raise ValueError("Coord2RBM(...) only supports 1, 3 or 6 PDEs per spatial location, i.e. numPDEs = [1 | 3 | 6].  You've entered " \
                + str(numPDEs) + "." )

    if( (max(x.shape) != numNodes) or (max(y.shape) != numNodes) or (max(z.shape) != numNodes) ):
        raise ValueError("Coord2RBM(...) requires coordinate vectors of equal length.  Length must be numNodes = " + str(numNodes)) 

    #if( (min(x.shape) != 1) or (min(y.shape) != 1) or (min(z.shape) != 1) ):
    #    raise ValueError("Coord2RBM(...) requires coordinate vectors that are (numNodes x 1) or (1 x numNodes).") 


    #preallocate rbm
    rbm = numpy.mat(numpy.zeros((numNodes*numPDEs, numcols)))
    
    for node in range(numNodes):
        dof = node*numPDEs

        if(numPDEs == 1):
            rbm[node] = 1.0 
                
        if(numPDEs == 6): 
            for ii in range(3,6):        #lower half = [ 0 I ]
                for jj in range(0,6):
                    if(ii == jj):
                        rbm[dof+ii, jj] = 1.0 
                    else: 
                        rbm[dof+ii, jj] = 0.0

        if((numPDEs == 3) or (numPDEs == 6) ): 
            for ii in range(0,3):        #upper left = [ I ]
                for jj in range(0,3):
                    if(ii == jj):
                        rbm[dof+ii, jj] = 1.0 
                    else: 
                        rbm[dof+ii, jj] = 0.0

            for ii in range(0,3):        #upper right = [ Q ]
                for jj in range(3,6):
                    if( ii == (jj-3) ):
                        rbm[dof+ii, jj] = 0.0
                    else:
                        if( (ii+jj) == 4):
                            rbm[dof+ii, jj] = z[node]
                        elif( (ii+jj) == 5 ): 
                            rbm[dof+ii, jj] = y[node]
                        elif( (ii+jj) == 6 ): 
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


#from functools import partial, update_wrapper
#def dispatcher(name_to_handle):
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

