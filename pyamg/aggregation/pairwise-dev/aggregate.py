"""Aggregation methods"""

__docformat__ = "restructuredtext en"

import pdb

from warnings import warn
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_csr, isspmatrix_csc
from pyamg import amg_core
from pyamg.graph import lloyd_cluster
from pyamg.util.utils import relaxation_as_linear_operator, mat_mat_complexity

__all__ = ['standard_aggregation', 'naive_aggregation', 'lloyd_aggregation',
           'notay_pairwise', 'weighted_matching']


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
    Cpts : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

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
    >>> standard_aggregation(A)[0].todense() # two aggregates
    matrix([[1, 0],
            [1, 0],
            [0, 1],
            [0, 1]], dtype=int8)
    >>> A = csr_matrix([[1,0,0],[0,1,1],[0,1,1]])
    >>> A.todense()                      # first vertex is isolated
    matrix([[1, 0, 0],
            [0, 1, 1],
            [0, 1, 1]])
    >>> standard_aggregation(A)[0].todense() # one aggregate
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
    num_rows = C.shape[0]

    Tj = np.empty(num_rows, dtype=index_type)  # stores the aggregate #s
    Cpts = np.empty(num_rows, dtype=index_type)  # stores the Cpts

    fn = amg_core.standard_aggregation

    num_aggregates = fn(num_rows, C.indptr, C.indices, Tj, Cpts)
    Cpts = Cpts[:num_aggregates]

    if num_aggregates == 0:
        # return all zero matrix and no Cpts
        return csr_matrix((num_rows, 1), dtype='int8'),\
            np.array([], dtype=index_type)
    else:
        shape = (num_rows, num_aggregates)
        if Tj.min() == -1:
            # some nodes not aggregated
            mask = Tj != -1
            row = np.arange(num_rows, dtype=index_type)[mask]
            col = Tj[mask]
            data = np.ones(len(col), dtype='int8')
            return coo_matrix((data, (row, col)), shape=shape).tocsr(), Cpts
        else:
            # all nodes aggregated
            Tp = np.arange(num_rows+1, dtype=index_type)
            Tx = np.ones(len(Tj), dtype='int8')
            return csr_matrix((Tx, Tj, Tp), shape=shape), Cpts


def naive_aggregation(C):
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
    Cpts : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import naive_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.todense()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> naive_aggregation(A)[0].todense() # two aggregates
    matrix([[1, 0],
            [1, 0],
            [0, 1],
            [0, 1]], dtype=int8)
    >>> A = csr_matrix([[1,0,0],[0,1,1],[0,1,1]])
    >>> A.todense()                      # first vertex is isolated
    matrix([[1, 0, 0],
            [0, 1, 1],
            [0, 1, 1]])
    >>> naive_aggregation(A)[0].todense() # two aggregates
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

    if not isspmatrix_csr(C):
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
        return csr_matrix((num_rows, 1), dtype='int8'), Cpts
    else:
        shape = (num_rows, num_aggregates)
        # all nodes aggregated
        Tp = np.arange(num_rows+1, dtype=index_type)
        Tx = np.ones(len(Tj), dtype='int8')
        return csr_matrix((Tx, Tj, Tp), shape=shape), Cpts


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
    seeds : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

    See Also
    --------
    amg_core.standard_aggregation

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
    >>> lloyd_aggregation(A)[0].todense() # one aggregate
    matrix([[1],
            [1],
            [1],
            [1]], dtype=int8)
    >>> # more seeding for two aggregates
    >>> Agg = lloyd_aggregation(A,ratio=0.5)[0].todense()
    """

    if ratio <= 0 or ratio > 1:
        raise ValueError('ratio must be > 0.0 and <= 1.0')

    if not (isspmatrix_csr(C) or isspmatrix_csc(C)):
        raise TypeError('expected csr_matrix or csc_matrix')

    if distance == 'unit':
        data = np.ones_like(C.data).astype(float)
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
        data = np.real(data)

    assert(data.min() >= 0)

    G = C.__class__((data, C.indices, C.indptr), shape=C.shape)

    num_seeds = int(min(max(ratio * G.shape[0], 1), G.shape[0]))

    distances, clusters, seeds = lloyd_cluster(G, num_seeds, maxiter=maxiter)

    row = (clusters >= 0).nonzero()[0]
    col = clusters[row]
    data = np.ones(len(row), dtype='int8')
    AggOp = coo_matrix((data, (row, col)),
                       shape=(G.shape[0], num_seeds)).tocsr()
    return AggOp, seeds


def weighted_matching(A, B=None, matchings=2,
                      get_weights=True,
                      improve_candidates=('gauss_seidel',
                                          {'sweep': 'forward',
                                           'iterations': 4}),
                      **kwargs):
    """ Pairwise aggregation of nodes using Drake approximate
        1/2-matching algorithm.

    Parameters
    ----------
    A : csr_matrix or bsr_matrix
        matrix for linear system.
    B : array_like : default None
        Right near-nullspace candidates stored in the columns of an NxK array.
        If no target vector provided, constant vector is used. In the case of
        multiple targets, k>1, only the first is used to construct coarse grid
        matrices for pairwise aggregations. 
    matchings : int : default 2
        Number of pairwise matchings to do. k matchings will lead to 
        a coarsening factor of under 2^k.
    get_weights : function handle : Default None
        Optional function handle to compute weights used in the matching,
        e.g. a strength of connection routine. Additional arguments for
        this routine should be provided in **kwargs. 
    improve_candidates : {tuple, string, list} : Default -
        ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 4, 'init': 'rand'})
        The list elements are relaxation descriptors of the form used for
        presmoother and postsmoother.  A value of None implies no action on B.
        Things to note:
            - If an initial target, B, is provided no smoothing is applied to it
              on the first pairwise pass, assuming it has already been smoothed.
            - If improve_candidates = None, an unsmoothed constant vector is 
              used as the target on each pairwise pass, as in [2].
    get_Cpts : {bool} : Default False
        Return list of C-points with aggregation matrix. Not currently
        implemented.

    NOTES
    -----
        - Not implemented for block systems or complex. 
            + Need to define how a matching is done nodally.
            + Also must consider what targets are used to form coarse grid
              in nodal approach...
            + Drake should be accessible in complex, but not Notay due to the
              hard minimum. Is there a < operator overloaded for complex?
              Could I overload it perhaps? Probably would do magnitude or something
              though, which is not what we want... 
        - Need to set up function to pick C-points too
        - Need to think about for nonsymmetric
            + Because new coarse grid is formed for each pairwise, not sure
              if we should call the function as is separately for P and R, i.e.
              form multiple pairwise Galerkin coarse grids for each as in the
              symmetric case, or simultaneously compute a pairwise for A and A^T
              and then form a petrov Galerkin coarse grid for the next pairwise...

    REFERENCES
    ----------
    [1] D'Ambra, Pasqua, and Panayot S. Vassilevski. "Adaptive AMG with
    coarsening based on compatible weighted matching." Computing and
    Visualization in Science 16.2 (2013): 59-76.

    [2] Drake, Doratha E., and Stefan Hougardy. "A simple approximation
    algorithm for the weighted matching problem." Information Processing
    Letters 85.4 (2003): 211-213.

    """

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        else:
            return v, {}

    if A.dtype == 'complex':
        raise TypeError("Not currently implemented for complex.")

    if not isinstance(matchings, int):
        raise TypeError("Number of matchings must be an integer.")

    if matchings < 1:
        raise ValueError("Number of matchings must be > 0.")

    if (A.getformat() != 'csr'):
        try:
            A = A.tocsr()
        except:
            raise TypeError("Must pass in CSR matrix, or sparse matrix "
                            "which can be converted to CSR.")

    n = A.shape[0]

    # If target vectors provided, take first.
    if B is not None:
        if len(B.shape) == 2:
            target = B[:,0]
        else:
            target = B[:,]
    else:
        target = None

    # Get arguments to improve targets. Targets are not improved
    # on the first level, as it is assumed the targets passed in
    # are sufficiently smooth.
    improve_fn, improve_args = unpack_arg(improve_candidates)

    # Compute weights if function provided, otherwise let W = A
    Ac = A      # Let Ac reference A for loop purposes
    if get_weights:
        weights = np.empty((A.nnz,),dtype=A.dtype)
        temp_cost = np.ones((1,), dtype=A.dtype)
        if target is None:
            amg_core.compute_weights(A.indptr, A.indices, A.data,
                                     weights, temp_cost)
        else:
            amg_core.compute_weights(A.indptr, A.indices, A.data,
                                     weights, target, temp_cost)

    else:
        weights = A.data

    # Loop over the number of pairwise matchings to be done
    for i in range(0,matchings):

        # Get matching and form sparse P
        rowptr = np.empty(n+1, dtype='intc')
        colinds = np.empty(n, dtype='intc')
        shape = np.empty(2, dtype='intc')
        temp_cost = np.ones((1,), dtype=A.dtype)
        if target is None:
            amg_core.drake_matching(Ac.indptr, Ac.indices, weights,
                                    rowptr, colinds, shape, temp_cost )
            T_temp = csr_matrix( (np.ones(n,), colinds, rowptr), shape=shape )
        else:
            data = np.empty(n, dtype=A.dtype)
            amg_core.drake_matching(Ac.indptr, Ac.indices, weights,
                                    target, rowptr, colinds, data,
                                    shape, temp_cost )
            T_temp = csr_matrix( (data, colinds, rowptr), shape=shape )


        # Form aggregation matrix 
        if i == 0:
            T = T_temp
        else:
            T = T * T_temp

        # Form coarse grid operator and restrict target to coarse grid 
        if i < (matchings-1):
            # Get complexity of Ac = T^TAT
            TA = T_temp.T * Ac
            Ac = TA * T_temp
            if target is not None:
                target = T_temp.T*target

            # If not last iteration, improve target by relaxing on A*target = 0.
            # If last iteration, we will not use target - set to None.
            n = Ac.shape[0]
            if (target is not None) and (improve_fn is not None):
                b = np.zeros((n, 1), dtype=Ac.dtype)
                target = relaxation_as_linear_operator((improve_fn, improve_args),
                                                       Ac, b, cost) * target         

            # Compute optional weights on coarse grid operator
            if get_weights:
                weights = np.empty((Ac.nnz,),dtype=Ac.dtype)
                temp_cost = np.ones((1,), dtype=Ac.dtype)
                if target is None:
                    amg_core.compute_weights(Ac.indptr, Ac.indices, Ac.data,
                                             weights, temp_cost)
                else:
                    amg_core.compute_weights(Ac.indptr, Ac.indices, Ac.data,
                                             weights, target, temp_cost)

            else:
                weights = Ac.data

    return T


def notay_pairwise(A, B=None, beta=0.25, matchings=2,
                   improve_candidates=None):
    """ Pairwise aggregation of nodes using Notay approach. 

    Parameters
    ----------
    A : csr_matrix or bsr_matrix
        matrix for linear system.
    B : array_like : default None
        Right near-nullspace candidates stored in the columns of an NxK array.
        If no target vector provided, constant vector is used. In the case of
        multiple targets, k>1, only the first is used to construct coarse grid
        matrices for pairwise aggregations. 
    beta : float

    matchings : int : default 2
        Number of pairwise matchings to do. k matchings will lead to 
        a coarsening factor of under 2^k.
    get_Cpts : {bool} : Default False
        Return list of C-points with aggregation matrix. Not currently
        implemented.

    NOTES
    -----
        - Not implemented for block systems or complex. 
            + Need to define how a matching is done nodally.
            + Also must consider what targets are used to form coarse grid
              in nodal approach...
            + Drake should be accessible in complex, but not Notay due to the
              hard minimum. Is there a < operator overloaded for complex?
              Could I overload it perhaps? Probably would do magnitude or something
              though, which is not what we want... 
        - Need to set up function to pick C-points too
        - Need to think about for nonsymmetric
            + Because new coarse grid is formed for each pairwise, not sure
              if we should call the function as is separately for P and R, i.e.
              form multiple pairwise Galerkin coarse grids for each as in the
              symmetric case, or simultaneously compute a pairwise for A and A^T
              and then form a petrov Galerkin coarse grid for the next pairwise...
            + As is, bases pairwise on row space.

    REFERENCES
    ----------
    [1] Notay, Yvan. "An aggregation-based algebraic multigrid method." 
    Electronic transactions on numerical analysis 37.6 (2010): 123-146.

    """

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        else:
            return v, {}

    if A.dtype == 'complex':
        raise TypeError("Not currently implemented for complex.")

    if not isinstance(matchings, int):
        raise TypeError("Number of matchings must be an integer.")

    if matchings < 1:
        raise ValueError("Number of matchings must be > 0.")

    if (A.getformat() != 'csr'):
        try:
            A = A.tocsr()
        except:
            raise TypeError("Must pass in CSR matrix, or sparse matrix "
                            "which can be converted to CSR.")

    n = A.shape[0]

    # If target vectors provided, take first.
    if B is not None:
        if len(B.shape) == 2:
            target = B[:,0]
        else:
            target = B[:,]
    else:
        target = None

    # Get arguments to improve targets. Targets are not improved
    # on the first level, as it is assumed the targets passed in
    # are sufficiently smooth.
    improve_fn, improve_args = unpack_arg(improve_candidates)

    # Compute weights if function provided, otherwise let W = A
    Ac = A      # Let Ac reference A for loop purposes
    T = None

    # Loop over the number of pairwise matchings to be done
    for i in range(0,matchings):

        # Get matching and form sparse P
        rowptr = np.empty(n+1, dtype='intc')
        colinds = np.empty(n, dtype='intc')
        shape = np.empty(2, dtype='intc')
        temp_cost = np.ones((1,), dtype='float64')
        if target is None:
            amg_core.notay_pairwise(Ac.indptr, Ac.indices, Ac.data,
                                    rowptr, colinds, shape, temp_cost,
                                    beta )
            T_temp = csr_matrix( (np.ones(n,), colinds, rowptr), shape=shape )
        else:
            data = np.empty(n, dtype=float)
            amg_core.notay_pairwise(Ac.indptr, Ac.indices, Ac.data,
                                    target, rowptr, colinds, data,
                                    shape, temp_cost, beta )
            T_temp = csr_matrix( (data, colinds, rowptr), shape=shape )


        # Form aggregation matrix 
        if i == 0:
            T = T_temp
        else:
            T = T * T_temp

        # Form coarse grid operator and restrict target to coarse grid 
        if i < (matchings-1):
            # Get complexity of Ac = T^TAT
            TA = T_temp.T * Ac
            Ac = TA * T_temp
            if target is not None:
                target = T_temp.T*target

            # If not last iteration, improve target by relaxing on A*target = 0.
            # If last iteration, we will not use target - set to None.
            n = Ac.shape[0]
            if (target is not None) and (improve_fn is not None):
                b = np.zeros((n, 1), dtype=Ac.dtype)
                target = relaxation_as_linear_operator((improve_fn, improve_args),
                                                       Ac, b, cost) * target         

    # NEED TO IMPLEMENT A WAY TO CHOOSE C-POINTS
    if get_Cpts:
        raise TypeError("Cannot return C-points - not yet implemented.")
    else:
        return T



""" TODO : 
    - Make sure same number of aggregates are formed.
        + Maybe should make function that takes "C-points," i.e. one
          node from each pair in first aggregate, and forms a new
          aggregate from its strongest neighbor?

    - Figure out what to do for C-points.

    """
def nonsymmetric_notay_pairwise(A, B=None, Bh=None, beta=0.25, matchings=2,
                                get_Cpts=False, **kwargs):

    # Get initial pairwise for A and A^T
    P = notay_pairwise(A=A, B=B, beta=beta, matchings=1,
                       get_Cpts=False, cost=cost)
    R = notay_pairwise(A=A.T, B=Bh, beta=beta, matchings=1,
                       get_Cpts=False, cost=cost)

    # References for loop purposes
    P_temp = P
    R_temp = R
    Ac = A

    # Loop over rest of matchings
    for i in range(1,matchings):

        # Form coarse grid and get complexity
        RA = R_temp * Ac
        Ac = RA * P_temp

        P_temp = notay_pairwise(A=Ac, B=B, beta=beta, matchings=1,
                                get_Cpts=False, cost=cost)
        R_temp = notay_pairwise(A=Ac.T, B=Bh, beta=beta, matchings=1,
                                get_Cpts=False, cost=cost)
        
        # Form updated R, P, get complexity
        P = P * P_temp
        R = R * R_temp

    return R, P



