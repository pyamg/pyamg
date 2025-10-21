"""Relaxation-centric utilities."""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from ..multilevel import MultilevelSolver
from .. import relaxation


def relaxation_as_linear_operator(method, A, b, splitting=None):
    """Create a linear operator that applies a relaxation method to a right-hand-side.

    Parameters
    ----------
    method : {tuple or string}
        Relaxation descriptor: Each tuple must be of the form ('method','opts')
        where 'method' is the name of a supported smoother, e.g., gauss_seidel,
        and 'opts' a dict of keyword arguments to the smoother, e.g., opts =
        {'sweep':symmetric}.  If string, must be that of a supported smoother,
        e.g., gauss_seidel.
    A : sparse matrix
        System matrix in A x = b
    b : array
        Right-hand side in A x = b

    A : {sparse matrix}
        Matrix for which we define relaxation, i.e., the linear operator object
        will carry out one iteration of relaxation with A on a vector

    b : {array}
        Right-hand-side for which we define relaxation (must be zero for the
        operator to be linear and not affine)

    splitting : {None or array}
        If relaxation type is fc_jacobi or cf_jacobi, then this must be a 
        boolean array defining the splitting

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
    >>> from pyamg.relaxation.utils import relaxation_as_linear_operator
    >>> import numpy as np
    >>> A = poisson((100,100), format='csr')           # matrix
    >>> B = np.ones((A.shape[0],1))                 # Candidate vector
    >>> b = np.zeros((A.shape[0]))                  # RHS
    >>> relax = relaxation_as_linear_operator('gauss_seidel', A, b)
    >>> B = relax*B

    """

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}

    # setup variables
    accepted_methods = ['gauss_seidel', 'block_gauss_seidel', 'sor',
                        'gauss_seidel_ne', 'gauss_seidel_nr', 'jacobi',
                        'block_jacobi', 'richardson', 'schwarz',
                        'strength_based_schwarz', 'jacobi_ne',
                        'fc_jacobi', 'cf_jacobi', 'fcf_jacobi',
                        'ff_l1_jacobi']

    b = np.array(b, dtype=A.dtype)
    fn, kwargs = unpack_arg(method)
    lvl = MultilevelSolver.Level()
    lvl.A = A
    lvl.splitting = splitting

    # Retrieve setup call from relaxation.smoothing for this relaxation method
    if fn not in accepted_methods:
        raise NameError(f'invalid relaxation method: {fn}')
    try:
        if (fn == 'fc_jacobi') or (fn == 'cf_jacobi') or (fn == 'fcf_jacobi') or (fn == 'ff_l1_jacobi'):
            if not isinstance(splitting, np.ndarray):
                raise NameError(f'splitting must be array of C and F points CF-style relaxation used')

        setup_smoother = getattr(relaxation.smoothing, 'setup_' + fn)
    except NameError as e:
        raise NameError(f'invalid presmoother method: {fn}') from e

    # Get relaxation routine that takes only (A, x, b) as parameters
    relax = setup_smoother(lvl, **kwargs)

    # Define matvec
    def matvec(x):
        xcopy = x.copy()
        relax(A, xcopy, b)
        return xcopy

    return LinearOperator(A.shape, matvec, dtype=A.dtype)
