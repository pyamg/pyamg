"""Solve an arbitrary system"""
from __future__ import print_function


import numpy as np
import scipy as sp
from scipy.sparse import isspmatrix_csr, isspmatrix_bsr, csr_matrix
from pyamg import smoothed_aggregation_solver
from pyamg.util.linalg import ishermitian, norm

__all__ = ['solve', 'solver', 'solver_configuration']


def make_csr(A):
    """
    Convert A to CSR, if A is not a CSR or BSR matrix already.

    Parameters
    ----------
    A : {array, matrix, sparse matrix}
        (n x n) matrix to convert to CSR

    Returns
    -------
    A : {csr_matrix, bsr_matrix}
        If A is csr_matrix or bsr_matrix, then do nothing and return A.
        Else, convert A to CSR if possible and return.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.blackbox import make_csr
    >>> A = poisson((40,40),format='csc')
    >>> Acsr = make_csr(A)
    Implicit conversion of A to CSR in pyamg.blackbox.make_csr
    """

    # Convert to CSR or BSR if necessary
    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        try:
            A = csr_matrix(A)
            print('Implicit conversion of A to CSR in pyamg.blackbox.make_csr')
        except:
            raise TypeError('Argument A must have type csr_matrix or\
                    bsr_matrix, or be convertible to csr_matrix')
    #
    if A.shape[0] != A.shape[1]:
        raise TypeError('Argument A must be a square')
    #
    A = A.asfptype()

    return A


def solver_configuration(A, B=None, verb=True):
    """
    Given an arbitrary matrix A, generate a dictionary of parameters with
    which to generate a smoothed_aggregation_solver.

    Parameters
    ----------
    A : array, matrix, csr_matrix, bsr_matrix
        (n x n) matrix to invert, CSR or BSR format preferred for efficiency
    B : None, array
        Near null-space modes used to construct the smoothed aggregation solver
        If None, the constant vector is used
        If (n x m) array, then B is passed to smoothed_aggregation_solver
    verb : bool
        If True, print verbose output during runtime

    Returns
    -------
    config : dict
        A dictionary of solver configuration parameters that one uses to
        generate a smoothed aggregation solver

    Notes
    -----
    The config dictionary contains the following parameter entries: symmetry,
    smooth, presmoother, postsmoother, B, strength, max_levels, max_coarse,
    coarse_solver, aggregate, keep.  See smoothed_aggregtion_solver for each
    parameter's description.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg import solver_configuration
    >>> A = poisson((40,40),format='csr')
    >>> solver_config = solver_configuration(A,verb=False)

    """

    # Ensure acceptable format of A
    A = make_csr(A)
    config = {}

    # Detect symmetry
    if ishermitian(A, fast_check=True):
        config['symmetry'] = 'hermitian'
        if verb:
            print("  Detected a Hermitian matrix")
    else:
        config['symmetry'] = 'nonsymmetric'
        if verb:
            print("  Detected a non-Hermitian matrix")

    # Symmetry dependent parameters
    if config['symmetry'] == 'hermitian':
        config['smooth'] = ('energy', {'krylov': 'cg', 'maxiter': 3,
                            'degree': 2, 'weighting': 'local'})
        config['presmoother'] = ('block_gauss_seidel',
                                 {'sweep': 'symmetric', 'iterations': 1})
        config['postsmoother'] = ('block_gauss_seidel',
                                  {'sweep': 'symmetric', 'iterations': 1})
    else:
        config['smooth'] = ('energy', {'krylov': 'gmres', 'maxiter': 3,
                            'degree': 2, 'weighting': 'local'})
        config['presmoother'] = ('gauss_seidel_nr',
                                 {'sweep': 'symmetric', 'iterations': 2})
        config['postsmoother'] = ('gauss_seidel_nr',
                                  {'sweep': 'symmetric', 'iterations': 2})

    # Determine near null-space modes B
    if B is None:
        # B is the constant for each variable in a node
        if isspmatrix_bsr(A) and A.blocksize[0] > 1:
            bsize = A.blocksize[0]
            config['B'] = np.kron(np.ones((int(A.shape[0] / bsize), 1),
                                  dtype=A.dtype), np.eye(bsize))
        else:
            config['B'] = np.ones((A.shape[0], 1), dtype=A.dtype)
    elif (isinstance(B, type(np.zeros((1,)))) or
            isinstance(B, type(sp.mat(np.zeros((1,)))))):
        if len(B.shape) == 1:
            B = B.reshape(-1, 1)
        if (B.shape[0] != A.shape[0]) or (B.shape[1] == 0):
            raise TypeError('Invalid dimensions of B, B.shape[0] must equal \
                             A.shape[0]')
        else:
            config['B'] = np.array(B, dtype=A.dtype)
    else:
        raise TypeError('Invalid B')

    if config['symmetry'] == 'hermitian':
        config['BH'] = None
    else:
        config['BH'] = config['B'].copy()

    # Set non-symmetry related parameters
    config['strength'] = ('evolution', {'k': 2, 'proj_type': 'l2',
                          'epsilon': 3.0})
    config['max_levels'] = 15
    config['max_coarse'] = 500
    config['coarse_solver'] = 'pinv'
    config['aggregate'] = 'standard'
    config['keep'] = False

    return config


def solver(A, config):
    """
    Given a matrix A and a solver configuration dictionary, generate a
    smoothed_aggregation_solver

    Parameters
    ----------
    A : {array, matrix, csr_matrix, bsr_matrix}
        Matrix to invert, CSR or BSR format preferred for efficiency
    config : {dict}
        A dictionary of solver configuration parameters that is used to
        generate a smoothed aggregation solver

    Returns
    -------
    ml : {smoothed_aggregation_solver}
        smoothed aggregation hierarchy

    Notes
    -----
    config must contain the following parameter entries for
    smoothed_aggregation_solver: symmetry, smooth, presmoother, postsmoother,
    B, strength, max_levels, max_coarse, coarse_solver, aggregate, keep

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg import solver_configuration,solver
    >>> A = poisson((40,40),format='csr')
    >>> config = solver_configuration(A,verb=False)
    >>> ml = solver(A,config)

    """

    # Convert A to acceptable format
    A = make_csr(A)

    # Generate smoothed aggregation solver
    try:
        return \
            smoothed_aggregation_solver(A,
                                        B=config['B'],
                                        BH=config['BH'],
                                        smooth=config['smooth'],
                                        strength=config['strength'],
                                        max_levels=config['max_levels'],
                                        max_coarse=config['max_coarse'],
                                        coarse_solver=config['coarse_solver'],
                                        symmetry=config['symmetry'],
                                        aggregate=config['aggregate'],
                                        presmoother=config['presmoother'],
                                        postsmoother=config['postsmoother'],
                                        keep=config['keep'])
    except:
        raise TypeError('Failed generating smoothed_aggregation_solver')


def solve(A, b, x0=None, tol=1e-5, maxiter=400, return_solver=False,
          existing_solver=None, verb=True, residuals=None):
    """
    Solve the arbitrary system Ax=b with the best out-of-the box choice for a
    solver.  The matrix A can be non-Hermitian, indefinite, Hermitian
    positive-definite, complex, etc...  Generic and robust settings for
    smoothed_aggregation_solver(..) are used to invert A.


    Parameters
    ----------
    A : {array, matrix, csr_matrix, bsr_matrix}
        Matrix to invert, CSR or BSR format preferred for efficiency
    b : {array}
        Right hand side.
    x0 : {array} : default random vector
        Initial guess
    tol : {float} : default 1e-5
        Stopping criteria: relative residual r[k]/r[0] tolerance
    maxiter : {int} : default 400
        Stopping criteria: maximum number of allowable iterations
    return_solver : {bool} : default False
        True: return the solver generated
    existing_solver : {smoothed_aggregation_solver} : default None
        If instance of a multilevel solver, then existing_solver is used
        to invert A, thus saving time on setup cost.
    verb : {bool}
        If True, print verbose output during runtime
    residuals : list
        List to contain residual norms at each iteration.
        The preconditioned norm is used, namely
        ||r||_M = (M r, r)^(1/2) = (r, r)^(1/2)

    Returns
    -------
    x : {array}
        Solution to Ax = b
    ml : multilevel_solver
        Optional return of the multilevel structure used for the solve

    Notes
    -----
    If calling solve(...) multiple times for the same matrix, A, solver reuse
    is easy and efficient.  Set "return_solver=True", and the return value will
    be a tuple, (x,ml), where ml is the solver used to invert A, and x is the
    solution to Ax=b.  Then, the next time solve(...) is called, set
    "existing_solver=ml".

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg import solve
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> A = poisson((40,40),format='csr')
    >>> b = np.array(np.arange(A.shape[0]), dtype=float)
    >>> x = solve(A,b,verb=False)
    >>> print "%1.2e"%(norm(b - A*x)/norm(b))
    6.28e-06
    """

    # Convert A to acceptable CSR/BSR format
    A = make_csr(A)

    # Generate solver if necessary
    if existing_solver is None:

        # Parameter dictionary for smoothed_aggregation_solver
        config = solver_configuration(A, B=None, verb=verb)
        # Generate solver
        existing_solver = solver(A, config)

    else:
        if existing_solver.levels[0].A.shape[0] != A.shape[0]:
            raise TypeError('Argument existing_solver must have level 0 matrix\
                             of same size as A')

    # Krylov acceleration depends on symmetry of A
    if existing_solver.levels[0].A.symmetry == 'hermitian':
        accel = 'cg'
    else:
        accel = 'gmres'

    # Initial guess
    if x0 is None:
        x0 = np.array(sp.rand(A.shape[0],), dtype=A.dtype)

    # Callback function to print iteration number
    if verb:
        iteration = np.zeros((1,))
        print("    maxiter = %d" % maxiter)

        def callback(x, iteration):
            iteration[0] = iteration[0] + 1
            print("    iteration %d" % iteration[0])

        def callback2(x):
            return callback(x, iteration)
    else:
        callback2 = None

    # Solve with accelerated Krylov method
    x = existing_solver.solve(b, x0=x0, accel=accel, tol=tol, maxiter=maxiter,
                              callback=callback2, residuals=residuals)

    if verb:
        r0 = b - A * x0
        rk = b - A * x
        M = existing_solver.aspreconditioner()
        nr0 = np.sqrt(np.inner(np.conjugate(M*r0), r0))
        nrk = np.sqrt(np.inner(np.conjugate(M*rk), rk))
        print("  Residuals ||r_k||_M, ||r_0||_M = %1.2e, %1.2e" % (nrk, nr0))
        if np.abs(nr0) > 1e-15:
            print("  Residual reduction ||r_k||_M/||r_0||_M = %1.2e"
                  % (nrk / nr0))

    if return_solver:
        return (x.reshape(b.shape), existing_solver)
    else:
        return x.reshape(b.shape)
