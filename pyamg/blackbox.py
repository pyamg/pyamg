import numpy as np
from scipy.sparse import isspmatrix_csr, isspmatrix_bsr, csr_matrix
from .aggregation import smoothed_aggregation_solver
from .util.linalg import ishermitian


def make_csr(matrix):
    """Convert matrix to CSR format if necessary."""
    if not (isspmatrix_csr(matrix) or isspmatrix_bsr(matrix)):
        try:
            matrix = csr_matrix(matrix)
            print('Implicit conversion of matrix to CSR in pyamg.blackbox.make_csr')
        except Exception as e:
            raise TypeError('Matrix must be convertible to csr_matrix') from e

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Matrix must be square')

    return matrix.asfptype()


def configure_solver(matrix, near_nullspace=None, verbose=True):
    """Generate solver configuration for a matrix."""
    matrix = make_csr(matrix)
    config = {}

    # Detect symmetry
    is_symmetric = ishermitian(matrix, fast_check=True)
    config['symmetry'] = 'hermitian' if is_symmetric else 'nonsymmetric'
    if verbose:
        print(f"Detected a {'Hermitian' if is_symmetric else 'non-Hermitian'} matrix")

    # Symmetry-dependent parameters
    config['smooth'] = configure_smooth(config['symmetry'])
    config['presmoother'] = configure_smoother(config['symmetry'])
    config['postsmoother'] = configure_smoother(config['symmetry'])

    # Configure near-nullspace
    config['B'] = configure_near_nullspace(matrix, near_nullspace)
    config['BH'] = None if is_symmetric else config['B'].copy()

    # General parameters
    config.update({
        'strength': ('evolution', {'k': 2, 'proj_type': 'l2', 'epsilon': 3.0}),
        'max_levels': 15,
        'max_coarse': 500,
        'coarse_solver': 'pinv',
        'aggregate': 'standard',
        'keep': False
    })

    return config


def configure_smooth(symmetry):
    """Return smoothing parameters based on matrix symmetry."""
    krylov_method = 'cg' if symmetry == 'hermitian' else 'gmres'
    return ('energy', {'krylov': krylov_method, 'maxiter': 3, 'degree': 2, 'weighting': 'local'})


def configure_smoother(symmetry):
    """Return smoother parameters based on matrix symmetry."""
    if symmetry == 'hermitian':
        return ('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1})
    return ('gauss_seidel_nr', {'sweep': 'symmetric', 'iterations': 2})


def configure_near_nullspace(matrix, near_nullspace):
    """Determine near-nullspace modes for the solver."""
    if near_nullspace is None:
        if isspmatrix_bsr(matrix) and matrix.blocksize[0] > 1:
            block_size = matrix.blocksize[0]
            return np.kron(
                np.ones((matrix.shape[0] // block_size, 1), dtype=matrix.dtype),
                np.eye(block_size)
            )
        return np.ones((matrix.shape[0], 1), dtype=matrix.dtype)

    if isinstance(near_nullspace, np.ndarray):
        if near_nullspace.ndim == 1:
            near_nullspace = near_nullspace.reshape(-1, 1)
        if near_nullspace.shape[0] != matrix.shape[0]:
            raise ValueError('Dimension mismatch between matrix and near_nullspace.')
        return near_nullspace.astype(matrix.dtype)

    raise TypeError('Invalid near_nullspace input.')


def build_solver(matrix, config):
    """Generate a smoothed aggregation solver."""
    matrix = make_csr(matrix)

    try:
        return smoothed_aggregation_solver(
            matrix,
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
            keep=config['keep']
        )
    except Exception as e:
        raise RuntimeError('Failed to generate smoothed aggregation solver') from e


def solve(matrix, rhs, initial_guess=None, tol=1e-5, max_iter=400, 
          return_solver=False, existing_solver=None, verbose=True, residuals=None):
    """
    Solve Ax = b using a smoothed aggregation solver.
    """
    matrix = make_csr(matrix)

    # Generate or reuse solver
    if existing_solver is None:
        config = configure_solver(matrix, verbose=verbose)
        solver_instance = build_solver(matrix, config)
    else:
        if existing_solver.levels[0].A.shape[0] != matrix.shape[0]:
            raise ValueError('Existing solver level 0 matrix must match matrix size.')
        solver_instance = existing_solver

    # Determine acceleration method
    accel = 'cg' if solver_instance.levels[0].A.symmetry == 'hermitian' else 'gmres'

    # Initialize solution guess
    if initial_guess is None:
        initial_guess = np.random.rand(matrix.shape[0],).astype(matrix.dtype)

    # Callback for verbosity
    iteration_counter = [0]
    callback = create_callback(iteration_counter, verbose) if verbose else None

    # Solve the system
    solution = solver_instance.solve(
        rhs, x0=initial_guess, accel=accel, tol=tol, maxiter=max_iter,
        callback=callback, residuals=residuals
    )

    if verbose:
        print_residuals(matrix, rhs, initial_guess, solution, solver_instance)

    return (solution.reshape(rhs.shape), solver_instance) if return_solver else solution.reshape(rhs.shape)


def create_callback(iteration_counter, verbose):
    """Create a callback to track and print iterations."""
    def callback(_):
        iteration_counter[0] += 1
        if verbose:
            print(f"Iteration {iteration_counter[0]}")
    return callback


def print_residuals(matrix, rhs, initial_guess, solution, solver_instance):
    """Print residuals and convergence metrics."""
    residual_initial = rhs - matrix @ initial_guess
    residual_final = rhs - matrix @ solution
    preconditioner = solver_instance.aspreconditioner()

    norm_initial = np.sqrt(np.inner(np.conjugate(preconditioner @ residual_initial), residual_initial))
    norm_final = np.sqrt(np.inner(np.conjugate(preconditioner @ residual_final), residual_final))

    print(f"Residuals ||r_k||_M, ||r_0||_M = {norm_final:.2e}, {norm_initial:.2e}")
    if abs(norm_initial) > 1e-15:
        print(f"Residual reduction ||r_k||_M / ||r_0||_M = {norm_final / norm_initial:.2e}")
