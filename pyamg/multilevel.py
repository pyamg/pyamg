"""Generic AMG solver."""

from warnings import warn

import scipy as sp
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import LinearOperator
import numpy as np

from pkg_resources import parse_version  # included with setuptools

from . import krylov
from .util.utils import to_type
from .util.params import set_tol
from .relaxation import smoothing
from .util import upcast

if parse_version(sp.__version__) >= parse_version('1.7'):
    from scipy.linalg import pinv           # pylint: disable=ungrouped-imports
else:
    from scipy.linalg import pinv2 as pinv  # pylint: disable=no-name-in-module


class MultilevelSolver:
    """Stores multigrid hierarchy and implements the multigrid cycle.

    The class constructs the cycling process and points to the methods for
    coarse grid solves.  A MultilevelSolver object is typically returned from a
    particular AMG method (see ruge_stuben_solver or smoothed_aggregation_solver
    for example).  A call to MultilevelSolver.solve() is a typical access
    point.  The class also defines methods for constructing operator, cycle, and
    grid complexities.

    Attributes
    ----------
    levels : level array
        Array of level objects that contain A, R, and P.
    coarse_solver : string
        String passed to coarse_grid_solver indicating the solve type
    CC : {dict}
        Dictionary storing cycle complexity with key as cycle type.

    Methods
    -------
    aspreconditioner()
        Create a preconditioner using this multigrid cycle
    cycle_complexity()
        A measure of the cost of a single multigrid cycle.
    grid_complexity()
        A measure of the rate of coarsening.
    operator_complexity()
        A measure of the size of the multigrid hierarchy.
    solve()
        Iteratively solves a linear system for the right hand side.
    change_solve_matrix(A)
        Change matrix solve/preconditioning matrix.
        This also changes the corresponding relaxation routines on the fine
        grid.  This can be used, for example, to precondition a
        quadratic finite element discretization with linears.
    visualize_coarse_grids()
        Dump a visualization of the coarse grids in the given directory.
    """

    class Level:  # pylint: disable=too-few-public-methods
        """Stores one level of the multigrid hierarchy.

        All level objects will have an 'A' attribute referencing the matrix
        of that level.  All levels, except for the coarsest level, will
        also have 'P' and 'R' attributes referencing the prolongation and
        restriction operators that act between each level and the next
        coarser level.

        Attributes
        ----------
        A : csr_matrix
            Problem matrix for Ax=b
        R : csr_matrix
            Restriction matrix between levels (often R = P.T)
        P : csr_matrix
            Prolongation or Interpolation matrix.
        smoothers : {dict}
            Dictionary with keys 'presmoother' and 'postsmoother', giving
            the relaxation schemes used on this level. Used to compute
            cycle complexity.
        complexity : {dict}
            Dictionary to store complexity for each step in setup process,
            such as constructing P or computing strength-of-connection
        verts : n x 2 array
            degree of freedom locations

        Notes
        -----
        The functionality of this class is a struct

        """

        def __init__(self):
            """Level construct (empty)."""
            self.smoothers = {}
            self.smoothers['presmoother'] = [None, {}]
            self.smoothers['postsmoother'] = [None, {}]
            self.A = None

    class level(Level):  # noqa: N801 pylint: disable=too-few-public-methods
        """Deprecated level class."""

        def __init__(self):
            """Raise deprecation warning on use, not import."""
            super().__init__()
            warn('level() is deprectated.  use Level()',
                 category=DeprecationWarning, stacklevel=2)

    def __init__(self, levels, coarse_solver='pinv'):
        """Class constructor to initialize the cycle and ensure list of levels is complete.

        Parameters
        ----------
        levels : level array
            Array of level objects that contain A, R, and P.
        coarse_solver: string, callable, tuple
            The solver method is either (1) a string such as 'splu' or 'pinv'
            of a callable object which receives only parameters (A, b) and
            returns an (approximate or exact) solution to the linear system Ax
            = b, or (2) a callable object that takes parameters (A,b) and
            returns an (approximate or exact) solution to Ax = b, or (3) a
            tuple of the form (string|callable, args), where args is a
            dictionary of arguments to be passed to the function denoted by
            string or callable.

            Sparse direct methods:

            * splu         : sparse LU solver

            Sparse iterative methods:

            * any method in scipy.sparse.linalg or pyamg.krylov (e.g. 'cg').
            * Methods in pyamg.krylov take precedence.
            * relaxation method, such as 'gauss_seidel' or 'jacobi',

            Dense methods:

            * pinv     : pseudoinverse (SVD)
            * lu       : LU factorization
            * cholesky : Cholesky factorization

        Notes
        -----
        If not defined, the R attribute on each level is set to
        the transpose of P.

        Examples
        --------
        >>> # manual construction of a two-level AMG hierarchy
        >>> from pyamg.gallery import poisson
        >>> from pyamg.multilevel import MultilevelSolver
        >>> from pyamg.strength import classical_strength_of_connection
        >>> from pyamg.classical.interpolate import direct_interpolation
        >>> from pyamg.classical.split import RS
        >>> # compute necessary operators
        >>> A = poisson((100, 100), format='csr')
        >>> C = classical_strength_of_connection(A)
        >>> splitting = RS(A)
        >>> P = direct_interpolation(A, C, splitting)
        >>> R = P.T
        >>> # store first level data
        >>> levels = []
        >>> levels.append(MultilevelSolver.Level())
        >>> levels.append(MultilevelSolver.Level())
        >>> levels[0].A = A
        >>> levels[0].C = C
        >>> levels[0].splitting = splitting
        >>> levels[0].P = P
        >>> levels[0].R = R
        >>> # store second level data
        >>> levels[1].A = R @ A @ P                      # coarse-level matrix
        >>> # create MultilevelSolver
        >>> ml = MultilevelSolver(levels, coarse_solver='splu')
        >>> print(ml)
        MultilevelSolver
        Number of Levels:     2
        Operator Complexity:  1.891
        Grid Complexity:      1.500
        Coarse Solver:        'splu'
          level   unknowns     nonzeros
             0       10000        49600 [52.88%]
             1        5000        44202 [47.12%]
        <BLANKLINE>

        """
        self.symmetric_smoothing = False  # force change_smoothers to set to True
        self.levels = levels
        self.coarse_solver = coarse_grid_solver(coarse_solver)
        self.CC = {}

        for level in levels[:-1]:
            if not hasattr(level, 'R'):
                level.R = level.P.H

    def __repr__(self):
        """Print basic statistics about the multigrid hierarchy."""
        output = 'MultilevelSolver\n'
        output += f'Number of Levels:     {len(self.levels)}\n'
        output += f'Operator Complexity:  {self.operator_complexity():6.3f}\n'
        output += f'Grid Complexity:      {self.grid_complexity():6.3f}\n'
        output += f'Cycle Complexity:     {self.cycle_complexity():6.3f}\n'
        output += f'Coarse Solver:        {self.coarse_solver.name()}\n'

        total_nnz = sum(level.A.nnz for level in self.levels)

        #          123456712345678901 123456789012 123456789
        #               0       10000        49600 [52.88%]
        output += '  level   unknowns     nonzeros\n'
        for n, level in enumerate(self.levels):
            A = level.A
            ratio = 100 * A.nnz / total_nnz
            output += f'{n:>6} {A.shape[1]:>11} {A.nnz:>12} [{ratio:2.2f}%]\n'

        return output

    def cycle_complexity(self, cycle='V', cycles_per_level=1, init_level=0,
                         recompute=False):
        """Cycle complexity of this multigrid hierarchy.

        Cycle complexity is an approximate measure of the number of
        floating point operations (FLOPs) required to perform a single
        multigrid cycle relative to the cost a single smoothing operation.

        Parameters
        ----------
        cycle : {'V','W','F','AMLI'}
            Type of multigrid cycle to perform in each iteration.
        cycles_per_level : int, default 1
            Number of cycles per level
        init_level : int, default 0
            Compute CC for levels init_level,...,end. Used primarily
            for tracking SC in adaptive methods.
        recompute : bool, default False
            Recompute CC if already stored. Used if matrices or
            options in hierarchy have changed between computing CC.

        Returns
        -------
        cc : float
            Complexity of a single multigrid iteration in work units.
            A 'work unit' is defined as the number of FLOPs required
            to perform a matrix-vector multiply on the finest grid.

        Notes
        -----
        Once computed, CC is stored in dictionary self.CC, with a key
        given by the cycle type. Note, this is only for init_level=0.

        """
        if init_level > len(self.levels)-1:
            raise ValueError(f'Initial CC level must be in the range'
                             f'[0, {len(self.levels)}]')

        # Return if already stored
        cycle = str(cycle).upper()
        if cycle in self.CC and not recompute and init_level == 0:
            return self.CC[cycle]

        # Get nonzeros per level and nonzeros per level relative to finest
        nnz = np.array([float(level.A.nnz) for level in self.levels])
        rel_nnz_A = np.array([level.A.nnz/nnz[0] for level in self.levels])
        rel_nnz_P = np.array([level.P.nnz/nnz[0] for level in self.levels[:-1]])
        rel_nnz_R = np.array([level.R.nnz/nnz[0] for level in self.levels[:-1]])

        # Determine cost per nnz for smoothing on each level
        # Note: It is assumed that the default parameters in smoothing.py for each
        # relaxation scheme corresponds to a single workunit operation.
        smoother_cost = []
        for i in range(0, len(self.levels)-1):
            lvl = self.levels[i]
            presmoother = lvl.smoothers['presmoother']
            postsmoother = lvl.smoothers['postsmoother']

            # Presmoother
            if presmoother[0] is not None:
                pre_factor = 1
                if presmoother[0].endswith(('nr', 'ne')):
                    pre_factor *= 2
                if 'sweep' in presmoother[1]:
                    if presmoother[1]['sweep'] == 'symmetric':
                        pre_factor *= 2
                if 'iterations' in presmoother[1]:
                    pre_factor *= presmoother[1]['iterations']
                if 'maxiter' in presmoother[1]:
                    pre_factor *= presmoother[1]['maxiter']
                if 'degree' in presmoother[1]:
                    pre_factor *= presmoother[1]['degree']
                if presmoother[0].startswith(('CF', 'FC')):
                    temp = 0
                    if 'F_iterations' in presmoother[1]:
                        temp += presmoother[1]['F_iterations'] * lvl.nf / lvl.A.shape[0]
                    else:
                        temp += lvl.nf / lvl.A.shape[0]
                    if 'C_iterations' in presmoother[1]:
                        temp += presmoother[1]['C_iterations'] * lvl.nc / lvl.A.shape[0]
                    else:
                        temp += lvl.nc / lvl.A.shape[0]
                    pre_factor *= temp
            else:
                pre_factor = 0

            # Postsmoother
            if postsmoother[0] is not None:
                post_factor = 1
                if postsmoother[0].endswith(('nr', 'ne')):
                    post_factor *= 2
                if 'sweep' in postsmoother[1]:
                    if postsmoother[1]['sweep'] == 'symmetric':
                        post_factor *= 2
                if 'iterations' in postsmoother[1]:
                    post_factor *= postsmoother[1]['iterations']
                if 'maxiter' in postsmoother[1]:
                    post_factor *= postsmoother[1]['maxiter']
                if 'degree' in postsmoother[1]:
                    post_factor *= postsmoother[1]['degree']
                if postsmoother[0].startswith(('CF', 'FC')):
                    temp = 0
                    if 'F_iterations' in postsmoother[1]:
                        temp += postsmoother[1]['F_iterations'] * lvl.nf / lvl.A.shape[0]
                    else:
                        temp += lvl.nf / lvl.A.shape[0]
                    if 'C_iterations' in postsmoother[1]:
                        temp += postsmoother[1]['C_iterations'] * lvl.nc / lvl.A.shape[0]
                    else:
                        temp += lvl.nc / lvl.A.shape[0]
                    post_factor *= temp
            else:
                post_factor = 0

            # Smoothing cost scaled by A_i.nnz / A_0.nnz
            smoother_cost.append((pre_factor + post_factor)*rel_nnz_A[i])

        # Compute work for any Schwarz relaxation
        #   - The multiplier is the average row length, which is how many times
        #     the residual (on average) must be computed for each row.
        #   - schwarz_work is the cost of multiplying with the
        #     A[region_i, region_i]^{-1}
        schwarz_multiplier = np.zeros((len(self.levels)-1,))
        schwarz_work = np.zeros((len(self.levels)-1,))
        for i, lvl in enumerate(self.levels[:-1]):
            presmoother = lvl.smoothers['presmoother'][0]
            postsmoother = lvl.smoothers['postsmoother'][0]
            if (presmoother == 'schwarz') or (postsmoother == 'schwarz'):
                S = lvl.A
            if (presmoother == 'strength_based_schwarz') or \
               (postsmoother == 'strength_based_schwarz'):
                S = lvl.C
            if (presmoother is not None and presmoother.find('schwarz') > 0) or \
               (postsmoother is not None and postsmoother.find('schwarz') > 0):
                rowlen = S.indptr[1:] - S.indptr[:-1]
                schwarz_work[i] = np.sum(rowlen**2)
                schwarz_multiplier[i] = np.mean(rowlen)
                # Note this scaling only applies to multiplicative
                # Schwarz, which is what is currently available.
                smoother_cost[i] *= schwarz_multiplier[i]

        # Compute work for computing residual, restricting to coarse grid,
        # and coarse grid correction
        correction_cost = rel_nnz_A[:-1] + rel_nnz_R + rel_nnz_P

        # Recursive functions to sum cost of given cycle type over all levels.
        # Note, ignores coarse grid direct solve.
        def V(level, level_cycles):
            if len(self.levels) == 1:
                return rel_nnz_A[0]

            if level == len(self.levels) - 2:
                return smoother_cost[level] + correction_cost[level] + schwarz_work[level]

            return smoother_cost[level] + correction_cost[level] + \
                schwarz_work[level] + level_cycles * V(level+1, level_cycles=level_cycles)

        def W(level, level_cycles):
            if len(self.levels) == 1:
                return rel_nnz_A[0]

            if level == len(self.levels) - 2:
                return smoother_cost[level] + correction_cost[level] + schwarz_work[level]

            return smoother_cost[level] + correction_cost[level] + \
                schwarz_work[level] + 2 * level_cycles * W(level+1, level_cycles)

        def F(level, level_cycles):
            if len(self.levels) == 1:
                return rel_nnz_A[0]

            if level == len(self.levels) - 2:
                return smoother_cost[level] + correction_cost[level] + schwarz_work[level]

            return smoother_cost[level] + correction_cost[level] + \
                schwarz_work[level] + F(level+1, level_cycles) + \
                level_cycles * V(level+1, level_cycles=1)

        if cycle == 'V':
            flops = V(init_level, cycles_per_level)
        elif cycle in ('W', 'AMLI'):
            flops = W(init_level, cycles_per_level)
        elif cycle == 'F':
            flops = F(init_level, cycles_per_level)
        else:
            raise TypeError(f'Unrecognized cycle type ({cycle})')

        # Only save CC if computed for all levels to avoid confusion
        if init_level == 0:
            self.CC[cycle] = float(flops)

        return float(flops)

    def operator_complexity(self):
        """Operator complexity of this multigrid hierarchy.

        Defined as:
            Number of nonzeros in the matrix on all levels /
            Number of nonzeros in the matrix on the finest level

        """
        return sum(level.A.nnz for level in self.levels) /\
            float(self.levels[0].A.nnz)

    def grid_complexity(self):
        """Grid complexity of this multigrid hierarchy.

        Defined as:
            Number of unknowns on all levels /
            Number of unknowns on the finest level

        """
        return sum(level.A.shape[0] for level in self.levels) /\
            float(self.levels[0].A.shape[0])

    def change_solve_matrix(self, A):
        """Change matrix solve/preconditioning matrix.

        Parameters
        ----------
        A : csr_matrix
            Target solution matrix

        Notes
        -----
        This also changes the corresponding relaxation routines on the fine
        grid.  This can be used, for example, to precondition a
        quadratic finite element discretization with linears.
        """
        self.levels[0].A = A
        fn1, kwargs1 = self.levels[0].smoothers['presmoother']
        fn2, kwargs2 = self.levels[0].smoothers['postsmoother']

        # Rebuild presmoother
        try:
            setup_presmoother = smoothing._setup_call(str(fn1).lower())
        except NameError as exc:
            raise NameError(f'Invalid presmoother method: {fn1}') from exc
        self.levels[0].presmoother = setup_presmoother(self.levels[0], **kwargs1)

        # Rebuild postsmoother
        try:
            setup_postsmoother = smoothing._setup_call(str(fn2).lower())
        except NameError as exc:
            raise NameError(f'Invalid presmoother method: {fn2}') from exc
        self.levels[0].postsmoother = setup_postsmoother(self.levels[0], **kwargs2)

    def psolve(self, b):
        """Legacy solve interface."""
        return self.solve(b, maxiter=1)

    def aspreconditioner(self, cycle='V'):
        """Create a preconditioner using this multigrid cycle.

        Parameters
        ----------
        cycle : {'V','W','F','AMLI'}
            Type of multigrid cycle to perform in each iteration.

        Returns
        -------
        precond : LinearOperator
            Preconditioner suitable for the iterative solvers in defined in
            the scipy.sparse.linalg module (e.g. cg, gmres) and any other
            solver that uses the LinearOperator interface.  Refer to the
            LinearOperator documentation in scipy.sparse.linalg

        See Also
        --------
        MultilevelSolver.solve, scipy.sparse.linalg.LinearOperator

        Examples
        --------
        >>> from pyamg.aggregation import smoothed_aggregation_solver
        >>> from pyamg.gallery import poisson
        >>> from scipy.sparse.linalg import cg
        >>> import scipy as sp
        >>> A = poisson((100, 100), format='csr')          # matrix
        >>> b = np.random.rand(A.shape[0])                 # random RHS
        >>> ml = smoothed_aggregation_solver(A)            # AMG solver
        >>> M = ml.aspreconditioner(cycle='V')             # preconditioner
        >>> x, info = cg(A, b, tol=1e-8, maxiter=30, M=M)  # solve with CG

        """
        shape = self.levels[0].A.shape
        dtype = self.levels[0].A.dtype

        def matvec(b):
            return self.solve(b, maxiter=1, cycle=cycle, tol=1e-12)

        return LinearOperator(shape, matvec, dtype=dtype)

    # pylint: disable-next=too-many-return-statements
    def solve(self, b, x0=None, tol=1e-5, maxiter=100, cycle='V', accel=None,
              callback=None, residuals=None, cycles_per_level=1, return_info=False):
        """Execute multigrid cycling.

        Parameters
        ----------
        b : array
            Right hand side.
        x0 : array
            Initial guess.
        tol : float
            Stopping criteria: relative residual r[k]/||b|| tolerance.
            If `accel` is used, the stopping criteria is set by the Krylov method.
        maxiter : int
            Stopping criteria: maximum number of allowable iterations.
        cycle : {'V','W','F','AMLI'}
            Type of multigrid cycle to perform in each iteration.
        accel : string, function
            Defines acceleration method.  Can be a string such as 'cg'
            or 'gmres' which is the name of an iterative solver in
            pyamg.krylov (preferred) or scipy.sparse.linalg.
            If accel is not a string, it will be treated like a function
            with the same interface provided by the iterative solvers in SciPy.
        callback : function
            User-defined function called after each iteration.  It is
            called as callback(xk) where xk is the k-th iterate vector.
        residuals : list
            List to contain residual norms at each iteration.  The residuals
            will be the residuals from the Krylov iteration -- see the `accel`
            method to see verify whether this ||r|| or ||Mr|| (as in the case of
            GMRES).
        cycles_per_level: int, default 1
            Number of V-cycles on each level of an F-cycle
        return_info : bool
            If true, will return (x, info)
            If false, will return x (default)

        Returns
        -------
        x : array
            Approximate solution to Ax=b after k iterations

        info : string
            Halting status

            ==  =======================================
            0   successful exit
            >0  convergence to tolerance not achieved,
                return iteration count instead.
            ==  =======================================

        See Also
        --------
        aspreconditioner

        Examples
        --------
        >>> from numpy import ones
        >>> from pyamg import ruge_stuben_solver
        >>> from pyamg.gallery import poisson
        >>> A = poisson((100, 100), format='csr')
        >>> b = A * ones(A.shape[0])
        >>> ml = ruge_stuben_solver(A, max_coarse=10)
        >>> residuals = []
        >>> x = ml.solve(b, tol=1e-12, residuals=residuals) # standalone solver

        """
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = np.array(x0)  # copy

        A = self.levels[0].A

        cycle = str(cycle).upper()

        # AMLI cycles require hermitian matrix
        if (cycle == 'AMLI') and hasattr(A, 'symmetry'):
            if A.symmetry != 'hermitian':
                raise ValueError('AMLI cycles require \
                    symmetry to be hermitian')

        if accel is not None:

            # Check for symmetric smoothing scheme when using CG
            if (accel == 'cg') and (not self.symmetric_smoothing):
                warn('Incompatible non-symmetric multigrid preconditioner '
                     'detected, due to presmoother/postsmoother combination. '
                     'CG requires SPD preconditioner, not just SPD matrix.')

            # Check for AMLI compatability
            if (accel != 'fgmres') and (cycle == 'AMLI'):
                raise ValueError('AMLI cycles require acceleration (accel) '
                                 'to be fgmres, or no acceleration')

            # Acceleration is being used
            kwargs = {}
            if isinstance(accel, str):
                kwargs = {}
                if hasattr(krylov, accel):
                    accel = getattr(krylov, accel)
                else:
                    accel = getattr(sla, accel)
                    kwargs['atol'] = 'legacy'

            M = self.aspreconditioner(cycle=cycle)

            try:  # try PyAMG style interface which has a residuals parameter
                x, info = accel(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M,
                                callback=callback, residuals=residuals, **kwargs)
                if return_info:
                    return x, info
                return x
            except TypeError:
                # try the scipy.sparse.linalg style interface,
                # which requires a callback function if a residual
                # history is desired

                if residuals is not None:
                    residuals[:] = [np.linalg.norm(b - A @ x)]

                    def callback_wrapper(x):
                        if np.isscalar(x):
                            residuals.append(x)
                        else:
                            residuals.append(np.linalg.norm(b - A @ x))
                        if callback is not None:
                            callback(x)
                else:
                    callback_wrapper = callback

                x, info = accel(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M,
                                callback=callback_wrapper, **kwargs)
                if return_info:
                    return x, info
                return x

        else:
            # Scale tol by normb
            # Don't scale tol earlier. The accel routine should also scale tol
            normb = np.linalg.norm(b)
            if normb == 0.0:
                normb = 1.0  # set so that we have an absolute tolerance

        # Start cycling (no acceleration)
        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals[:] = [normr]  # initial residual

        # Create uniform types for A, x and b
        # Clearly, this logic doesn't handle the case of real A and complex b
        tp = upcast(b.dtype, x.dtype, A.dtype)
        [b, x] = to_type(tp, [b, x])
        b = np.ravel(b)
        x = np.ravel(x)

        it = 0

        while True:  # it <= maxiter and normr >= tol:
            if len(self.levels) == 1:
                # hierarchy has only 1 level
                x = self.coarse_solver(A, b)
            else:
                self.__solve(0, x, b, cycle, cycles_per_level)

            it += 1

            normr = np.linalg.norm(b - A @ x)
            if residuals is not None:
                residuals.append(normr)

            if callback is not None:
                callback(x)

            if normr < tol * normb:
                if return_info:
                    return x, 0
                return x

            if it == maxiter:
                if return_info:
                    return x, it
                return x

    def __solve(self, lvl, x, b, cycle, cycles_per_level=1):
        """Multigrid cycling.

        Parameters
        ----------
        lvl : int
            Solve problem on level `lvl`
        x : numpy array
            Initial guess `x` and return correction
        b : numpy array
            Right-hand side for Ax=b
        cycle : {'V','W','F','AMLI'}
            Recursively called cycling function.  The
            Defines the cycling used:
            cycle = 'V',    V-cycle
            cycle = 'W',    W-cycle
            cycle = 'F',    F-cycle
            cycle = 'AMLI', AMLI-cycle
        cycles_per_level : int, default 1
            Number of V-cycles on each level of an F-cycle
        """
        A = self.levels[lvl].A

        self.levels[lvl].presmoother(A, x, b)

        residual = b - A @ x

        coarse_b = self.levels[lvl].R * residual
        coarse_x = np.zeros_like(coarse_b)

        if lvl == len(self.levels) - 2:
            coarse_x[:] = self.coarse_solver(self.levels[-1].A, coarse_b)
        else:
            if cycle == 'V':
                self.__solve(lvl + 1, coarse_x, coarse_b, 'V')
            elif cycle == 'W':
                self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
                self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
            elif cycle == 'F':
                self.__solve(lvl + 1, coarse_x, coarse_b, cycle, cycles_per_level)
                for _ in range(0, cycles_per_level):
                    self.__solve(lvl + 1, coarse_x, coarse_b, 'V', 1)
            elif cycle == 'AMLI':
                # Run nAMLI AMLI cycles, which compute "optimal" corrections by
                # orthogonalizing the coarse-grid corrections in the A-norm
                nAMLI = 2
                Ac = self.levels[lvl + 1].A
                p = np.zeros((nAMLI, coarse_b.shape[0]), dtype=coarse_b.dtype)
                beta = np.zeros((nAMLI, nAMLI), dtype=coarse_b.dtype)
                for k in range(nAMLI):
                    # New search direction --> M^{-1}*residual
                    p[k, :] = 1
                    self.__solve(lvl + 1, p[k, :].reshape(coarse_b.shape),
                                 coarse_b, cycle)

                    # Orthogonalize new search direction to old directions
                    for j in range(k):  # loops from j = 0...(k-1)
                        beta[k, j] = np.inner(p[j, :].conj(), Ac * p[k, :]) /\
                            np.inner(p[j, :].conj(), Ac * p[j, :])
                        p[k, :] -= beta[k, j] * p[j, :]

                    # Compute step size
                    Ap = Ac * p[k, :]
                    alpha = np.inner(p[k, :].conj(), np.ravel(coarse_b)) /\
                        np.inner(p[k, :].conj(), Ap)

                    # Update solution
                    coarse_x += alpha * p[k, :].reshape(coarse_x.shape)

                    # Update residual
                    coarse_b -= alpha * Ap.reshape(coarse_b.shape)
            else:
                raise TypeError(f'Unrecognized cycle type ({cycle})')

        x += self.levels[lvl].P * coarse_x   # coarse grid correction

        self.levels[lvl].postsmoother(A, x, b)


def coarse_grid_solver(solver):
    """Return a coarse grid solver suitable for MultilevelSolver.

    Parameters
    ----------
    solver : string, callable, tuple
        The solver method is either (1) a string such as 'splu' or 'pinv' of a
        callable object which receives only parameters (A, b) and returns an
        (approximate or exact) solution to the linear system Ax = b, or (2) a
        callable object that takes parameters (A,b) and returns an (approximate
        or exact) solution to Ax = b, or (3) a tuple of the form
        (string|callable, args), where args is a dictionary of arguments to
        be passed to the function denoted by string or callable.

        The set of valid string arguments is:
            - Sparse direct methods:
                + splu : sparse LU solver
            - Sparse iterative methods:
                + the name of any method in scipy.sparse.linalg or
                  pyamg.krylov (e.g. 'cg').
                  Methods in pyamg.krylov take precedence.
                + relaxation method, such as 'gauss_seidel' or 'jacobi',
                  present in pyamg.relaxation
            - Dense methods:
                + pinv     : pseudoinverse (SVD)
                + lu       : LU factorization
                + cholesky : Cholesky factorization

    Returns
    -------
    ptr : GenericSolver
        A class for use as a standalone or coarse grids solver

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import spdiags
    >>> from pyamg.gallery import poisson
    >>> from pyamg import coarse_grid_solver
    >>> A = poisson((10, 10), format='csr')
    >>> b = A * np.ones(A.shape[0])
    >>> cgs = coarse_grid_solver('lu')
    >>> x = cgs(A, b)

    """

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}

    solver, kwargs = unpack_arg(solver)

    if solver in ['pinv', 'pinv2']:
        def solve(self, A, b):
            if not hasattr(self, 'P'):
                self.P = pinv(A.toarray(), **kwargs)
            return np.dot(self.P, b)

    elif solver == 'lu':
        def solve(self, A, b):
            if not hasattr(self, 'LU'):
                self.LU = sp.linalg.lu_factor(A.toarray(), **kwargs)
            return sp.linalg.lu_solve(self.LU, b)

    elif solver == 'cholesky':
        def solve(self, A, b):
            if not hasattr(self, 'L'):
                self.L = sp.linalg.cho_factor(A.toarray(), **kwargs)
            return sp.linalg.cho_solve(self.L, b)

    elif solver == 'splu':
        def solve(self, A, b):
            if not hasattr(self, 'LU'):
                # for multiple candidates in B, A will often have a couple zero
                # rows/columns that must be removed
                Acsc = A.tocsc()
                Acsc.eliminate_zeros()
                diffptr = Acsc.indptr[:-1] - Acsc.indptr[1:]
                nonzero_cols = (diffptr != 0).nonzero()[0]
                Map = sp.sparse.eye(Acsc.shape[0], Acsc.shape[1], format='csc')
                Map = Map[:, nonzero_cols]
                Acsc = Map.T.tocsc() * Acsc * Map
                self.LU = sp.sparse.linalg.splu(Acsc, **kwargs)
                self.LU_Map = Map

            return self.LU_Map * self.LU.solve(np.ravel(self.LU_Map.T * b))

    elif solver in ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'qmr', 'minres']:
        if hasattr(krylov, solver):
            fn = getattr(krylov, solver)
        else:
            fn = getattr(sla, solver)

        def solve(_, A, b):
            if 'tol' not in kwargs:
                kwargs['tol'] = set_tol(A.dtype)

            return fn(A, b, **kwargs)[0]

    elif solver in ['gauss_seidel', 'jacobi', 'block_gauss_seidel', 'schwarz',
                    'block_jacobi', 'richardson', 'sor', 'chebyshev',
                    'jacobi_ne', 'gauss_seidel_ne', 'gauss_seidel_nr']:

        if 'iterations' not in kwargs:
            kwargs['iterations'] = 10

        def solve(_, A, b):

            lvl = MultilevelSolver.Level()
            lvl.A = A
            fn = getattr(smoothing, 'setup_' + str(solver))
            relax = fn(lvl, **kwargs)
            x = np.zeros_like(b)
            relax(A, x, b)

            return x

    elif solver is None:
        # No coarse grid solve
        def solve(_, __, b):
            return 0 * b  # should this return b instead?

    elif callable(solver):
        def solve(_, A, b):
            return solver(A, b, **kwargs)

    else:
        raise ValueError(f'unknown solver: {solver}')

    class GenericSolver:
        """Generic solver class."""

        def __call__(self, A, b):
            # make sure x is same dimensions and type as b
            b = np.asanyarray(b)

            if A.nnz == 0:
                # if A.nnz = 0, then we expect no correction
                x = np.zeros(b.shape)
            else:
                x = solve(self, A, b)

            if isinstance(b, np.ndarray):
                x = np.asarray(x)
            elif isinstance(b, np.matrix):
                # convert to ndarray
                b = np.asarray(b)
                x = np.asarray(x)
            else:
                raise ValueError('unrecognized type')

            return x.reshape(b.shape)

        def __repr__(self):
            return 'coarse_grid_solver(' + repr(solver) + ')'

        @classmethod
        def name(cls):
            """Return the coarse solver name."""
            return repr(solver)

    return GenericSolver()


class multilevel_solver(MultilevelSolver):  # noqa: N801
    """Deprecated level class."""

    def __init__(self, *args, **kwargs):
        """Raise deprecation warning on use, not import."""
        super().__init__(*args, **kwargs)
        warn('multilevel_solver is deprectated.  use MultilevelSolver()',
             category=DeprecationWarning, stacklevel=2)
