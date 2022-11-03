"""Generic AMG solver."""


from warnings import warn
from pyamg.util.utils import unpack_arg
from pyamg.vis.vis_coarse import vis_splitting

import scipy as sp
import numpy as np


__all__ = ['multilevel_solver', 'coarse_grid_solver']


class multilevel_solver:
    """Stores multigrid hierarchy and implements the multigrid cycle.

    The class constructs the cycling process and points to the methods for
    coarse grid solves.  A multilevel_solver object is typically returned from a
    particular AMG method (see ruge_stuben_solver or smoothed_aggregation_solver
    for example).  A call to multilevel_solver.solve() is a typical access
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
    visualize_coarse_grids()
        Dump a visualization of the coarse grids in the given directory.
    """

    class level:
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
            self.smoothers = {}
            self.smoothers['presmoother'] = [None, {}]
            self.smoothers['postsmoother'] = [None, {}]
            self.auxiliary = None
            self.SC = None

    def __init__(self, levels, coarse_solver='pinv'):
        """Class constructor responsible for initializing the cycle and ensuring the list of levels is complete.

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

            * the name of any method in scipy.sparse.linalg.isolve or pyamg.krylov (e.g. 'cg').  Methods in pyamg.krylov take precedence.
            * relaxation method, such as 'gauss_seidel' or 'jacobi',

            Dense methods:

            * pinv     : pseudoinverse (QR)
            * pinv2    : pseudoinverse (SVD)
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
        >>> from pyamg.multilevel import multilevel_solver
        >>> from pyamg.strength import classical_strength_of_connection
        >>> from pyamg.classical import direct_interpolation
        >>> from pyamg.classical.split import RS
        >>> # compute necessary operators
        >>> A = poisson((100, 100), format='csr')
        >>> C = classical_strength_of_connection(A)
        >>> splitting = RS(A)
        >>> P = direct_interpolation(A, C, splitting)
        >>> R = P.T
        >>> # store first level data
        >>> levels = []
        >>> levels.append(multilevel_solver.level())
        >>> levels.append(multilevel_solver.level())
        >>> levels[0].A = A
        >>> levels[0].C = C
        >>> levels[0].splitting = splitting
        >>> levels[0].P = P
        >>> levels[0].R = R
        >>> # store second level data
        >>> levels[1].A = R * A * P                      # coarse-level matrix
        >>> # create multilevel_solver
        >>> ml = multilevel_solver(levels, coarse_solver='splu')
        >>> print ml
        multilevel_solver
        Number of Levels:     2
        Operator Complexity:  1.891
        Grid Complexity:      1.500
        Coarse Solver:        'splu'
          level   unknowns     nonzeros
            0        10000        49600 [52.88%]
            1         5000        44202 [47.12%]
        <BLANKLINE>

        """
        self.levels = levels
        self.coarse_solver = coarse_grid_solver(coarse_solver)
        self.CC = {}

        for level in levels[:-1]:
            if not hasattr(level, 'R'):
                level.R = level.P.H
            if not hasattr(level, 'auxiliary'):
                level.auxiliary = None


    def __repr__(self):
        """Print basic statistics about the multigrid hierarchy."""
        output = 'multilevel_solver\n'
        output += 'Number of Levels:     %d\n' % len(self.levels)
        output += 'Operator Complexity: %6.3f\n' % self.operator_complexity()
        output += 'Grid Complexity:     %6.3f\n' % self.grid_complexity()
        output += 'Cycle Complexity:    %6.3f\n' % self.cycle_complexity()
        output += 'Coarse Solver:        %s\n' % self.coarse_solver.name()

        total_nnz = sum([level.A.nnz for level in self.levels])

        output += '  level   unknowns     nonzeros\n'
        for n, level in enumerate(self.levels):
            A = level.A
            output += '   %2d   %10d   %10d [%5.2f%%]\n' %\
                (n, A.shape[1], A.nnz,
                 (100 * float(A.nnz) / float(total_nnz)))

        return output

    def cycle_complexity(self, cycle='V', cyclesPerLevel=1, init_level=0, recompute=False):
        """Cycle complexity of this multigrid hierarchy.

        Cycle complexity is an approximate measure of the number of
        floating point operations (FLOPs) required to perform a single
        multigrid cycle relative to the cost a single smoothing operation.

        Parameters
        ----------
        cycle : {'V','W','F','AMLI'}
            Type of multigrid cycle to perform in each iteration.
        init_level : int : Default 0
            Compute CC for levels init_level,...,end. Used primarily
            for tracking SC in adaptive methods. 
        recompute : bool : Default False
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
            raise ValueError("Initial CC level must be in the range [0, %i]"%len(self.levels))

        # Return if already stored
        cycle = str(cycle).upper()
        if cycle in self.CC and not recompute and init_level==0:
            return self.CC[cycle]

        # Get nonzeros per level and nonzeros per level relative to finest
        nnz = [float(level.A.nnz) for level in self.levels]
        rel_nnz_A = [level.A.nnz/nnz[0] for level in self.levels]
        rel_nnz_P = [level.P.nnz/nnz[0] for level in self.levels[0:-1]]
        rel_nnz_R = [level.R.nnz/nnz[0] for level in self.levels[0:-1]]

        # Determine cost per nnz for smoothing on each level
        # Note: It is assumed that the default parameters in smoothing.py for each
        # relaxation scheme corresponds to a single workunit operation.
        smoother_cost = []
        for i in range(0,len(self.levels)-1):
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
                if presmoother[0].startswith(('CF','FC')):
                    temp = 0
                    if 'F_iterations' in presmoother[1]:
                        temp += presmoother[1]['F_iterations'] * lvl.nf / float(lvl.A.shape[0])
                    else:
                        temp += lvl.nf / float(lvl.A.shape[0])
                    if 'C_iterations' in presmoother[1]:
                        temp += presmoother[1]['C_iterations'] * lvl.nc / float(lvl.A.shape[0])
                    else:
                        temp += lvl.nc / float(lvl.A.shape[0])
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
                if postsmoother[0].startswith(('CF','FC')):
                    temp = 0
                    if 'F_iterations' in postsmoother[1]:
                        temp += postsmoother[1]['F_iterations'] * lvl.nf / float(lvl.A.shape[0])
                    else:
                        temp += lvl.nf / float(lvl.A.shape[0])
                    if 'C_iterations' in postsmoother[1]:
                        temp += postsmoother[1]['C_iterations'] * lvl.nc / float(lvl.A.shape[0])
                    else:
                        temp += lvl.nc / float(lvl.A.shape[0])
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
        correction_cost = []
        for i in range(len(rel_nnz_P)):
            cost = 0
            cost += rel_nnz_A[i]    # Computing residual
            cost += rel_nnz_R[i]    # Restricting residual
            cost += rel_nnz_P[i]    # Coarse grid correction
            correction_cost.append(cost)

        # Recursive functions to sum cost of given cycle type over all levels.
        # Note, ignores coarse grid direct solve.
        def V(level, level_cycles):
            if len(self.levels) == 1:
                return rel_nnz_A[0]
            elif level == len(self.levels) - 2:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level]
            else:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level] + level_cycles * V(level+1, level_cycles=level_cycles)

        def W(level, level_cycles):
            if len(self.levels) == 1:
                return rel_nnz_A[0]
            elif level == len(self.levels) - 2:
                return smoother_cost[level] +  correction_cost[level] + \
                    schwarz_work[level] 
            else:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level] + 2*level_cycles*W(level+1, level_cycles)

        def F(level, level_cycles):
            if len(self.levels) == 1:
                return rel_nnz_A[0]
            elif level == len(self.levels) - 2:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level]
            else:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level] + F(level+1, level_cycles) + \
                    level_cycles * V(level+1, level_cycles=1)

        if cycle == 'V':
            flops = V(init_level, cyclesPerLevel)
        elif (cycle == 'W') or (cycle == 'AMLI'):
            flops = W(init_level, cyclesPerLevel)
        elif cycle == 'F':
            flops = F(init_level, cyclesPerLevel)
        else:
            raise TypeError('Unrecognized cycle type (%s)' % cycle)

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
        return sum([level.A.nnz for level in self.levels]) /\
            float(self.levels[0].A.nnz)

    def grid_complexity(self):
        """Grid complexity of this multigrid hierarchy.

        Defined as:
            Number of unknowns on all levels /
            Number of unknowns on the finest level

        """
        return sum([level.A.shape[0] for level in self.levels]) /\
            float(self.levels[0].A.shape[0])

    def psolve(self, b):
        """Lagacy solve interface."""
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
        multilevel_solver.solve, scipy.sparse.linalg.LinearOperator

        Examples
        --------
        >>> from pyamg.aggregation import smoothed_aggregation_solver
        >>> from pyamg.gallery import poisson
        >>> from scipy.sparse.linalg import cg
        >>> import scipy as sp
        >>> A = poisson((100, 100), format='csr')          # matrix
        >>> b = sp.rand(A.shape[0])                        # random RHS
        >>> ml = smoothed_aggregation_solver(A)            # AMG solver
        >>> M = ml.aspreconditioner(cycle='V')             # preconditioner
        >>> x, info = cg(A, b, tol=1e-8, maxiter=30, M=M)  # solve with CG

        """
        from scipy.sparse.linalg import LinearOperator

        shape = self.levels[0].A.shape
        dtype = self.levels[0].A.dtype

        def matvec(b):
            return self.solve(b, maxiter=1, cycle=cycle, tol=1e-12)

        return LinearOperator(shape, matvec, dtype=dtype)

    def solve(self, b, x0=None, tol=1e-5, maxiter=100, cycle='V', accel=None,
              callback=None, residuals=None, return_residuals=False, cyclesPerLevel=1):
        """Execute multigrid cycling.

        Parameters
        ----------
        b : array
            Right hand side.
        x0 : array
            Initial guess.
        tol : float
            Stopping criteria: relative residual r[k]/r[0] tolerance.
        maxiter : int
            Stopping criteria: maximum number of allowable iterations.
        cycle : {'V','W','F','AMLI'}
            Type of multigrid cycle to perform in each iteration.
        accel : string, function
            Defines acceleration method.  Can be a string such as 'cg'
            or 'gmres' which is the name of an iterative solver in
            pyamg.krylov (preferred) or scipy.sparse.linalg.isolve.
            If accel is not a string, it will be treated like a function
            with the same interface provided by the iterative solvers in SciPy.
        callback : function
            User-defined function called after each iteration.  It is
            called as callback(xk) where xk is the k-th iterate vector.
        residuals : list
            List to contain residual norms at each iteration.
        cyclesPerLevel: int
            number of V-cycles on each level of an F-cycle

        Returns
        -------
        x : array
            Approximate solution to Ax=b

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
        from pyamg.util.linalg import residual_norm, norm

        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = np.array(x0)  # copy

        cycle = str(cycle).upper()

        # AMLI cycles require hermitian matrix
        if (cycle == 'AMLI') and hasattr(self.levels[0].A, 'symmetry'):
            if self.levels[0].A.symmetry != 'hermitian':
                raise ValueError('AMLI cycles require \
                    symmetry to be hermitian')

        if accel is not None:

            # Check for symmetric smoothing scheme when using CG
            if (accel is 'cg') and (not self.symmetric_smoothing):
                warn('Incompatible non-symmetric multigrid preconditioner '
                     'detected, due to presmoother/postsmoother combination. '
                     'CG requires SPD preconditioner, not just SPD matrix.')

            # Check for AMLI compatability
            if (accel != 'fgmres') and (cycle == 'AMLI'):
                raise ValueError('AMLI cycles require acceleration (accel) '
                                 'to be fgmres, or no acceleration')

            # py23 compatibility:
            try:
                basestring
            except NameError:
                basestring = str

            # Acceleration is being used
            if isinstance(accel, basestring):
                from pyamg import krylov
                from scipy.sparse.linalg import isolve
                if hasattr(krylov, accel):
                    accel = getattr(krylov, accel)
                else:
                    accel = getattr(isolve, accel)

            A = self.levels[0].A
            M = self.aspreconditioner(cycle=cycle)

            try:  # try PyAMG style interface which has a residuals parameter
                return accel(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M,
                             callback=callback, residuals=residuals)[0]
            except BaseException:
                # try the scipy.sparse.linalg.isolve style interface,
                # which requires a call back function if a residual
                # history is desired

                cb = callback
                if residuals is not None:
                    residuals[:] = [residual_norm(A, x, b)]

                    def callback(x):
                        if sp.isscalar(x):
                            residuals.append(x)
                        else:
                            residuals.append(residual_norm(A, x, b))
                        if cb is not None:
                            cb(x)

                return accel(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M,
                             callback=callback)[0]

        else:
            # Scale tol by normb
            # Don't scale tol earlier. The accel routine should also scale tol
            normb = norm(b)
            if normb != 0:
                tol = tol * normb

        if return_residuals:
            warn('return_residuals is deprecated.  Use residuals instead')
            residuals = []
        if residuals is None:
            residuals = []
        else:
            residuals[:] = []

        # Create uniform types for A, x and b
        # Clearly, this logic doesn't handle the case of real A and complex b
        from scipy.sparse.sputils import upcast
        from pyamg.util.utils import to_type
        tp = upcast(b.dtype, x.dtype, self.levels[0].A.dtype)
        [b, x] = to_type(tp, [b, x])
        b = np.ravel(b)
        x = np.ravel(x)

        A = self.levels[0].A

        residuals.append(residual_norm(A, x, b))

        self.first_pass = True

        while len(residuals) <= maxiter and residuals[-1] > tol:
            if len(self.levels) == 1:
                # hierarchy has only 1 level
                x = self.coarse_solver(A, b)
            else:
                self.__solve(0, x, b, cycle, cyclesPerLevel)

            residuals.append(residual_norm(A, x, b))

            self.first_pass = False

            if callback is not None:
                callback(x)

        if return_residuals:
            return x, residuals
        else:
            return x

    def __auxiliary_solve(self, lvl, bc):
        """ Auxiliary solve using CG for Gen-AIR
        """
        xc = sp.sparse.linalg.cg(self.levels[lvl].auxiliary['M_aux'], bc, tol=1e-05, maxiter=10)[0]
        return xc

    def __solve(self, lvl, x, b, cycle, cyclesPerLevel=1):
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
        cyclesPerLevel: number of V-cycles on each level of an F-cycle
        """
        A = self.levels[lvl].A

        self.levels[lvl].presmoother(A, x, b)

        residual = b - A * x

        # Auxiliary coarse-grid solve for gen-AIR
        if self.levels[lvl].auxiliary is not None:
            coarse_b0 = self.levels[lvl].R * residual
            coarse_b = self.__auxiliary_solve(lvl, coarse_b0)
        else:
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
                self.__solve(lvl + 1, coarse_x, coarse_b, cycle, cyclesPerLevel)
                for ci in range(0,cyclesPerLevel):
                    self.__solve(lvl + 1, coarse_x, coarse_b, 'V', 1)
            elif cycle == "AMLI":
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
                raise TypeError('Unrecognized cycle type (%s)' % cycle)

        x += self.levels[lvl].P * coarse_x   # coarse grid correction

        # Auxiliary coarse-grid correction for gen-AIR
        if self.levels[lvl].auxiliary is not None:
            x += self.levels[lvl].auxiliary['P_aux'] * coarse_b

        self.levels[lvl].postsmoother(A, x, b)

    def visualize_coarse_grids(self, directory):
        # Dump a visualization of the coarse grids in the given directory.
        # If called for, output a visualization of the C/F splitting
        if (self.levels[0].verts.any()):
            for i in range(len(self.levels) - 1):
                filename = directory + '/cf_' + str(i) + '.vtu'
                vis_splitting(self.levels[i].verts, self.levels[i].splitting, fname=filename)
        else:
            print('Cannot visulize coarse grids: missing dof locations or \
                   splittings in multilevel instance. Pass in parameters \
                   verts = [nx2 array of dof locations] and keep = True when \
                   creating multilevel object.')


def coarse_grid_solver(solver):
    """Return a coarse grid solver suitable for multilevel_solver.

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
                + the name of any method in scipy.sparse.linalg.isolve or
                  pyamg.krylov (e.g. 'cg').
                  Methods in pyamg.krylov take precedence.
                + relaxation method, such as 'gauss_seidel' or 'jacobi',
                  present in pyamg.relaxation
            - Dense methods:
                + pinv     : pseudoinverse (QR)
                + pinv2    : pseudoinverse (SVD)
                + lu       : LU factorization
                + cholesky : Cholesky factorization

    Returns
    -------
    ptr : generic_solver
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
    solver, kwargs = unpack_arg(solver)

    if solver in ['pinv', 'pinv2']:
        def solve(self, A, b):
            if not hasattr(self, 'P'):
                self.P = getattr(sp.linalg, solver)(A.todense(), **kwargs)
            return np.dot(self.P, b)

    elif solver == 'lu':
        def solve(self, A, b):
            if not hasattr(self, 'LU'):
                self.LU = sp.linalg.lu_factor(A.todense(), **kwargs)
            return sp.linalg.lu_solve(self.LU, b)

    elif solver == 'cholesky':
        def solve(self, A, b):
            if not hasattr(self, 'L'):
                self.L = sp.linalg.cho_factor(A.todense(), **kwargs)
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
        from pyamg import krylov
        if hasattr(krylov, solver):
            fn = getattr(krylov, solver)
        else:
            fn = getattr(sp.sparse.linalg.isolve, solver)

        def solve(self, A, b):
            if 'tol' not in kwargs:
                eps = np.finfo(np.float).eps
                feps = np.finfo(np.single).eps
                geps = np.finfo(np.longfloat).eps
                _array_precision = {'f': 0, 'd': 1, 'g': 2,
                                    'F': 0, 'D': 1, 'G': 2}
                kwargs['tol'] = {0: feps * 1e3, 1: eps * 1e6,
                                 2: geps * 1e6}[_array_precision[A.dtype.char]]

            return fn(A, b, **kwargs)[0]

    elif solver in ['gauss_seidel', 'jacobi', 'block_gauss_seidel', 'schwarz',
                    'block_jacobi', 'richardson', 'sor', 'chebyshev',
                    'jacobi_ne', 'gauss_seidel_ne', 'gauss_seidel_nr']:

        if 'iterations' not in kwargs:
            kwargs['iterations'] = 10

        def solve(self, A, b):
            from pyamg.relaxation import smoothing
            from pyamg import multilevel_solver

            lvl = multilevel_solver.level()
            lvl.A = A
            fn = getattr(smoothing, 'setup_' + str(solver))
            relax = fn(lvl, **kwargs)
            x = np.zeros_like(b)
            relax(A, x, b)

            return x

    elif solver is None:
        # No coarse grid solve
        def solve(self, A, b):
            return 0 * b  # should this return b instead?

    elif callable(solver):
        def solve(self, A, b):
            return solver(A, b, **kwargs)

    else:
        raise ValueError('unknown solver: %s' % solver)

    class generic_solver:
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
                x = np.asmatrix(x)
            else:
                raise ValueError('unrecognized type')

            return x.reshape(b.shape)

        def __repr__(self):
            return 'coarse_grid_solver(' + repr(solver) + ')'

        def name(self):
            return repr(solver)

    return generic_solver()
