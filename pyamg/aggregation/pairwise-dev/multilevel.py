"""Generic AMG solver"""

__docformat__ = "restructuredtext en"

from warnings import warn
from pyamg.util.utils import unpack_arg

import scipy as sp
import numpy as np
from copy import deepcopy


__all__ = ['multilevel_solver', 'coarse_grid_solver', 'multilevel_solver_set']


class multilevel_solver:
    """Stores multigrid hierarchy and implements the multigrid cycle

    The class constructs the cycling process and points to the methods for
    coarse grid solves.  A multilevel_solver object is typically returned from
    a particular AMG method (see ruge_stuben_solver or
    smoothed_aggregation_solver for example).  A call to
    multilevel_solver.solve() is a typical access point.  The class also
    defines methods for constructing operator, cycle, and grid complexities.

    Attributes
    ----------
    levels : level array
        Array of level objects that contain A, R, and P.
    coarse_solver : string
        String passed to coarse_grid_solver indicating the solve type
    CC : {dict}
        Dictionary storing cycle complexity with key as cycle type.
    SC : float
        Setup complexity for constructing solver.

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
    """

    class level:
        """Stores one level of the multigrid hierarchy

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
        SC : float
            Setup complexity on this level in WUs relative to fine grid. 

        Notes
        -----
        The functionality of this class is a struct
        """
        def __init__(self):
            self.smoothers = {}
            self.smoothers['presmoother'] = [None, {}]
            self.smoothers['postsmoother'] = [None, {}]
            self.complexity = {}
            self.SC = None

    def __init__(self, levels, coarse_solver='pinv2'):
        """
        Class constructor responsible for initializing the cycle and ensuring
        the list of levels is complete.

        Parameters
        ----------
        levels : level array
            Array of level objects that contain A, R, and P.
        coarse_solver : {string, callable, tuple}
            The solver method is either (1) a string such as 'splu' or 'pinv'
            of a callable object which receives only parameters (A, b) and
            returns an (approximate or exact) solution to the linear system Ax
            = b, or (2) a callable object that takes parameters (A,b) and
            returns an (approximate or exact) solution to Ax = b, or (3) a
            tuple of the form (string|callable, args), where args is a
            dictionary of arguments to be passed to the function denoted by
            string or callable.

            The set of valid string arguments is:
            - Sparse direct methods:
                + splu         : sparse LU solver
            - Sparse iterative methods:
                + the name of any method in scipy.sparse.linalg.isolve or
                  pyamg.krylov (e.g. 'cg').  Methods in pyamg.krylov
                  take precedence.
                + relaxation method, such as 'gauss_seidel' or 'jacobi',
           - Dense methods:
                + pinv     : pseudoinverse (QR)
                + pinv2    : pseudoinverse (SVD)
                + lu       : LU factorization
                + cholesky : Cholesky factorization

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
        self.SC = None

        for level in levels[:-1]:
            if not hasattr(level, 'R'):
                level.R = level.P.H


    def __repr__(self):
        """Prints basic statistics about the multigrid hierarchy.
        """
        output = 'multilevel_solver\n'
        output += 'Number of Levels:     %d\n' % len(self.levels)
        output += 'Setup Complexity:     %6.3f\n' % self.setup_complexity()
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

    def setup_complexity(self, verbose=False):
        """Setup complexity of this multigrid hierarchy.

        Setup complexity is an approximate measure of the number of
        floating point operations (FLOPs) required to construct the
        multigrid hierarchy, relative to the cost of performing a
        single matrix-vector multiply on the finest grid.

        Parameters
        ----------
        verbose : bool
            If True, prints setup cost of each step, e.g. strength,
            aggregation, etc., in setup process on each level. 

        Returns
        -------
        sc : float
            Complexity of a constructing hierarchy in work units.
            A 'work unit' is defined as the number of FLOPs required
            to perform a matrix-vector multiply on the finest grid.

        Notes
        -----
            - Once computed, SC is stored in self.SC.

        """

        nnz = float(self.levels[0].A.nnz)

        if self.SC is None: 
            self.SC = 0.0
            for lvl in self.levels:
                if lvl.SC is None:
                    lvl.SC = 0.0
                    for cost in (lvl.complexity).itervalues():
                        lvl.SC += cost * (lvl.A.nnz / nnz)

                self.SC += lvl.SC

        if verbose:
            for i in range(0,len(self.levels)-1):
                lvl = self.levels[i]
                print "Level",i,"setup cost = ","%.3f"%lvl.SC, "WUs"
                for method, cost in (lvl.complexity).iteritems(): 
                    temp = cost*(lvl.A.nnz / nnz)
                    if method == "RAP":
                        print "\t",method,"\t\t= ","%.3f"%temp,"WUs"
                    else:
                        print "\t",method,"\t= ","%.3f"%temp,"WUs"

        return self.SC

    def cycle_complexity(self, cycle='V', init_level=0, recompute=False):
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
        def V(level):
            if len(self.levels) == 1:
                return rel_nnz_A[0]
            elif level == len(self.levels) - 2:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level]
            else:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level] + V(level + 1)

        def W(level):
            if len(self.levels) == 1:
                return rel_nnz_A[0]
            elif level == len(self.levels) - 2:
                return smoother_cost[level] +  correction_cost[level] + \
                    schwarz_work[level] 
            else:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level] + 2*W(level + 1)

        def F(level):
            if len(self.levels) == 1:
                return rel_nnz_A[0]
            elif level == len(self.levels) - 2:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level]
            else:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level] + F(level + 1) + V(level + 1)

        if cycle == 'V':
            flops = V(init_level)
        elif (cycle == 'W') or (cycle == 'AMLI'):
            flops = W(init_level)
        elif cycle == 'F':
            flops = F(init_level)
        else:
            raise TypeError('Unrecognized cycle type (%s)' % cycle)

        # Only save CC if computed for all levels to avoid confusion
        if init_level == 0:
            self.CC[cycle] = float(flops)

        return float(flops)


    def operator_complexity(self):
        """Operator complexity of this multigrid hierarchy

        Defined as:
            Number of nonzeros in the matrix on all levels /
            Number of nonzeros in the matrix on the finest level
        """
        return sum([level.A.nnz for level in self.levels]) /\
            float(self.levels[0].A.nnz)


    def grid_complexity(self):
        """Grid complexity of this multigrid hierarchy

        Defined as:
            Number of unknowns on all levels /
            Number of unknowns on the finest level
        """
        return sum([level.A.shape[0] for level in self.levels]) /\
            float(self.levels[0].A.shape[0])


    def psolve(self, b):
        return self.solve(b, maxiter=1)


    def aspreconditioner(self, cycle='V'):
        """Create a preconditioner using this multigrid cycle

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
        >>> from scipy import rand
        >>> A = poisson((100, 100), format='csr')          # matrix
        >>> b = rand(A.shape[0])                           # random RHS
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
              callback=None, residuals=None, return_residuals=False):
        """Main solution call to execute multigrid cycling.

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
        accel : {string, function}
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
            if (accel is 'cg') and (self.symmetric_smoothing == False):
                warn('Incompatible non-symmetric multigrid preconditioner detected, '
                     'due to presmoother/postsmoother combination. CG requires SPD '
                     'preconditioner, not just SPD matrix.')

            # Check for AMLI compatability
            if (accel != 'fgmres') and (cycle == 'AMLI'):
                raise ValueError('AMLI cycles require acceleration (accel) \
                        to be fgmres, or no acceleration')

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
            except:
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
                self.__solve(0, x, b, cycle)

            residuals.append(residual_norm(A, x, b))

            self.first_pass = False

            if callback is not None:
                callback(x)

        if return_residuals:
            return x, residuals
        else:
            return x


    def __solve(self, lvl, x, b, cycle):
        """
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
        """

        A = self.levels[lvl].A

        self.levels[lvl].presmoother(A, x, b)

        residual = b - A * x

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
                self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
                self.__solve(lvl + 1, coarse_x, coarse_b, 'V')
            elif cycle == "AMLI":
                # Run nAMLI AMLI cycles, which compute "optimal" corrections by
                # orthogonalizing the coarse-grid corrections in the A-norm
                nAMLI = 2
                Ac = self.levels[lvl+1].A
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
                        p[k, :] -= beta[k, j]*p[j, :]

                    # Compute step size
                    Ap = Ac*p[k, :]
                    alpha = np.inner(p[k, :].conj(), np.ravel(coarse_b)) /\
                        np.inner(p[k, :].conj(), Ap)

                    # Update solution
                    coarse_x += alpha*p[k, :].reshape(coarse_x.shape)

                    # Update residual
                    coarse_b -= alpha*Ap.reshape(coarse_b.shape)
            else:
                raise TypeError('Unrecognized cycle type (%s)' % cycle)

        x += self.levels[lvl].P * coarse_x   # coarse grid correction
        self.levels[lvl].postsmoother(A, x, b)
        
        # Only used in experimental additive cycle in multilevel_solver_set
        return self.levels[lvl].P * coarse_x

    def test_solve(self, lvl, x, b, cycle, additive=False):
        return self.__solve(lvl=lvl, x=x, b=b, cycle=cycle)


def coarse_grid_solver(solver):
    """Return a coarse grid solver suitable for multilevel_solver

    Parameters
    ----------
    solver : {string, callable, tuple}
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
    >>> from numpy import ones
    >>> from scipy.sparse import spdiags
    >>> from pyamg.gallery import poisson
    >>> from pyamg import coarse_grid_solver
    >>> A = poisson((10, 10), format='csr')
    >>> b = A * ones(A.shape[0])
    >>> cgs = coarse_grid_solver('lu')
    >>> x = cgs(A, b)
    """

    solver, kwargs = unpack_arg(solver, cost=False)

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


class multilevel_solver_set: 

    # Constructor to initialize an empty list of multilevel solver objects
    def __init__(self, hierarchy_set=None):
        self.num_hierarchies = 0
        self.hierarchy_set = []
        if hierarchy_set is not None:
            for h in hierarchy_set:
                if isinstance(h, multilevel_solver):
                    self.hierarchy_set.append(h)
                    self.num_hierarchies += 1
                else:
                    raise TypeError("Can only construct from list of multilevel_solver objects.")


    def add_hierarchy(self, hierarchy):
        if isinstance(hierarchy, multilevel_solver):
            self.hierarchy_set.append(hierarchy)
            self.num_hierarchies += 1
        else:
            raise TypeError("Can only add multilevel_solver objects to hierarchy.")


    def remove_hierarchy(self, ind):
        if ind < self.num_hierarchies:
            del self.hierarchy_set[ind]
            self.num_hierarchies -= 1
        else:
            raise ValueError("Hierarchy only contains %i sets, cannot remove set %i."%(self.num_hierarchies,ind))


    def replace_hierarchy(self, hierarchy, ind):
        if not isinstance(hierarchy, multilevel_solver):
            raise TypeError("Can only add multilevel_solver objects to hierarchy.")

        if ind < self.num_hierarchies:
            self.hierarchy_set[ind] = hierarchy
        else:
            raise ValueError("Hierarchy only contains %i sets, cannot remove set %i."%(self.num_hierarchies,ind))


    def cycle_complexity(self, cycle='V'):
        complexity = 0.0
        for h in self.hierarchy_set:
            complexity += h.cycle_complexity(cycle=cycle)

        return complexity
    

    def operator_complexity(self):
        operator_complexity = 0.0
        for h in self.hierarchy_set:
            operator_complexity += h.operator_complexity()

        return operator_complexity
    

    def grid_complexity(self):
        grid_complexity = 0.0
        for h in self.hierarchy_set:
            grid_complexity += h.grid_complexity(cycle=cycle)

        return grid_complexity


    def aspreconditioner(self, cycle='V'):
        from scipy.sparse.linalg import LinearOperator
        shape = self.hierarchy_set[0].levels[0].A.shape
        dtype = self.hierarchy_set[0].levels[0].A.dtype
        def matvec(b):
            return self.solve(b, maxiter=1, cycle=cycle, tol=1e-12, accel=None)

        return LinearOperator(shape, matvec, dtype=dtype)


    def solve(self, b, x0=None, tol=1e-5, maxiter=100, cycle='V', accel=None,
              callback=None, residuals=None, return_residuals=False, additive=False):

        if self.num_hierarchies == 0:
            raise ValueError("Cannot solve - zero hierarchies stored.")

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

        # Create uniform types for A, x and b
        # Clearly, this logic doesn't handle the case of real A and complex b
        from scipy.sparse.sputils import upcast
        from pyamg.util.utils import to_type

        A = self.hierarchy_set[0].levels[0].A
        tp = upcast(b.dtype, x.dtype, A.dtype)
        [b, x] = to_type(tp, [b, x])
        b = np.ravel(b)
        x = np.ravel(x)

        if accel is not None:

            # Check for AMLI compatability
            if (accel != 'fgmres') and (cycle == 'AMLI'):
                raise ValueError('AMLI cycles require acceleration (accel) \
                        to be fgmres, or no acceleration')

            # Acceleration is being used
            if isinstance(accel, basestring):
                from pyamg import krylov
                from scipy.sparse.linalg import isolve

                if hasattr(krylov, accel):
                    accel = getattr(krylov, accel)
                else:
                    accel = getattr(isolve, accel)

            M = self.aspreconditioner(cycle=cycle)

            n = x.shape[0] 
            try:  # try PyAMG style interface which has a residuals parameter
                return accel(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M,
                             callback=callback, residuals=residuals)[0].reshape((n,1))
            except:
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
                             callback=callback)[0].reshape((n,1))

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

        residuals.append(residual_norm(A, x, b))
        iter_num = 0

        while iter_num < maxiter and residuals[-1] > tol:
            # ----------- Additive solve ----------- #
            # ------ This doesn't really work ------ #
            if additive:
               x_copy = deepcopy(x)
               for hierarchy in self.hierarchy_set:
                    this_x = deepcopy(x_copy)
                    if len(hierarchy.levels) == 1:
                        this_x = hierarchy.coarse_solver(A, b)
                    else:
                        temp = hierarchy.test_solve(0, this_x, b, cycle)

                    x += temp
            # ----------- Normal solve ----------- #
            else:
                # One solve for each hierarchy in set
                for hierarchy in self.hierarchy_set:
                    # hierarchy has only 1 level
                    if len(hierarchy.levels) == 1:
                        x = hierarchy.coarse_solver(A, b)
                    else:
                        hierarchy.test_solve(0, x, b, cycle)

            residuals.append(residual_norm(A, x, b))
            iter_num += 1

            if callback is not None:
                callback(x)

        n = x.shape[0] 
        if return_residuals:
            return x.reshape((n,1)), residuals
        else:
            return x.reshape((n,1))



