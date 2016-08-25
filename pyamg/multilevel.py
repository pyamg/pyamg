"""Generic AMG solver"""

__docformat__ = "restructuredtext en"

from warnings import warn

import scipy as sp
import numpy as np

__all__ = ['multilevel_solver', 'coarse_grid_solver']


def unpack_arg(v):
    if isinstance(v, tuple):
        return v[0], v[1]
    else:
        return v, {}


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

        Notes
        -----
        The functionality of this class is a struct
        """
        def __init__(self):
            pass

    def __init__(self, levels, solver_type, params,
                 coarse_solver='pinv2', **kwargs):
        """
        Class constructor responsible for initializing the cycle and ensuring
        the list of levels is complete.

        Parameters
        ----------
        levels : level array
            Array of level objects that contain A, R, and P.
        solver_type : string
            Type of hierarchy this is, options are
            - 'sa'  => smoothed aggregation
            - 'asa' => adaptive smoothed aggregation
            - 'rn'  => root node
            - 'amg' => ruge-stuben amg
        params : {dictionary}
            Params is a dictionary containing all parameter choices used in
            setup. Is passed in from solver construction routines.
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
        self.nlevels = len(levels)
        self.coarse_solver = coarse_grid_solver(coarse_solver)
        try:
            self.symmetry = levels[0].A.symmetry
        except:
            warn("Symmetry not specified, assuming A is symmetric.")
            self.symmetry = True

        # Empty variables for complexity measures
        self.CC = {}
        self.SC = None

        # Store solver type and parameters used to construct hierarchy
        self.solver_type = solver_type
        self.solver_params = {}
        for key, value in params.iteritems():
            # Convert pre and post smoother to list of length num_levels - 1
            if (key is 'presmoother') or (key is 'postsmoother'):
                if isinstance(value, tuple):
                    temp = [value]
                elif isinstance(value, str):
                    temp = [(value, {})]
                else:
                    temp = value
                # Repeat final smoothing strategy until end of hiearchy
                for i in range(len(temp), self.nlevels-1):
                    temp.append(temp[-1])

                self.solver_params[key] = temp

            # Levelize strength and aggregate parameters
            elif (key is 'strength') or (key is 'aggregate'):
                pass

            # Levelize interp smoothing and improve candidates parameters
            elif (key is 'smooth') or (key is 'improve'):
                pass

            else:
                self.solver_params[key] = value

        if len(presmoother) != len(postsmoother):
            raise ValueError("pre and postsmoother must be same length")

        for level in levels[:-1]:
            if not hasattr(level, 'R'):
                level.R = level.P.H

    def __repr__(self):
        """Prints basic statistics about the multigrid hierarchy.
        """
        output = 'multilevel_solver\n'
        output += 'Number of Levels:     %d\n' % self.nlevels
        output += 'Operator Complexity: %6.3f\n' % self.operator_complexity()
        output += 'Grid Complexity:     %6.3f\n' % self.grid_complexity()
        output += 'Coarse Solver:        %s\n' % self.coarse_solver.name()

        total_nnz = sum([level.A.nnz for level in self.levels])

        output += '  level   unknowns     nonzeros\n'
        for n, level in enumerate(self.levels):
            A = level.A
            output += '   %2d   %10d   %10d [%5.2f%%]\n' %\
                (n, A.shape[1], A.nnz,
                 (100 * float(A.nnz) / float(total_nnz)))

        return output

    def setup_complexity(self):
        """Setup complexity of this multigrid hierarchy.

        Setup complexity is an approximate measure of the number of
        floating point operations (FLOPs) required to construct the
        multigrid hierarchy, relative to the cost of performing a
        single matrix-vector multiply on the finest grid.

        Parameters
        ----------
        None, passed into solver object when constructed.

        Returns
        -------
        sc : float
            Complexity of a constructing hierarchy in work units.
            A 'work unit' is defined as the number of FLOPs required
            to perform a matrix-vector multiply on the finest grid.

        Notes
        -----
            - Designed to be modular with individual functions called
              for the complexity of each part of setup. This will
              allow for easy inclusion of future methods into the
              interface.
            - Once computed, SC is stored in self.SC.
            - Needs to be done:
                + all distance based SOC measures
                + clean up Evolution SOC measure
                + all CF splittings
                + lloyd aggregation
                + Schwarz relaxation
                + energy smoothing
                    ~ same cost for SA vs. RN?
                    ~ why is satisy constraints commented out?
                    ~ should save in code number of iters done,
                      and any info on post-filtering.
                    ~ filtering takes 1 or 2 A.nnz / A0.nnz?

        """

        def _cost_strength(fn, fn_args, lvl, A0_nnz):

            # One pass through all entries to find largest element in each row,
            # one pass to filter by theta and
            # one to scale each row by largest entry
            if fn == 'symmetric':
                return 2.0 * lvl.A.nnz / A0_nnz
            elif fn == 'classical':
                return 3.0 * lvl.A.nnz / A0_nnz

            # TODO MAYBE SAVE SIZE WHE COMPUTING?
            #      COMPUTING MATRIX POWERS AGAIN IS SUBOPTIMAL, AND PERHAPS NOT
            #      ACCURATE BASED ON EPISLON? WHAT ABOUT K <= 2?
            elif (fn == 'ode') or (fn == 'evolution'):
                # work += lvl.A.nnz*(lvl.A.nnz/float(lvl.A.shape[0]))
                # Compute the work for kwargs['k'] > 2
                #  (nnz to compute) * (avg stencil size in multiplying matrix)
                Apow = lvl.A**(int(fn_args['k']/2))
                return lvl.A.nnz*(Apow.nnz / float(lvl.A.shape[0]))

            elif fn == 'energy_based':
                warn("Setup cost not implemented for energy-based SOC.\n\
                      Only implemented in python, so it is always slow.")
                return 0
            elif fn == 'distance':
                warn("Setup cost not implemented for distance-based SOC.")
                return 0
            elif fn == 'algebraic_distance':
                warn("Setup cost not implemented for algebraic distance SOC.")
                return 0
            elif fn == 'affinity_distance':
                warn("Setup cost not implemented for affinity distance SOC.")
                return 0
            else:
                warn("Unrecognized SOC measure. Setup cost not computed.")
                return 0

        def _cost_CF(fn, fn_args, lvl, A0_nnz):
            if fn == 'RS':
                warn("Setup cost not implemented for RS splitting.")
                return 0
            elif fn == 'PMIS':
                warn("Setup cost not implemented for PMIS splitting.")
                return 0
            elif fn == 'PMISc':
                warn("Setup cost not implemented for PMISc splitting.")
                return 0
            elif fn == 'CLJP':
                warn("Setup cost not implemented for CLJP splitting.")
                return 0
            elif fn == 'CLJPc':
                warn("Setup cost not implemented for CLJPc splitting.")
                return 0
            elif fn == 'CR':
                warn("Setup cost not implemented for CR.")
                return 0
            else:
                warn("Unrecognized CF splitting. Setup cost not computed.")
                return 0

        def _cost_aggregate(fn, fn_args, lvl, A0_nnz):
            # Roughly one pass through all nonzero elements to aggregate.
            # Technically it is one pass through strength matrix, C. If C is
            # not stored, A is reasonable estimate, A.nnz >= C.nnz, as more
            # counting one pass through nonzeros is a slight underestimate.
            if (fn == 'standard')or (fn == 'naive'):
                try:
                    return 1.0*lvl.C.nnz / A0_nnz
                except:
                    return 1.0*lvl.A.nnz / A0_nnz
            elif fn == 'lloyd':
                warn("Setup cost not implemented for lloyd aggregation.")
                return 0
            else:
                warn("Unrecognized aggregation method."
                     "Setup cost not computed.")
                return 0

        def _cost_improve(fn, fn_args, lvl, A0_nnz, symmetry):
            # Compute cost multiplier for relaxation method
            cost_factor = 1
            if fn.endswith(('nr', 'ne')):
                cost_factor *= 2
            if 'sweep' in fn_args:
                if fn_args['sweep'] == 'symmetric':
                    cost_factor *= 2
            if 'iterations' in fn_args:
                cost_factor *= fn_args['iterations']
            if 'degree' in fn_args:
                cost_factor *= fn_args['degree']
            if symmetry is False:
                cost_factor *= 2

            return cost_factor*lvl.A.nnz*lvl.B.shape[1] / A0_nnz

        def _cost_schwarz(fn, fn_args, lvl, A0_nnz):
            return None

        def _cost_interpsmooth(fn, fn_args, lvl, A0_nnz, symmetry):
            cost = 0
            if (fn == 'jacobi') or (fn == 'richardson'):
                cost += lvl.A.nnz*(lvl.P.nnz/float(lvl.P.shape[0]))
                if symmetry is False:
                    cost += lvl.A.nnz*(lvl.R.nnz/float(lvl.R.shape[1]))
                if 'degree' in fn_args:
                    cost *= fn_args['degree']

            # TODO FILL THIS IN
            # --------------------
            elif fn == 'energy':
                cost = 1

            return cost

        def _cost_RAP(lvl, A0_nnz):
            RA = lvl.A.nnz*(lvl.R.nnz/float(lvl.R.shape[1])) / A0_nnz
            AP = lvl.A.nnz*(lvl.P.nnz/float(lvl.P.shape[0])) / A0_nnz
            return (RA + AP)

        def _get_sa_cost():
            cost = 0
            A0_nnz = float(self.levels[0].A.nnz)
            symmetry = self.params
            # Loop through all but last level (no work done on final level)
            for i in range(0, self.nlevels-1):
                lvl = self.levels[i]

                # Strength of connection cost
                fn, fn_args = unpack_arg(strength[i])
                if fn is not None:
                    cost += _cost_strength(fn=fn, fn_args=fn_args, lvl=lvl,
                                           A0_nnz=A0_nnz)

                # Aggregation cost
                fn, fn_args = unpack_arg(aggregate[i])
                if fn is not None:
                    cost += _cost_aggregate(fn=fn, fn_args=fn_args, lvl=lvl,
                                            A0_nnz=A0_nnz)

                # Improve candidate vectors cost
                fn, fn_args = unpack_arg(improve_candidates[i])
                if fn is not None:
                    cost += _cost_improve(fn=fn, fn_args=fn_args, lvl=lvl,
                                          A0_nnz=A0_nnz, symmetry=symmetry)

                # Schwarz relaxation cost
                # ---> TODO : verify
                fn, fn_args = unpack_arg(smooth[i])
                if fn is not None:
                    cost += _cost_schwarz(fn=fn, fn_args=fn_args, lvl=lvl,
                                          A0_nnz=A0_nnz)

                # Smoothing interpolation operator cost
                fn, fn_args = unpack_arg(smooth[i])
                if fn is not None:
                    cost += _cost_interpsmooth(fn=fn, fn_args=fn_args, lvl=lvl,
                                               A0_nnz=A0_nnz,
                                               symmetry=self.symmetry)

                # Cost of computing coarse grid
                cost += _cost_RAP(lvl=lvl, A0_nnz=A0_nnz)

            return cost

        # Check if setup complexity has already been computed
        if self.SC is not None:

            if self.solver_type == 'sa':
                self.SC = _get_sa_cost()
            elif self.solver_type == 'rn':
                warn("Setup complexity not implemented for aSA."
                     "Returning zero.")
            elif self.solver_type == 'amg':
                warn("Setup complexity not implemented for aSA."
                     "Returning zero.")
            elif self.solver_type == 'asa':
                warn("Setup complexity not implemented for aSA."
                     "Returning zero.")
                self.SC = 0
            else:
                warn("Unrecognized solver type."
                     "Returning setup complexity zero.")
                self.SC = 0

    def cycle_complexity(self, cycle='V'):
        """Cycle complexity of this multigrid hierarchy.

        Cycle complexity is an approximate measure of the number of
        floating point operations (FLOPs) required to perform a single
        multigrid cycle relative to the cost a single smoothing operation.

        Parameters
        ----------
        cycle : {'V','W','F','AMLI'}
            Type of multigrid cycle to perform in each iteration.

        Returns
        -------
        cc : float
            Complexity of a single multigrid iteration in work units.
            A 'work unit' is defined as the number of FLOPs required
            to perform a matrix-vector multiply on the finest grid.

        Notes
        -----
        Once computed, CC is stored in dictionary self.CC, with a key
        given by the cycle type.

        """

        # Return if already stored
        if cycle in self.CC:
            return self.CC[cycle]

        # Get nonzeros per level and nonzeros per level relative to finest
        nnz = [level.A.nnz for level in self.levels]
        rel_nnz_A = [level.A.nnz/float(nnz[0]) for level in self.levels]
        rel_nnz_P = [level.P.nnz/float(nnz[0]) for level in self.levels[0:-1]]
        rel_nnz_R = [level.R.nnz/float(nnz[0]) for level in self.levels[0:-1]]

        # Cycle type and pre/post smoothers
        presmoother = self.solver_params['presmoother']
        postsmoother = self.solver_params['postsmoother']
        cycle = str(cycle).upper()

        # Determine cost per nnz for smoothing on each level
        smoother_cost = []
        for i in range(len(presmoother)):

            # Presmoother
            pre_factor = 1
            if presmoother[i][0].endswith(('nr', 'ne')):
                pre_factor *= 2
            if 'sweep' in presmoother[i][1]:
                if presmoother[i][1]['sweep'] == 'symmetric':
                    pre_factor *= 2
            if 'iterations' in presmoother[i][1]:
                pre_factor *= presmoother[i][1]['iterations']
            if 'degree' in presmoother[i][1]:
                pre_factor *= presmoother[i][1]['degree']

            # Postsmoother
            post_factor = 1
            if postsmoother[i][0].endswith(('nr', 'ne')):
                post_factor *= 2
            if 'sweep' in postsmoother[i][1]:
                if postsmoother[i][1]['sweep'] == 'symmetric':
                    post_factor *= 2
            if 'iterations' in postsmoother[i][1]:
                post_factor *= postsmoother[i][1]['iterations']
            if 'degree' in postsmoother[i][1]:
                post_factor *= postsmoother[i][1]['degree']

            # Smoothing cost scaled by A_i.nnz / A_0.nnz
            smoother_cost.append((pre_factor + post_factor)*rel_nnz_A[i])

        # Compute work for any Schwarz relaxation
        #   - The multiplier is the average row length, which is how many times
        #     the residual (on average) must be computed for each row.
        #   - schwarz_work is the cost of multiplying with the
        #     A[region_i, region_i]^{-1}
        schwarz_multiplier = np.zeros((len(presmoother),))
        schwarz_work = np.zeros((len(presmoother),))
        for i, lvl in enumerate(self.levels[:-1]):
            fn1, kwargs1 = unpack_arg(presmoother[i])
            fn2, kwargs2 = unpack_arg(postsmoother[i])
            if (fn1 == 'schwarz') or (fn2 == 'schwarz'):
                S = lvl.A
            if (fn1 == 'strength_based_schwarz') or \
               (fn2 == 'strength_based_schwarz'):
                S = lvl.C
            if (fn1.find('schwarz') > 0) or \
               (fn2.find('schwarz') > 0):
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

        # Recursive functions to sum cost of given cycle type over all levels,
        # including coarse grid direct solve (assume ~ 30n^3 for pinv).
        C_pinv = 1.0

        def V(level):
            if self.nlevels == 1:
                return rel_nnz_A[0]
            elif level == self.nlevels - 2:
                return smoother_cost[level] + schwarz_work[level] + \
                    C_pinv*(self.levels[level + 1].A.shape[0])**3 / nnz[0]
            else:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level] + V(level + 1)

        def W(level):
            if self.nlevels == 1:
                return rel_nnz_A[0]
            elif level == self.nlevels - 2:
                return smoother_cost[level] + schwarz_work[level] + \
                    C_pinv*(self.levels[level + 1].A.shape[0])**3 / nnz[0]
            else:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level] + 2*W(level + 1)

        def F(level):
            if self.nlevels == 1:
                return rel_nnz_A[0]
            elif level == self.nlevels - 2:
                return smoother_cost[level] + schwarz_work[level] + \
                    C_pinv*(self.levels[level + 1].A.shape[0])**3 / nnz[0]
            else:
                return smoother_cost[level] + correction_cost[level] + \
                    schwarz_work[level] + F(level + 1) + V(level + 1)

        if cycle == 'V':
            flops = V(0)
        elif (cycle == 'W') or (cycle == 'AMLI'):
            flops = W(0)
        elif cycle == 'F':
            flops = F(0)
        else:
            raise TypeError('Unrecognized cycle type (%s)' % cycle)

        self.CC[cycle] = float(flops)
        return self.CC[cycle]

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
            if self.nlevels == 1:
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

        if lvl == self.nlevels - 2:
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
        else:
            return v, {}

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
