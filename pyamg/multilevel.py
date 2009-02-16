"""Generic AMG solver"""

__docformat__ = "restructuredtext en"

from warnings import warn
import scipy
import numpy
from numpy import ones, zeros, zeros_like, array, asarray, empty, asanyarray, ravel
from scipy.sparse import csc_matrix
from scipy.sparse.sputils import upcast

#from pyamg import relaxation
from pyamg.relaxation import *
from pyamg.util.utils import symmetric_rescaling, diag_sparse, to_type
from pyamg.util.linalg import residual_norm, norm

__all__ = ['multilevel_solver', 'coarse_grid_solver']


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
    solve()
        Iteratively solves a linear system for the right hand side.
    operator_complexity()
        A measure of the size of the multigrid hierarchy.
    grid_complexity()
        A measure of the rate of coarsening.
    cycle_complexity()
        A measure of the cost of a single multigrid cycle.
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

    def __init__(self, levels, coarse_solver='pinv2'):
        """
        Class constructor responsible for initializing the cycle and ensuring
        the list of levels is complete.

        Parameters
        ----------
        levels : level array
            Array of level objects that contain A, R, and P.
        coarse_solver : string
            String passed to coarse_grid_solver indicating the type of coarse
            grid solve to perform.  The default 'pinv2' is robust, but 
            expands the coarse level matrix into a dense format.  Therefore,
            larger coars level systems should use sparse coarse methods 
            like 'splu'.
    
        Notes
        -----
        If not defined, the R attribute on each level is set to the
        transpose of P.

        """
        """
        Class constructor responsible for initializing the cycle and ensuring
        the list of levels is complete.

        Parameters
        ----------
        levels : level array
            Array of level objects that contain A, R, and P.
        coarse_solver : string
            String passed to coarse_grid_solver indicating the type of coarse
            grid solve to perform.  The default 'pinv2' is robust, but 
            expands the coarse level matrix into a dense format.  Therefore,
            larger coars level systems should use sparse coarse methods 
            like 'splu'.
    
        Notes
        -----
        If not defined, the R attribute on each level is set to the
        transpose of P.

        """
        self.levels = levels
        
        self.coarse_solver = coarse_grid_solver(coarse_solver)

        for level in levels[:-1]:
            if not hasattr(level, 'R'):
                level.R = level.P.H

    def __repr__(self):
        """Prints basic statistics about the multigrid hierarchy.
        """
        output = 'multilevel_solver\n'
        output += 'Number of Levels:     %d\n'   % len(self.levels)
        output += 'Operator Complexity: %6.3f\n' % self.operator_complexity()
        output += 'Grid Complexity:     %6.3f\n' % self.grid_complexity()

        total_nnz =  sum([level.A.nnz for level in self.levels])

        output += '  level   unknowns     nonzeros\n'
        for n,level in enumerate(self.levels):
            A = level.A
            output += '   %2d   %10d   %10d [%5.2f%%]\n' % (n,A.shape[1],A.nnz,(100*float(A.nnz)/float(total_nnz)))

        return output

    def cycle_complexity(self, cycle='V'):
        """Cycle complexity of this Multigrid hierarchy
       
        Cycle complexity is an approximate measure of the number of
        floating point operations (FLOPs) required to perform a single 
        multigrid cycle relative to the cost a single smoothing operation.

        Parameters
        ----------
        cycle : {'V','W','F'}
            Type of multigrid cycle to perform in each iteration.

        Returns
        -------
        cc : float
            Defined as F_sum / F_0, where
            F_sum is the total number of nonzeros in the matrix on all 
            levels encountered during a cycle and F_0 is the number of 
            nonzeros in the matrix on the finest level.

        Notes
        -----
        This is only a rough estimate of the true cycle complexity. The 
        estimate assumes that the cost of pre and post-smoothing are 
        (each) equal to the number of nonzeros in the matrix on that level.
        This assumption holds for smoothers like Jacobi and Gauss-Seidel.
        However, the true cycle complexity of cycle using more expensive 
        methods, like symmetric Gauss-Seidel and Chebyshev smoothers will 
        be underestimated.
        
        """
        cycle = str(cycle).upper()

        nnz = [ level.A.nnz for level in self.levels ]

        def V(level):
            if len(self.levels) == 1:
                return nnz[0]
            elif level == len(self.levels) - 2:
                return 2*nnz[level] + nnz[level + 1]
            else:
                return 2*nnz[level] + V(level + 1)
        
        def W(level):
            if len(self.levels) == 1:
                return nnz[0]
            elif level == len(self.levels) - 2:
                return 2*nnz[level] + nnz[level + 1]
            else:
                return 2*nnz[level] + 2*W(level + 1)
        
        def F(level):
            if len(self.levels) == 1:
                return nnz[0]
            elif level == len(self.levels) - 2:
                return 2*nnz[level] + nnz[level + 1]
            else:
                return 2*nnz[level] + F(level + 1) + V(level + 1)

        if cycle == 'V':
            flops = V(0)
        elif cycle == 'W':
            flops = W(0)
        elif cycle == 'F':
            flops = F(0)
        else:
            raise TypeError('Unrecognized cycle type (%s)' % cycle)
        
        return float(flops) / float(nnz[0])


    def operator_complexity(self):
        """Operator complexity of this multigrid heirarchy 

        Defined as:
            Number of nonzeros in the matrix on all levels / 
            Number of nonzeros in the matrix on the finest level
        
        """
        return sum([level.A.nnz for level in self.levels])/float(self.levels[0].A.nnz)

    def grid_complexity(self):
        """Grid complexity of this multigrid heirarchy 
        
        Defined as:
            Number of unknowns on all levels / 
            Number of unknowns on the finest level

        """
        return sum([level.A.shape[0] for level in self.levels])/float(self.levels[0].A.shape[0])

    def psolve(self, b):
        return self.solve(b, maxiter=1)

    def aspreconditioner(self, cycle='V'):
        """Create a preconditioner using this multigrid cycle

        Parameters
        ----------
        cycle : {'V','W','F'}
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
        >>> from pyamg import smoothed_aggregation_solver, poisson
        >>> from scipy.sparse.linalg import cg
        >>> from scipy import rand
        >>> A = poisson((100,100), format='csr')           # matrix
        >>> b = rand(A.shape[0])                           # random RHS
        >>> ml = smoothed_aggregation_solver(A)            # AMG solver
        >>> M = ml.aspreconditioner(cycle='V')             # preconditioner
        >>> x,info = cg(A, b, tol=1e-8, maxiter=30, M=M)   # solve with CG
                    
        """
        from scipy.sparse.linalg import LinearOperator
        
        shape = self.levels[0].A.shape
        dtype = self.levels[0].A.dtype

        def matvec(b):
            return self.solve(b, maxiter=1, cycle=cycle)
                    
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
        cycle : {'V','W','F'}
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
        >>> from pyamg import ruge_stuben_solver, poisson
        >>> A = poisson((100,100), format='csr')
        >>> b = A*ones(A.shape[0])
        >>> ml = ruge_stuben_solver(A, max_coarse=10)
        >>> residuals = []
        >>> x = ml.solve(b, tol=1e-12, residuals=residuals) #standalone solver

        """
        
        if x0 is None:
            x = zeros_like(b)
        else:
            x = array(x0) #copy
        
        cycle = str(cycle).upper()

        if accel is not None:
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

            # wrap callback function to compute residuals
            cb = callback
            if residuals is not None:
                residuals[:] = [residual_norm(A,x,b)]
                def callback(x):
                    residuals.append(residual_norm(A,x,b))
                    if cb is not None:
                        cb(x)

            return accel(A, b, x0=x0, tol=tol, M=M, callback=callback)[0]
        
        else:
            # Scale tol by normb
            # Don't scale tol any earlier. The accel routine should also scale tol
            normb = norm(b) 
            if normb != 0:
                tol = tol*normb

        if return_residuals:
            warn('return_residuals is deprecated.  Use residuals instead')
            residuals = []
        if residuals is None:
            residuals = []
        else:
            residuals[:] = []


        # Create uniform types for A, x and b
        # Clearly, this logic doesn't handle the case of A being real and b complex
        tp = upcast(b.dtype, x.dtype, self.levels[0].A.dtype)
        [b, x] = to_type(tp, [b, x])
        
        A = self.levels[0].A

        residuals.append(residual_norm(A,x,b))

        self.first_pass=True

        while len(residuals) <= maxiter and residuals[-1]/residuals[0] > tol:
            if len(self.levels) == 1:
                # hierarchy has only 1 level
                x = self.coarse_solver(A, b)
            else:
                self.__solve(0, x, b, cycle)

            residuals.append(residual_norm(A,x,b))

            self.first_pass=False

            if callback is not None:
                callback(x)

        if return_residuals:
            return x,residuals
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
        cycle : {'V','W','F'}
            Recursively called cycling function.  The 
            Defines the cycling used:
            cycle = 'V', V-cycle
            cycle = 'W', W-cycle
            cycle = 'F', F-cycle
        """


        A = self.levels[lvl].A

        self.levels[lvl].presmoother(A,x,b)

        residual = b - A*x

        coarse_b = self.levels[lvl].R * residual
        coarse_x = zeros_like(coarse_b)

        if lvl == len(self.levels) - 2:
            coarse_x[:] = self.coarse_solver(self.levels[-1].A, coarse_b)
        else:
            if cycle == 'V':
                self.__solve(lvl + 1, coarse_x, coarse_b, 'V')
            elif cycle =='W':
                self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
                self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
            elif cycle == 'F':
                self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
                self.__solve(lvl + 1, coarse_x, coarse_b, 'V')
            else:
                raise TypeError('Unrecognized cycle type (%s)' % cycle)

        x += self.levels[lvl].P * coarse_x   #coarse grid correction

        self.levels[lvl].postsmoother(A,x,b)



#TODO support (solver,opts) format also
def coarse_grid_solver(solver):
    """Return a coarse grid solver suitable for multilevel_solver
    
    Parameters
    ----------
    solver: string
        - Sparse direct methods:
            + splu         : sparse LU solver
        - Sparse iterative methods:
            + the name of any method in scipy.sparse.linalg.isolve (e.g. 'cg')
        - Dense methods:
            + pinv     : pseudoinverse (QR)
            + pinv2    : pseudoinverse (SVD)
            + lu       : LU factorization 
            + cholesky : Cholesky factorization

    Return
    ------
    ptr : function pointer
        A method is returned for use as a standalone or coarse grids solver

    Examples
    --------
    >>> from numpy import ones
    >>> from scipy.sparse import spdiags
    >>> from pyamg.multlevel import coarse_grid_solver
    >>> n=100
    >>> e = ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n)
    >>> b = A*ones(A.shape[0])
    >>> cgs = coarse_grid_solver('LU')
    >>> x=cgs(A,b)

    TODO
    ----
    add relaxation methods
        
    """
    
    if solver in ['pinv', 'pinv2']:
        def solve(self,A,b):
            if not hasattr(self, 'P'):
                self.P = getattr(scipy.linalg, solver)( A.todense() )
            return numpy.dot(self.P,b)
    
    elif solver == 'lu':
        def solve(self,A,b):
            if not hasattr(self, 'LU'):
                self.LU = scipy.linalg.lu_factor( A.todense() )
            return scipy.linalg.lu_solve(self.LU, b)

    elif solver == 'cholesky':
        def solve(self,A,b):
            if not hasattr(self, 'L'):
                self.L = scipy.linalg.cho_factor( A.todense() )
            return scipy.linalg.cho_solve(self.L, b)
    
    elif solver == 'splu':
        def solve(self,A,b):
            if not hasattr(self, 'LU'):
                self.LU = scipy.sparse.linalg.splu( csc_matrix(A) )
            return self.LU.solve( ravel(b) )
    
    elif solver in ['bicg','bicgstab','cg','cgs','gmres','qmr','minres']:
        fn = getattr(scipy.sparse.linalg.isolve, solver)
        def solve(self,A,b):
            return fn(A, b, tol=1e-12)[0]

    elif solver is None:         
        # Identity
        def solve(self,A,b):
            return 0*b
    else:
        raise ValueError,('unknown solver: %s' % fn)
       
    def wrapped_solve(self,A,b):
        # make sure x is same dimensions and type as b
        b = asanyarray(b)
        if A.nnz==0:
            # if A.nnz = 0, then we expect no correction
            x=zeros(b.shape)
        else:
            x = solve(self,A,b)
        if isinstance(b,numpy.ndarray):
            x = asarray(x)
        elif isinstance(b,numpy.matrix):
            x = asmatrix(x)
        else:
            raise ValueError('unrecognized type')
        return x.reshape(b.shape)

    class generic_solver:
        __call__ = wrapped_solve

    return generic_solver()
