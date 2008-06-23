"""Generic AMG solver"""

__docformat__ = "restructuredtext en"

import scipy
import numpy
from numpy import ones, zeros, zeros_like, array, asarray, empty, asanyarray, ravel
from scipy.sparse import csc_matrix

#from pyamg import relaxation
from pyamg.relaxation import *
from utils import symmetric_rescaling, diag_sparse, norm

__all__ = ['multilevel_solver', 'coarse_grid_solver']


class multilevel_solver:
    class level:
        pass

    def __init__(self, levels, preprocess=None, postprocess=None, \
            presmoother  = ('gauss_seidel', {'sweep':'symmetric'}),
            postsmoother = ('gauss_seidel', {'sweep':'symmetric'}),
            coarse_solver='pinv2'):

        self.levels = levels
        
        self.preprocess  = preprocess
        self.postprocess = postprocess

        self.coarse_solver = coarse_grid_solver(coarse_solver)
        self.presmoother  = presmoother
        self.postsmoother = postsmoother

        for level in levels[:-1]:
            if not hasattr(level, 'R'):
                level.R = level.P.T

    def __repr__(self):
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

    def operator_complexity(self):
        """number of nonzeros on all levels / number of nonzeros on the finest level"""
        return sum([level.A.nnz for level in self.levels])/float(self.levels[0].A.nnz)

    def grid_complexity(self):
        """number of unknowns on all levels / number of unknowns on the finest level"""
        return sum([level.A.shape[0] for level in self.levels])/float(self.levels[0].A.shape[0])

    def psolve(self, b):
        return self.solve(b,maxiter=1)

    def solve(self, b, x0=None, tol=1e-5, maxiter=100, callback=None, return_residuals=False):
        """
        TODO
        """

        if x0 is None:
            x = zeros_like(b)
        else:
            x = array(x0) #copy

        if self.preprocess is not None:
            x,b = self.preprocess(x,b)

        #TODO change use of tol (relative tolerance) to agree with other iterative solvers
        A = self.levels[0].A
        residuals = [ norm(b-A*x) ]

        while len(residuals) <= maxiter and residuals[-1]/residuals[0] > tol:
            if len(self.levels) == 1:
                # hierarchy has only 1 level
                x = self.coarse_solver(A,b)
            else:
                self.__solve(0,x,b)

            residuals.append( norm(b-A*x) )

            if callback is not None:
                callback(x)

        if self.postprocess is not None:
            x = self.postprocess(x)

        if return_residuals:
            return x,residuals
        else:
            return x


    def __solve(self,lvl,x,b):

        A = self.levels[lvl].A

        self.presmooth(A,x,b)

        residual = b - A*x

        coarse_b = self.levels[lvl].R * residual
        coarse_x = zeros_like(coarse_b)

        if lvl == len(self.levels) - 2:
            coarse_x[:] = self.coarse_solver(self.levels[-1].A, coarse_b)
        else:
            self.__solve(lvl + 1, coarse_x, coarse_b)

        x += self.levels[lvl].P * coarse_x   #coarse grid correction

        self.postsmooth(A,x,b)

    def presmooth(self,A,x,b):
        def unpack_arg(v):
            if isinstance(v,tuple):
                return v[0],v[1]
            else:
                return v,{}

        fn, kwargs = unpack_arg(self.presmoother)
        if fn == 'gauss_seidel':
            gauss_seidel(A, x, b, **kwargs)
        elif fn == 'kaczmarz_gauss_seidel':
            kaczmarz_gauss_seidel(A, x, b, **kwargs)
        else:
            raise TypeError('Unrecognized presmoother')
        #fn = relaxation.dispatch(self.presmoother)
        #fn(A,x,b)

    def postsmooth(self,A,x,b):
        def unpack_arg(v):
            if isinstance(v,tuple):
                return v[0],v[1]
            else:
                return v,{}
        
        fn, kwargs = unpack_arg(self.presmoother)
        if fn == 'gauss_seidel':
            gauss_seidel(A, x, b, **kwargs)
        elif fn == 'kaczmarz_gauss_seidel':
            kaczmarz_gauss_seidel(A, x, b, **kwargs)
        else:
            raise TypeError('Unrecognized postsmoother')
        #fn = relaxation.dispatch(self.postsmoother)
        #fn(A,x,b)


#TODO support (solver,opts) format also
def coarse_grid_solver(solver):
    """Return a coarse grid solver suitable for multilevel_solver
    
    Parameters
    ----------
    solver: string
        Sparse direct methods:
            splu         : sparse LU solver
        Sparse iterative methods:
            the name of any method in scipy.sparse.linalg.isolve (e.g. 'cg')
        Dense methods:
            pinv     : pseudoinverse (QR)
            pinv2    : pseudoinverse (SVD)
            lu       : LU factorization 
            cholesky : Cholesky factorization

    TODO add relaxation methods

    Examples
    --------

    TODO
        
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
       
    #TODO handle case A.nnz == 0

    def wrapped_solve(self,A,b):
        # make sure x is same dimensions and type as b
        b = asanyarray(b)
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
    

