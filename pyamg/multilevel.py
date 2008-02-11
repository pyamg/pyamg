__all__ = ['multilevel_solver', 'coarse_grid_solver']

import scipy
import numpy
from numpy import ones, zeros, zeros_like, array, asarray, empty, asanyarray, ravel
from numpy.linalg import norm

from scipy.sparse import csc_matrix
from scipy.splinalg import spsolve

from relaxation import gauss_seidel,jacobi,sor
from utils import symmetric_rescaling, diag_sparse



class multilevel_solver:
    def __init__(self, As, Ps, Rs=None, preprocess=None, postprocess=None, coarse_solver=None):
        self.As = As
        self.Ps = Ps
        self.preprocess  = preprocess
        self.postprocess = postprocess

        if coarse_solver is None:
            self.coarse_solver = coarse_grid_solver('splu')
        else:
            self.coarse_solver = coarse_solver

        if Rs is None:
            self.Rs = [P.T for P in self.Ps]
        else:
            self.Rs = Rs

    def __repr__(self):
        output = 'multilevel_solver\n'
        output += 'Number of Levels:     %d\n' % len(self.As)
        output += 'Operator Complexity: %6.3f\n' % self.operator_complexity()
        output += 'Grid Complexity:     %6.3f\n' % self.grid_complexity()

        total_nnz =  sum([A.nnz for A in self.As])

        output += '  level   unknowns     nonzeros\n'
        for n,A in enumerate(self.As):
            output += '   %2d   %10d   %10d [%5.2f%%]\n' % (n,A.shape[1],A.nnz,(100*float(A.nnz)/float(total_nnz)))

        return output

    def operator_complexity(self):
        """number of nonzeros on all levels / number of nonzeros on the finest level"""
        return sum([A.nnz for A in self.As])/float(self.As[0].nnz)

    def grid_complexity(self):
        """number of unknowns on all levels / number of unknowns on the finest level"""
        return sum([A.shape[0] for A in self.As])/float(self.As[0].shape[0])


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
        A = self.As[0]
        residuals = [ norm(b-A*x) ]

        while len(residuals) <= maxiter and residuals[-1]/residuals[0] > tol:
            if len(self.As) == 1:
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
        A = self.As[lvl]

        self.presmoother(A,x,b)

        residual = b - A*x

        coarse_b = self.Rs[lvl] * residual
        coarse_x = zeros_like(coarse_b)

        if lvl == len(self.As) - 2:
            coarse_x[:] = self.coarse_solver(self.As[-1], coarse_b)
        else:
            self.__solve(lvl+1,coarse_x,coarse_b)

        x += self.Ps[lvl] * coarse_x   #coarse grid correction

        self.postsmoother(A,x,b)


    def presmoother(self,A,x,b):
        gauss_seidel(A,x,b,iterations=1,sweep="symmetric")

    def postsmoother(self,A,x,b):
        gauss_seidel(A,x,b,iterations=1,sweep="symmetric")



def coarse_grid_solver(solver):
    """return coarse grid solver suitable for multilevel_solver
    
    Parameters
    ==========
        solver: string
            Sparse methods:
                splu         : sparse LU solver

            Dense methods:
                pinv     : pseudoinverse (dense) 
                pinv2    : pseudoinverse (dense)
                lu       : LU factorization (dense)
                cholesky : Cholesky factorization (dense)

    Examples
    ========

        
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
                self.LU = scipy.splinalg.dsolve.splu( csc_matrix(A) )
            return self.LU.solve( ravel(b) )
    
    elif solver in ['bicg','bicgstab','cg','cgs','gmres','qmr','minres']:
        fn = getattr(scipy.linalg, solver)
        def solve(self,A,b):
            return fn(A, b, tol=1e-12)[0]
         
    else:
        raise ValueError,('unknown solver: %s' % solver)
       

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
    

