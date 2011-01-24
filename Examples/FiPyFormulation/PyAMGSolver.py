from fipy.matrices.pysparseMatrix import _PysparseMeshMatrix
from fipy.solvers.solver import Solver

import numpy
from scipy.sparse import csr_matrix
from pyamg import smoothed_aggregation_solver

class PyAMGSolver(Solver):
    """
    The PyAMGSolver class.
    """
    def __init__(self, *args, **kwargs):
        if kwargs.has_key('verbosity'):
            self.verbosity = kwargs['verbosity']
        else:
            self.verbosity = False

        if kwargs.has_key('MGSetupOpts'):
            self.MGSetupOpts = kwargs['MGSetupOpts']
        else:
            self.MGSetupOpts = {}

        if kwargs.has_key('MGSolveOpts'):
            self.MGSolveOpts = kwargs['MGSolveOpts']
        else:
            self.MGSolveOpts = {}

    @property
    def _matrixClass(self):
        return _PysparseMeshMatrix

    def _solve_(self, L, x, b):
        relres=[]
        
        # create scipy.sparse matrix view
        (data,row,col) = L.matrix.find()
        A = csr_matrix((data,(row,col)),shape=L.matrix.shape)

        # solve and deep copy data
        ml = smoothed_aggregation_solver(A,**self.MGSetupOpts)
        x[:] = ml.solve(b=b,residuals=relres,**self.MGSolveOpts)

        # fix relres and set info
        if len(relres)>0:
            relres=numpy.array(relres)/relres[0]
            info = 0
        iter = len(relres)

        if self.verbosity:
            print ml
            print 'MG iterations: %d'%iter
            print 'MG convergence factor: %g'%((relres[-1])**(1.0/iter))
            print 'MG residual history: ', relres

        self._raiseWarning(info, iter, relres)

    def _solve(self):
        self._solve_(self.matrix, self.var.numericValue, self.RHSvector)
            
    def _canSolveAssymetric(self):
        return False
