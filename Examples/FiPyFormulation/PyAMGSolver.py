from fipy.tools.pysparseMatrix import _PysparseMatrix
from fipy.solvers.solver import Solver

from numpy import array, copy
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

    def _getMatrixClass(self):
        return _PysparseMatrix
     
    def _solve(self, L, x, b):
        relres=[]
        
        # create scipy.sparse matrix view
        (data,row,col) = L._getMatrix().find()
        A = csr_matrix((data,(row,col)),shape=L._getMatrix().shape)

        # solve and deep copy data
        ml = smoothed_aggregation_solver(A,**self.MGSetupOpts)
        x[:] = ml.solve(b=b,residuals=relres,**self.MGSolveOpts)

        # fix relres and set info
        if len(relres)>0:
            relres=array(relres)/relres[0]
            info = 0
        iter = len(relres)

        if self.verbosity:
            print ml
            print 'MG iterations: %d'%iter
            print 'MG convergence factor: %g'%((relres[-1])**(1.0/iter))
            print 'MG residual history: ', relres

        self._raiseWarning(info, iter, relres)
            
    def _canSolveAssymetric(self):
        return False
