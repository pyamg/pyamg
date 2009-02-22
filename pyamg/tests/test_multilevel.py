from pyamg.testing import *

from numpy import matrix, array, diag, arange, ones
from scipy import rand
from scipy.linalg import norm
from scipy.sparse import csr_matrix

from pyamg.gallery import poisson
from pyamg.multilevel import *

class TestMultilevel(TestCase):
    def test_coarse_grid_solver(self):
        cases = []

        cases.append( csr_matrix(diag(arange(1,5))) )
        cases.append( poisson( (4,),  format='csr') )
        cases.append( poisson( (4,4), format='csr') )
      
        from pyamg.krylov import cg
        def fn(A, b):
            return cg(A, b)[0]

        # method should be almost exact for small matrices
        for A in cases:
            for solver in ['splu', 'pinv', 'pinv2', 'lu', 'cholesky', 'cg', fn]:
                s = coarse_grid_solver(solver)

                b = arange(A.shape[0], dtype=A.dtype)

                x = s(A,b)
                assert_almost_equal(A*x, b)

                # subsequent calls use cached data
                x = s(A,b)  
                assert_almost_equal(A*x, b)

    def test_aspreconditioner(self):
        from pyamg import smoothed_aggregation_solver
        from scipy.sparse.linalg import cg

        A = poisson((100,100), format='csr')
        b = rand(A.shape[0])

        ml = smoothed_aggregation_solver(A)

        for cycle in ['V','W','F']:
            M = ml.aspreconditioner(cycle='V')
            x,info = cg(A, b, tol=1e-8, maxiter=30, M=M)
            assert(norm(b - A*x) < 1e-8*norm(b))


    def test_accel(self):
        from pyamg import smoothed_aggregation_solver
        from pyamg.krylov import cg, bicgstab

        A = poisson((100,100), format='csr')
        b = rand(A.shape[0])

        ml = smoothed_aggregation_solver(A)

        for accel in ['cg', 'bicgstab', 'cgs', cg, bicgstab]:
            x = ml.solve(b, maxiter=30, tol=1e-8, accel=accel)
            assert(norm(b - A*x) < 1e-8*norm(b))
            residuals = []
            x = ml.solve(b, maxiter=30, tol=1e-8, residuals=residuals, accel=accel)
            assert(norm(b - A*x) < 1e-8*norm(b))
            #print residuals
            assert_almost_equal(norm(b - A*x), residuals[-1])

    def test_cycle_complexity(self):
        # four levels
        levels = []
        levels.append( multilevel_solver.level() )
        levels[0].A = csr_matrix(ones((10,10)))
        levels[0].P = csr_matrix(ones((10,5)))
        levels.append( multilevel_solver.level() )
        levels[1].A = csr_matrix(ones((5,5)))
        levels[1].P = csr_matrix(ones((5,3)))
        levels.append( multilevel_solver.level() )
        levels[2].A = csr_matrix(ones((3,3)))
        levels[2].P = csr_matrix(ones((3,2)))
        levels.append( multilevel_solver.level() )
        levels[3].A = csr_matrix(ones((2,2)))

        # one level hierarchy
        mg = multilevel_solver(levels[:1])
        assert_equal(mg.cycle_complexity(cycle='V'), 100.0/100.0) #1
        assert_equal(mg.cycle_complexity(cycle='W'), 100.0/100.0) #1
        assert_equal(mg.cycle_complexity(cycle='F'), 100.0/100.0) #1

        # two level hierarchy
        mg = multilevel_solver(levels[:2])
        assert_equal(mg.cycle_complexity(cycle='V'), 225.0/100.0) #2,1
        assert_equal(mg.cycle_complexity(cycle='W'), 225.0/100.0) #2,1
        assert_equal(mg.cycle_complexity(cycle='F'), 225.0/100.0) #2,1

        # three level hierarchy
        mg = multilevel_solver(levels[:3])
        assert_equal(mg.cycle_complexity(cycle='V'), 259.0/100.0) #2,2,1
        assert_equal(mg.cycle_complexity(cycle='W'), 318.0/100.0) #2,4,2
        assert_equal(mg.cycle_complexity(cycle='F'), 318.0/100.0) #2,4,2
        
        # four level hierarchy
        mg = multilevel_solver(levels[:4])
        assert_equal(mg.cycle_complexity(cycle='V'), 272.0/100.0) #2,2,2,1
        assert_equal(mg.cycle_complexity(cycle='W'), 388.0/100.0) #2,4,8,4
        assert_equal(mg.cycle_complexity(cycle='F'), 366.0/100.0) #2,4,6,3


class TestComplexMultilevel(TestCase):
    def test_coarse_grid_solver(self):
        cases = []

        cases.append( csr_matrix(diag(arange(1,5))) )
        cases.append( poisson( (4,),  format='csr') )
        cases.append( poisson( (4,4), format='csr') )
        
        # Make cases complex
        cases = [ G+1e-5j*G for G in cases ]
        cases = [ 0.5*(G + G.H) for G in cases ]
        
        # method should be almost exact for small matrices
        for A in cases:
            for solver in ['splu','pinv','pinv2','lu','cholesky','cg']:
                s = coarse_grid_solver(solver)

                b = arange(A.shape[0],dtype=A.dtype)

                x = s(A,b)
                assert_almost_equal( A*x, b)

                # subsequent calls use cached data
                x = s(A,b)  
                assert_almost_equal( A*x, b)


