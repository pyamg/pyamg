""" 
Test the convergence of a small recirculating flow problem that generates a
nonsymmetric matrix 
"""

import numpy
import scipy

from pyamg.gallery import load_example
from pyamg import smoothed_aggregation_solver

from convergence_tools import print_cycle_history

if __name__ == '__main__':

    # Rotated Anisotropic Diffusion
    data = load_example('recirc_flow')
    A = data['A'].tocsr()
    B = data['B']
    elements = data['elements']
    vertice = data['vertices']

    numpy.random.seed(625)
    x0 = scipy.rand(A.shape[0])
    b = A*scipy.rand(A.shape[0])
    
    ##
    # For demonstration, show that a solver constructed for a symmetric
    # operator fails for this matrix. 
    smooth=('energy', {'krylov' : 'cg'})
    SA_build_args={'max_levels':10, 'max_coarse':25, 'coarse_solver':'pinv2', \
                   'symmetry':'hermitian'}
    SA_solve_args={'cycle':'V', 'maxiter':15, 'tol':1e-8}
    strength=[('ode', {'k':2, 'epsilon':4.0})]
    presmoother =('gauss_seidel', {'sweep':'symmetric', 'iterations':1})
    postsmoother=('gauss_seidel', {'sweep':'symmetric', 'iterations':1})

    ##
    # Construct solver and solve
    sa_symmetric = smoothed_aggregation_solver(A, B=B, smooth=smooth, \
                   strength=strength, presmoother=presmoother, \
                   postsmoother=postsmoother, **SA_build_args)
    resvec = []
    x = sa_symmetric.solve(b, x0=x0, residuals=resvec, **SA_solve_args)
    print "\nObserve that standard SA parameters for Hermitian systems\n" + \
          "yield a nonconvergent stand-alone solver.\n"
    print_cycle_history(resvec, sa_symmetric, verbose=True, plotting=False)

    ##
    # Now, construct and solve with nonsymmetric SA parameters 
    smooth=('energy', {'krylov' : 'gmres'})
    SA_build_args['symmetry'] = 'nonsymmetric'
    strength=[('ode', {'k':2, 'epsilon':4.0})]
    presmoother =('gauss_seidel_nr', {'sweep':'symmetric', 'iterations':1})
    postsmoother=('gauss_seidel_nr', {'sweep':'symmetric', 'iterations':1})

    ##
    # Construct solver and solve
    sa_nonsymmetric = smoothed_aggregation_solver(A, B=B, smooth=smooth, \
                   strength=strength, presmoother=presmoother, \
                   postsmoother=postsmoother, **SA_build_args)
    resvec = []
    x = sa_nonsymmetric.solve(b, x0=x0, residuals=resvec, **SA_solve_args)
    print "\n*************************************************************"
    print "*************************************************************"
    print "Now using nonsymmetric parameters for SA, we obtain a\n" +\
          "convergent stand-alone solver. \n"
    print_cycle_history(resvec, sa_nonsymmetric, verbose=True, plotting=False)

    ##
    # Now, we accelerate GMRES with the nonsymmetric solver to obtain
    # a more efficient solver
    SA_solve_args['accel'] = 'gmres'
    resvec = []
    x = sa_nonsymmetric.solve(b, x0=x0, residuals=resvec, **SA_solve_args)
    print "\n*************************************************************"
    print "*************************************************************"
    print "Now, we use the nonsymmetric solver to accelerate GMRES. \n"
    print_cycle_history(resvec, sa_nonsymmetric, verbose=True, plotting=False)


