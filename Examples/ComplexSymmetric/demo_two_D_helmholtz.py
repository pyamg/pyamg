from numpy import array, zeros, ones, sqrt, ravel, arange, abs
from scipy import rand, real, pi, imag, hstack, vstack, linspace
from scipy.linalg import svd
import pylab

from smoothed_aggregation_helmholtz_solver import smoothed_aggregation_helmholtz_solver, \
                                                  planewaves
from pyamg.util.linalg import norm
from pyamg import gallery
from convergence_tools import *
from my_vis import my_vis, shrink_elmts

if __name__ == '__main__':

    # Retrieve 2-D Helmholtz Operator and problem data.
    # This is operator was discretized with a local 
    # discontinuous Galerkin method.
    data = gallery.load_example('helmholtz_2D')
    A = data['A'].tocsr()
    omega = data['omega']
    h = data['h']
    ppw = data['ppw']
    elements = data['elements']
    vertices = data['vertices']

    print "\nRunning 2D Helmholtz Example"
    print "-- %1.2f Points-per-wavelength"%ppw
    print "-- %1.2e = h,  %1.2f = omega"%(h,omega)
    print "-- Discretized with a local discontinuous Galerkin method\n   on annulus-shaped domain"
    
    # random initial guess for zero right-hand-side
    numpy.random.seed(625)
    x0 = scipy.rand(A.shape[0])
    b = numpy.zeros_like(x0)
 
    ##
    # Use constant in B for interpolation at all levels
    use_constant=(True, {'last_level':10})
    
    # Strength -- For level 0, we aggregate based on distance so that only algebraic
    # neighbors at the same spatial location are aggregated together.  For all coarse 
    # levels, all algebraic connections are considered strong.
    strength = [('distance', {'V' : vertices, 'theta' : 1e-5, 'relative_drop' : False}),\
                ('symmetric', {'theta':0.00})]

    # Prolongator smoother
    smooth=('energy', {'krylov' : 'cgnr'})

    # Aggregation -- non-standard 'naive' aggregation is done on level 0 so that only 
    # algebraic neighbors at the same spatial location are aggregated together.
    aggregate = ['naive', 'standard']

    # Note the matrix is complex-symmetric, not Hermitian, i.e. symmetry = 'symmetric'.
    SA_build_args={'max_levels':10, 'max_coarse':50, 'coarse_solver':'pinv2', \
                   'symmetry':'symmetric'} 
    SA_solve_args={'cycle':'W', 'maxiter':20, 'tol':1e-8, 'accel' : 'gmres'}

    # Pre- and post-smoothers -- gauss_seidel is an acceptable relaxation method for
    # resolved Helmholtz problems
    smoother = [ ('gauss_seidel', {'iterations' : 4, 'sweep':'forward'}), 
                 ('gauss_seidel_nr', {'iterations' : 4, 'sweep':'forward'}) ]

    # Bimprove[k] -- stipulates the relaxation method on level k used to "improve" B
    Bimprove = [ ('gauss_seidel', {'iterations' : 2, 'sweep':'forward'}), 
                  ('gauss_seidel_nr', {'iterations' : 1, 'sweep':'forward'}) ]

    # Construct solver using the "naive" constant mode for B
    sa = smoothed_aggregation_helmholtz_solver(A, planewaves=[None], 
               use_constant=use_constant, strength=strength, smooth=smooth,
               aggregate=aggregate, Bimprove=Bimprove, presmoother=smoother,
               postsmoother=smoother, **SA_build_args) 
    
    # Solve
    residuals = []
    x = sa.solve(b, x0=x0, residuals=residuals, **SA_solve_args)
    print "\n*************************************************************"
    print "*************************************************************"
    print "Using only a constant mode for interpolation yields an inefficient\n" + \
          "solver, even for typical PyAMG nonsymmetric SA parameters.  This is\n" + \
          "due to aliasing oscillatory, but algebraically smooth, modes on the\n" + \
          "coarse levels.\n"
    print_cycle_history(residuals, sa, verbose=True, plotting=False)

    
    ###
    # Now run a solver that introduces planewaves
    ###

    # Setup planewave parameters, such that planewaves[k] defines which planewaves
    # to introduce at level[k].  The final entry of None stipulates to introduce no 
    # new planewaves from that level on down.
    X = vertices[:,0].copy()
    Y = vertices[:,1].copy()
    pwave_args = [ None, 
                   (planewaves, {'X':X, 'Y':Y, 'omega':omega, \
                     'angles':list(linspace(0., pi/2.,2)) }),
                   (planewaves, {'X':X, 'Y':Y, 'omega':omega, \
                     'angles':list(linspace(-pi/8., 5*pi/8., 4)) }),
                   None ]

    ##
    # Use constant in B for interpolation, but only between levels 0 and 1
    use_constant=(True, {'last_level':0})
    
    # Construct solver using planewaves
    sa = smoothed_aggregation_helmholtz_solver(A, planewaves=pwave_args, 
               use_constant=use_constant, strength=strength, smooth=smooth,
                aggregate=aggregate, Bimprove=Bimprove, presmoother=smoother,
                postsmoother=smoother, **SA_build_args) 
   
    # Solve
    residuals = []
    x = sa.solve(b, x0=x0, residuals=residuals, **SA_solve_args)
    print "\n*************************************************************"
    print "*************************************************************"
    print "Note the improved performance from using planewaves in B.\n"
    print_cycle_history(residuals, sa, verbose=True, plotting=False)

    
    elements2,vertices2 =  shrink_elmts(elements, vertices)
    my_vis(sa, vertices2, error=abs(x), fname='helmholtz_', E2V=elements2)

