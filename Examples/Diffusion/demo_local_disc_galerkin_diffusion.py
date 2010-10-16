"""

Test the convergence of a small diffusion problem discretized with the local
discontinuous Galerkin method.  The polynomial order is 5.  To utilize the
visualization capabilities, you need to have paraview and scikits.delaunay
installed.

"""
import numpy
import scipy

from pyamg.gallery import load_example
from pyamg import smoothed_aggregation_solver

from convergence_tools import print_cycle_history

if __name__ == '__main__':

    print "\nDiffusion problem discretized with p=5 and the local\n" + \
          "discontinuous Galerkin method."

    # Discontinuous Galerkin Diffusion Problem
    data = load_example('local_disc_galerkin_diffusion')
    A = data['A'].tocsr()
    B = data['B']
    elements = data['elements']
    vertices = data['vertices']
    numpy.random.seed(625)
    x0 = scipy.rand(A.shape[0])
    b = numpy.zeros_like(x0)
    
    ##
    # For demonstration, show that a naive SA solver 
    # yields unsatisfactory convergence
    smooth=('jacobi', {'filter' : True})
    strength=('symmetric', {'theta' : 0.1})
    SA_solve_args={'cycle':'W', 'maxiter':20, 'tol':1e-8, 'accel' : 'cg'}
    SA_build_args={'max_levels':10, 'max_coarse':25, 'coarse_solver':'pinv2', \
                   'symmetry':'hermitian'}
    presmoother =('gauss_seidel', {'sweep':'symmetric', 'iterations':1})
    postsmoother=('gauss_seidel', {'sweep':'symmetric', 'iterations':1})
    
    ##
    # Construct solver and solve
    sa = smoothed_aggregation_solver(A, B=B, smooth=smooth, \
             strength=strength, presmoother=presmoother, \
             postsmoother=postsmoother, **SA_build_args)
    resvec = []
    x = sa.solve(b, x0=x0, residuals=resvec, **SA_solve_args)
    print "\n*************************************************************"
    print "*************************************************************"
    print "Observe that standard SA parameters for this p=5 discontinuous \n" + \
          "Galerkin system yield an inefficient solver.\n"
    print_cycle_history(resvec, sa, verbose=True, plotting=False)

    ##
    # Now, construct and solve with appropriate parameters 
    p = 5
    Bimprove = [('block_gauss_seidel', {'sweep':'symmetric', 'iterations':p}),
                ('gauss_seidel', {'sweep':'symmetric', 'iterations':p})]   
    aggregate = ['naive', 'standard']
    # the initial conforming aggregation step requires no prolongation smoothing
    smooth=[None, ('energy', {'krylov' : 'cg', 'maxiter' : p})]
    strength =[('distance', {'V' : data['vertices'], 'theta':5e-5, 'relative_drop':False}),\
               ('ode', {'k':4, 'proj_type':'l2', 'epsilon':2.0})]
    sa = smoothed_aggregation_solver(A, B=B, smooth=smooth, Bimprove=Bimprove,\
             strength=strength, presmoother=presmoother, aggregate=aggregate,\
             postsmoother=postsmoother, **SA_build_args)
    resvec = []
    x = sa.solve(b, x0=x0, residuals=resvec, **SA_solve_args)
    print "\n*************************************************************"
    print "*************************************************************"
    print "Now use appropriate parameters, especially \'energy\' prolongation\n" + \
          "smoothing and a distance based strength measure on level 0.  This\n" + \
          "yields a much more efficient solver.\n"
    print_cycle_history(resvec, sa, verbose=True, plotting=False)


##
# check for scikits and print message about needing to have paraview in order
# to view the visualization files
try:
    from scikits import delaunay
except:
    print "Must install scikits.delaunay to generate the visualization files (.vtu for paraview)."
# generate visualization files    
print "\n\n*************************************************************"
print "*************************************************************"
print "Generating visualization files in .vtu format for use with paraview."
print "\nAll values from coarse levels are interpolated using the aggregates,\n" +\
      "i.e., there is no fixed geometric hierarchy.  Additionally, the mesh\n" +\
      "has been artificially shrunk towards each element's barycenter, in order\n" +\
      "to highlight the discontinuous nature of the discretization.\n"
print "-- Near null-space mode from level * is in the file\n"+\
      "   DG_Example_B_variable0_lvl*.vtu"
print "-- Aggregtes from level * are in the two file\n"+\
      "   DG_Example_aggs_lvl*_point-aggs,  and \n"+\
      "   DG_Example_aggs_lvl*_aggs.vtu"
print "-- The mesh from from level * is in the file\n"+\
      "   DG_Example_mesh_lvl*.vtu"
print "-- The error is in file\n"+\
      "   DG_Example_error_variable0.vtu"
print ""
from my_vis import shrink_elmts, my_vis
elements2,vertices2 =  shrink_elmts(elements, vertices)
my_vis(sa, vertices2, error=x, fname="DG_Example_", E2V=elements2[:,0:3])




