from numpy import array, zeros, ones, sqrt, ravel, arange
from scipy import rand, real, pi, imag, hstack, vstack
from scipy.linalg import svd
import pylab

from pyamg import smoothed_aggregation_solver
from pyamg.util.linalg import norm

from one_D_helmholtz import *
from convergence_tools import *

if __name__ == '__main__':

    # Problem parameters
    h = 1024
    mesh_h =  1.0/(float(h)-1.0)
    points_per_wavelength = 15.0
       
    # Retrieve 1-D Helmholtz Operator 
    omega = (2*pi)/(mesh_h*points_per_wavelength)
    data = one_D_helmholtz(h, omega=omega, nplane_waves=2)
    A = data['A']             
    B = data['B']
    vertices = data['vertices']
    numpy.random.seed(625)
    x0 = scipy.rand(A.shape[0])
    b = numpy.zeros_like(x0)
    
    # Solver Parameters, note the solver is complex symmetric, not Hermitian.
    # Hence, symmetry = 'symmetric'.
    smooth=('energy', {'krylov' : 'gmres'})
    SA_solve_args={'cycle':'W', 'maxiter':20, 'tol':1e-8, 'accel' : 'gmres'}
    SA_build_args={'max_levels':10, 'max_coarse':5, 'coarse_solver':'pinv2', \
                   'symmetry':'symmetric'}
    smoother =('gauss_seidel_nr', {'sweep':'symmetric', 'iterations':1})

    # Construct solver using the "naive" constant mode for B
    sa = smoothed_aggregation_solver(A, B = ones((A.shape[0],1)), 
                strength=('symmetric', {'theta':0.0}), presmoother=smoother, 
                postsmoother=smoother, smooth=smooth, **SA_build_args)  
    
    # Solve
    residuals = []
    x = sa.solve(b, x0=x0, residuals=residuals, **SA_solve_args)
    print "\n*************************************************************"
    print "*************************************************************"
    print "Using only a constant mode for interpolation yields an inefficient\n" + \
          "solver, even for typical PyAMG nonsymmetric SA parameters"
    print_cycle_history(residuals, sa, verbose=True, plotting=False)

    # Construct solver using the wave-like modes for B
    sa = smoothed_aggregation_solver(A, B = B,
                strength=('symmetric', {'theta':0.0}), presmoother=smoother, 
                postsmoother=smoother, smooth=smooth, **SA_build_args)  
    
    # Solve
    residuals = []
    x = sa.solve(b, x0=x0, residuals=residuals, **SA_solve_args)
    print "\n*************************************************************"
    print "*************************************************************"
    print "Note the improved performance from wave-like interpolation."
    print_cycle_history(residuals, sa, verbose=True, plotting=False)

    
    # plot parameters
    fig_width_pt=300.
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    ratio = 2.95
    fig_size =  [ratio*fig_width,ratio*fig_height]
    params = {'backend': 'ps',
              'axes.labelsize':  20,
              'text.fontsize':   20,
              'axes.titlesize' : 20,
              'legend.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': True,
              'figure.figsize': fig_size,
              'lines.linewidth' : 4}
    pylab.rcParams.update(params)
    
    # plot B vs. the lowest right singular vector, which represents
    # the near null-space, for a segment of the domain
    pylab.figure(1)
    indys = arange(0,min(75,h))
    line_styles = [ "-b", "--m", ":k"]
    leg = []
    for i in range(B.shape[1]):
        pylab.plot(vertices[indys,0], real(B[indys,i]), line_styles[i])
        leg.append('NNS Mode %d'%i)
    [U,S,V] = svd(A.todense())
    V = V.T.copy()
    scale = 0.9/max(real(V[indys,-1]))
    pylab.plot(vertices[indys,0], scale*real(ravel(V[indys,-1])), line_styles[i+1])
    leg.append('Re$(\\nu)$')
     
    pylab.title('Near Null-Space Modes (NNS) vs. Lowest Right Singular Vector $\\nu$')
    pylab.legend(leg) 
    pylab.show()
     
