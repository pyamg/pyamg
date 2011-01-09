''' 
Example driver script for solver_diagnostics.py, which tries different
parameter combinations for smoothed_aggregation_solver(...).  The goal is to
find appropriate SA parameter settings for an arbitrary matrix.

Explore 4 different matrices: CSR matrix for basic isotropic diffusion
                              CSR matrix for basic rotated anisotropic diffusion
                              BSR matrix for basic 2D linearized elasticity
                              CSR matrix for a nonsymmetric recirculating flow problem  

Many more solver parameters may be specified than outlined in the below
examples.  Only the most basic are shown.

Run with 
    >>> python solver_diagnostics_driver.py
and examine the on-screen output and file output.
'''

from pyamg import gallery
from solver_diagnostics import solver_diagnostics
from scipy import pi
from pyamg.gallery import diffusion

##
# Try a basic isotropic diffusion problem from finite differences
# --> Only use V-cycles by specifying cycle_list
# --> Don't specify symmetry and definiteness and allow for auto-detection
A = gallery.poisson( (50,50), format='csr') 
solver_diagnostics(A, fname='iso_diff_diagnostic', cycle_list=['V'])

##
# To run the best solver found above, uncomment next two lines
#from iso_diff_diagnostic import iso_diff_diagnostic 
#iso_diff_diagnostic(A)

#################################################################################
#################################################################################

##
# Try a basic rotated anisotropic diffusion problem from bilinear finite elements
# --> Only use V-cycles by specifying cycle_list
# --> Specify symmetry and definiteness (the safest option) 
stencil = diffusion.diffusion_stencil_2d(type='FE', epsilon=0.001, theta=2*pi/16.0)
A = gallery.stencil_grid(stencil, (50,50), format='csr')
solver_diagnostics(A, fname='rot_ani_diff_diagnostic', 
                   cycle_list=['V'],
                   symmetry='symmetric', 
                   definiteness='positive')
##
# To run the best solver found above, uncomment next two lines
#from rot_ani_diff_diagnostic import rot_ani_diff_diagnostic 
#rot_ani_diff_diagnostic(A)

#################################################################################
#################################################################################

##
# Try a basic elasticity problem
# --> Try V- and W-cycles by specifying cycle_list
# --> Don't specify symmetry and definiteness and allow for auto-detection
A = gallery.linear_elasticity((30,30))[0].tobsr(blocksize=(2,2))
solver_diagnostics(A, fname='elas_diagnostic', cycle_list=['V', 'W'])

##
# To run the best solver found above, uncomment next two lines
#from elas_diagnostic import elas_diagnostic 
#elas_diagnostic(A)

#################################################################################
#################################################################################

##
# Try a basic nonsymmetric recirculating flow problem
# --> Only use V-cycles by specifying cycle_list
# --> Don't specify symmetry and definiteness and allow for auto-detection
# --> Specify the maximum coarse size and coarse grid solver with coarse_size_list
# --> Try two different Krylov wrappers and set the maximum number of iterations 
#     and halting tolerance with krylov_list.
A = gallery.load_example('recirc_flow')['A'].tocsr()
solver_diagnostics(A, fname='recirc_flow_diagnostic', 
                   cycle_list=['V'],
                   coarse_size_list = [ (15, 'pinv') ], 
                   krylov_list=[('gmres', {'tol':1e-12, 'maxiter':100}), 
                                ('bicgstab', {'tol':1e-12, 'maxiter':100})])
##
# To run the best solver found above, uncomment next two lines
#from recirc_flow_diagnostic import recirc_flow_diagnostic 
#recirc_flow_diagnostic(A)

#################################################################################
#################################################################################



