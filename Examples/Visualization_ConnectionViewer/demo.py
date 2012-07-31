##
# Import ConnectionViewer writers
import numpy, scipy

from pyamg import gallery, rootnode_solver
from pyamg.gallery import stencil_grid
from pyamg.gallery.diffusion import diffusion_stencil_2d

from cvoutput import *
from convergence_tools import print_cycle_history

##
# Run Rotated Anisotropic Diffusion
n = 10
nx = n
ny = n
stencil = diffusion_stencil_2d(type='FE',epsilon=0.001,theta=scipy.pi/3)
A = stencil_grid(stencil, (nx,ny), format='csr')
numpy.random.seed(625)
x = scipy.rand(A.shape[0])
b = A*scipy.rand(A.shape[0])

ml = rootnode_solver(A, strength=('evolution', {'epsilon':2.0}), 
                    smooth=('energy', {'degree':2}), max_coarse=10)
resvec = []
x = ml.solve(b, x0=x, maxiter=20, tol=1e-14, residuals=resvec)
print_cycle_history(resvec, ml, verbose=True, plotting=False)

##
# Write ConnectionViewer files for multilevel hierarchy ml
xV,yV = numpy.meshgrid(numpy.arange(0,ny,dtype=float),numpy.arange(0,nx,dtype=float))
Verts = numpy.concatenate([[xV.ravel()],[yV.ravel()]],axis=0).T
outputML("test", Verts, ml)


print "\n\nOutput files for matrix stencil visualizations in ConnectionViewer are: \n  \
test_A*.mat \n  test_fine*.marks \n  test_coarse*.marks \n  \
test_R*.mat \n  test_P*.mat \nwhere \'*\' is the level number"
##
print "\n\nYou can download ConnectionViewer from \nhttp://gcsc.uni-frankfurt.de/Members/mrupp/connectionviewer/ \n\nWhen you open test_A0.mat with ConnectionViewer, you'll get"
##
print "\nIn ConnectionViewer, you can zoom in with the mousewheel\n \
and drag the grid around. By clicking on a node, you can see its\n \
matrix connections."

