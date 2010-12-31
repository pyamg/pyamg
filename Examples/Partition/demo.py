import numpy
import scipy
from scipy.sparse.linalg import lobpcg

import pylab

from pyamg import smoothed_aggregation_solver

from helper import trimesh, graph_laplacian

meshnum = 2

if meshnum==1:
    from pyamg.gallery import mesh
    V,E = mesh.regular_triangle_mesh(20,6)
if meshnum==2:
    from scipy.io import loadmat
    mesh = loadmat('crack_mesh.mat')
    V=mesh['V']
    E=mesh['E']

A = graph_laplacian(V,E)

# construct preconditioner
ml = smoothed_aggregation_solver(A, coarse_solver='pinv2',max_coarse=10)
M = ml.aspreconditioner()

# solve for lowest two modes: constant vector and Fiedler vector
X = scipy.rand(A.shape[0], 2) 
(eval,evec,res) = lobpcg(A, X, M=None, tol=1e-12, largest=False, \
        verbosityLevel=0, retResidualNormsHistory=True)

fiedler = evec[:,1]

# use the median of the Fiedler vector as the separator
vmed = numpy.median(fiedler)
v = numpy.zeros((A.shape[0],))
K = numpy.where(fiedler<=vmed)[0]
v[K]=-1
K = numpy.where(fiedler>vmed)[0]
v[K]=1

# plot the mesh and partition
trimesh(V,E)
sub = pylab.gca()
sub.hold(True)
sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=v)
#sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=fiedler)
pylab.show()
