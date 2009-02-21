from numpy import ones, zeros, median, where
from scipy import rand
from scipy.sparse.linalg import lobpcg

from pylab import gca, show, figure

from pyamg import smoothed_aggregation_solver

from helper import trimesh, graph_laplacian

meshnum = 2

if meshnum==1:
    from pyamg.gallery import mesh
    V,E = mesh.uniform_tri(20,6)
if meshnum==2:
    from scipy.io import loadmat
    mesh = loadmat('crack_mesh_2.mat')
    V=mesh['V']
    E=mesh['E']

A = graph_laplacian(V,E)

# construct preconditioner
ml = smoothed_aggregation_solver(A, coarse_solver='pinv2',max_coarse=10)
M = ml.aspreconditioner()

# solve for lowest two modes: constant vector and Fiedler vector
X = rand(A.shape[0], 2) 
(eval,evec,res) = lobpcg(A, X, M=M, tol=1e-12, largest=False, \
        verbosityLevel=1, retResidualNormsHistory=True)

fiedler = evec[:,1]

# use the median of the Fiedler vector as a the separator
vmed = median(fiedler)
v = zeros((A.shape[0],))
K = where(fiedler>vmed)[0]
v[K]=-1
K = where(fiedler>vmed)[0]
v[K]=1

# plot the mesh and partition
trimesh(V,E)
sub = gca()
sub.hold(True)
sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=v)
#sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=fiedler)
show()
