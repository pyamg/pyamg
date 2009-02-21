def trimesh(vertices, indices, labels=False):
    """
    Plot a 2D triangle mesh
    """
    from scipy import asarray
    from matplotlib import collections
    from pylab import gca, axis, text
    from numpy import average
    
    print 'trimesh'
    vertices,indices = asarray(vertices),asarray(indices)

    #3d tensor [triangle index][vertex index][x/y value]
    triangles = vertices[indices.ravel(),:].reshape((indices.shape[0],3,2))
    
    col = collections.PolyCollection(triangles)
    col.set_facecolor('grey')
    col.set_alpha(0.5)
    col.set_linewidth(1)

    #sub =  subplot(111)
    sub = gca()
    sub.add_collection(col,autolim=True)
    axis('off')
    sub.autoscale_view()

    if labels:
        barycenters = average(triangles,axis=1)
        for n,bc in enumerate(barycenters):
            text(bc[0], bc[1], str(n), {'color' : 'k', 'fontsize' : 8,
                                        'horizontalalignment' : 'center',
                                        'verticalalignment' : 'center'})

########################################################################
from numpy import kron, ones, array, zeros, median, where
from scipy import rand
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import lobpcg, eigen
from pylab import gca, show, figure
from pyamg import smoothed_aggregation_solver

meshnum = 2

if meshnum==1:
    from pyamg.gallery import mesh
    V,E = mesh.uniform_tri(20,6)
if meshnum==2:
    from scipy.io import loadmat
    mesh = loadmat('crack_mesh.mat')
    V=mesh['V']
    E=mesh['E']

# build graph laplacian
Nel = E.shape[0]
Npts = E.max()+1
row = kron(range(0,Nel),[1,1,1])
col = E.ravel()
data = ones((col.size,),dtype=float)
A = coo_matrix((data,(row,col)), shape=(Nel,Npts)).tocsr()
A = A.T * A
A.data = -1*ones((A.nnz,),dtype=float)
A.setdiag(zeros((Npts,),dtype=float))
A.setdiag(-1*array(A.sum(axis=1)).ravel())
A = A.tocsr()

# construct preconditioner
ml = smoothed_aggregation_solver(A, coarse_solver='pinv2')
M = ml.aspreconditioner()

# solve for lowest two modes: constant vector and Fiedler vector
X = rand(A.shape[0], 2) 
eval,evec = lobpcg(A, X, M=None, tol=1e-12, largest=False)

# use the median of the Fiedler vector as a the separator
vmed = median(evec[:,1])
v = zeros((Npts,))
K = where(evec[:,1]>vmed)[0]
v[K]=-1
K = where(evec[:,1]>vmed)[0]
v[K]=1

# plot the mesh and partition
trimesh(V,E)
sub = gca()
sub.hold(True)
sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=v)
#sub.scatter(V[:,0],V[:,1],marker='o',s=50,c=evec[:,1])
show()
