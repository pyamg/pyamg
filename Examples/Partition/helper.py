def trimesh(vertices, indices, labels=False):
    """
    Plot a 2D triangle mesh
    """
    from scipy import asarray
    from matplotlib import collections
    from pylab import gca, axis, text
    from numpy import average
    
    vertices,indices = asarray(vertices),asarray(indices)

    #3d tensor [triangle index][vertex index][x/y value]
    triangles = vertices[indices.ravel(),:].reshape((indices.shape[0],3,2))
    
    col = collections.PolyCollection(triangles)
    col.set_facecolor('grey')
    col.set_alpha(0.5)
    col.set_linewidth(1)

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

def graph_laplacian(V,E):
    # build graph Laplacian
    from numpy import kron, ones, zeros, array
    from scipy.sparse import coo_matrix
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
    return A.tocsr()
