import numpy
from numpy import ones, arange
from scipy.sparse import spdiags, coo_matrix
from pydec import triangulate_ncube,cube_grid,simplicial_complex,triplot

from pyamg.graph import *

#v,s = simplicial_grid_2d(50)

v,s = triangulate_ncube(*cube_grid((50,50)))

sc = simplicial_complex((v,s))
G = sc[1].d.T.tocsr() * sc[1].d
#G = spdiags([G.diagonal()],[0],G.shape[0],G.shape[1],format=G.format) - G
v = (v[sc[1].simplices[:,0],:] + v[sc[1].simplices[:,1],:]) / 2.0
G = G.astype(float)
G.data[:] = 1

#random permutation
#P = coo_matrix((ones(G.shape[0]),(arange(G.shape[0]),numpy.random.permutation(G.shape[0])))).tocsr()
#G = P.T.tocsr() * G * P
#v[P.indices,:] = v.copy()

seeds =  G.shape[0] / 30 #maximal_independent_set(G)
distances, clusters, seeds = lloyd_cluster(G, seeds)

from pylab import *
#plot(numpy.bincount(distances.astype('i')),'.')
#show()

G_coo = G.tocoo()
perimeter_mask = zeros(G.shape[0],dtype='bool')
perimeter_mask[G_coo.row[(clusters[G_coo.row] != clusters[G_coo.col])]] = True

figure()
#triplot(v,s)
area = 50
scatter(v[:,0],v[:,1], area, clusters)
scatter(v[seeds,0], v[seeds,1], area, '0.0')
#scatter(v[perimeter_mask,0],v[perimeter_mask,1],area/10,'1.0')        
show()

