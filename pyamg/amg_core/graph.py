import numpy as np
from graph import cluster_node_incidence, \
    cluster_center, bellman_ford_balanced, lloyd_cluster_exact, \
    bellman_ford, lloyd_cluster
import scipy.sparse as sparse

xy = np.array([[0,2],
               [1,3],
               [1,2],
               [1,1],
               [2,0],
               [2,1],
               [2,2],
               [2,3],
               [3,2],
               [3,1],
               [3,0],
               [4,0]])
G = np.zeros((12,12))
G[0, [1,2]] = 1
G[1, [0,2,7]] = 1
G[2, [0,1,3,6]] = 1
G[3, [2,5]] = 1
G[4, [5,10]] = 1
G[5, [3,4,6,9]] = 1
G[6, [2,5,7,8]] = 1
G[7, [1,6]] = 1
G[8, [6,9]] = 1
G[9, [5,8,10,11]] = 1
G[10,[4,9,11]] = 1
G[11,[9,10]] = 1
G[np.arange(12),np.arange(12)]=[2,3,4,2,2,4,4,2,2,4,3,2]
G = sparse.csr_matrix(G)

num_nodes = 12
num_clusters = 2

cm = np.array([0,0,0,0,1,1,0,0,1,1,1,1], dtype=np.int32)




Iindptr = -1*np.ones(3,dtype=np.int32)
Iindices = -1*np.ones(num_nodes,dtype=np.int32)
L = -1*np.ones(num_nodes,dtype=np.int32)
cluster_node_incidence(num_nodes, num_clusters,
                       cm, Iindptr, Iindices, L)




Idata = np.ones(num_nodes)
I = sparse.csc_matrix((Idata, Iindices, Iindptr))

print(G.indptr, G.indices, G.data)
print("cm=",cm)
print("I.indptr=",I.indptr)
print("I.indices=",I.indices)
print("L=",L)
center = cluster_center(1,
               num_nodes,
               num_clusters,
               G.indptr,
               G.indices,
               G.data,
               cm,
               I.indptr,
               I.indices,
               L
              )

print("center=",center)
