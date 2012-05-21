# Illustrates the selection of aggregates in AMG based on smoothed aggregation

import numpy
from pyamg import rootnode_solver
from pyamg.gallery import load_example

data = load_example('unit_square')

A = data['A'].tocsr()                        # matrix
V = data['vertices'][:A.shape[0]]            # vertices of each variable
E = numpy.vstack((A.tocoo().row,A.tocoo().col)).T  # edges of the matrix graph

# Use Root-Node Solver
mls = rootnode_solver(A, max_levels=2, max_coarse=1, keep=True)

# AggOp[i,j] is 1 iff node i belongs to aggregate j
AggOp = mls.levels[0].AggOp

# determine which edges lie entirely inside an aggregate
# AggOp.indices[n] is the aggregate to which vertex n belongs
inner_edges = AggOp.indices[E[:,0]] == AggOp.indices[E[:,1]]  
outer_edges = -inner_edges

# Grab the root-nodes (i.e., the C/F splitting)
Cpts = mls.levels[0].Cpts
Fpts = mls.levels[0].Fpts

from draw import lineplot
from pylab import figure, axis, scatter, show, title

##
# Plot the aggregation
figure(figsize=(6,6))
title('Finest-Level Aggregation\nC-pts in Red, F-pts in Blue')
axis('equal')
lineplot(V, E[inner_edges], linewidths=3.0)
lineplot(V, E[outer_edges], linewidths=0.2)
scatter(V[:,0][Fpts], V[:,1][Fpts], c='b', s=100.0)  #plot F-nodes in blue
scatter(V[:,0][Cpts], V[:,1][Cpts], c='r', s=220.0)  #plot C-nodes in red

##
# Plot the C/F splitting
figure(figsize=(6,6))
title('Finest-Level C/F splitting\nC-pts in Red, F-pts in Blue')
axis('equal')
lineplot(V, E)
scatter(V[:,0][Cpts], V[:,1][Cpts], c='r', s=100.0)  #plot C-nodes in red
scatter(V[:,0][Fpts], V[:,1][Fpts], c='b', s=100.0)  #plot F-nodes in blue

show()
 
