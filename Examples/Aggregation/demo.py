# Illustrates the selection of aggregates in AMG based on smoothed aggregation

import numpy
from scipy.io import loadmat
from pyamg import smoothed_aggregation_solver
from pyamg.gallery import load_example

data = loadmat('square.mat')

A = data['A'].tocsr()                        # matrix
V = data['vertices'][:A.shape[0]]            # vertices of each variable
E = numpy.vstack((A.tocoo().row,A.tocoo().col)).T  # edges of the matrix graph

# Use Smoothed Aggregation Algorithm
mls = smoothed_aggregation_solver(A, max_levels=2, max_coarse=1, keep=True)

# AggOp[i,j] is 1 iff node i belongs to aggregate j
AggOp = mls.levels[0].AggOp

# determine which edges lie entirely inside an aggregate
inner_edges = AggOp.indices[E[:,0]] == AggOp.indices[E[:,1]]  #AggOp.indices[n] is the aggregate to which vertex n belongs
outer_edges = -inner_edges

from draw import lineplot
from pylab import figure, axis, scatter, show

figure(figsize=(6,6))
axis('equal')
lineplot(V, E[inner_edges], linewidths=3.0)
lineplot(V, E[outer_edges], linewidths=0.2)
scatter(V[:,0], V[:,1], c='r', s=100.0)
#scatter(V[:,0], V[:,1], c=numpy.ones(V.shape[0]), s=100.0)
show()
 
