# Illustrates the selection of Coarse-Fine (CF) 
# splittings in Classical AMG.

import numpy
from scipy.io import loadmat
from pyamg import ruge_stuben_solver
from pyamg.gallery import load_example

data = loadmat('square.mat') #load_example('airfoil')

A = data['A']                                # matrix
V = data['vertices'][:A.shape[0]]            # vertices of each variable
E = numpy.vstack((A.tocoo().row,A.tocoo().col)).T  # edges of the matrix graph

# Use Ruge-Stuben Splitting Algorithm
mls = ruge_stuben_solver(A, max_levels=2, max_coarse=1, CF='RS')
print mls

# The CF splitting, 1 == C-node and 0 == F-node
splitting = mls.levels[0].splitting
C_nodes = splitting == 1
F_nodes = splitting == 0

from draw import lineplot
from pylab import figure, axis, scatter, show

figure(figsize=(6,6))
axis('equal')
lineplot(V, E)
scatter(V[:,0][C_nodes], V[:,1][C_nodes], c='r', s=100.0)  #plot C-nodes in red
scatter(V[:,0][F_nodes], V[:,1][F_nodes], c='b', s=100.0)  #plot F-nodes in blue
show()
 
