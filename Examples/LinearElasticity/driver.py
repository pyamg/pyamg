# Linear Elasticity Example

import scipy
from pyamg.gallery import linear_elasticity
from pyamg import smoothed_aggregation_solver

# Create matrix and candidate vectors.  B has 3 columns, representing 
# rigid body modes of the mesh. B[:,0] and B[:,1] are translations in 
# the X and Y directions while B[:,2] is a rotation.
A,B = linear_elasticity((200,200), format='bsr')

# Construct solver using AMG based on Smoothed Aggregation (SA)
mls = smoothed_aggregation_solver(A, B=B)

# Display hierarchy information
print mls

# Create random right hand side
b = scipy.rand(A.shape[0],1)

# Solve Ax=b
residuals = []
x = mls.solve(b, tol=1e-10, residuals=residuals)

# Compute relative residuals
relative_residuals = scipy.array(residuals)/residuals[0]  

# Plot convergence
import pylab
pylab.figure()
pylab.title('Convergence History')
pylab.xlabel('Iteration')
pylab.ylabel('Relative Residual')
pylab.semilogy(relative_residuals, linestyle='None', marker='.')
pylab.show()

