# Linear Elasticity Example

import scipy
from pyamg.gallery import linear_elasticity
from pyamg import smoothed_aggregation_solver, rootnode_solver
from convergence_tools import print_cycle_history
    
print "Test convergence for a simple 200x200 Grid, Linearized Elasticity Problem"
choice = input('\n Input Choice:\n' + \
           '1:  Run smoothed_aggregation_solver\n' + \
           '2:  Run rootnode_solver\n' )

# Create matrix and candidate vectors.  B has 3 columns, representing 
# rigid body modes of the mesh. B[:,0] and B[:,1] are translations in 
# the X and Y directions while B[:,2] is a rotation.
A,B = linear_elasticity((200,200), format='bsr')

# Construct solver using AMG based on Smoothed Aggregation (SA)
if choice == 1:
    mls = smoothed_aggregation_solver(A, B=B, smooth='energy')
elif choice == 2:
    mls = rootnode_solver(A, B=B, smooth='energy')
else:
    raise ValueError("Enter a choice of 1 or 2")

# Display hierarchy information
print mls

# Create random right hand side
b = scipy.rand(A.shape[0],1)

# Solve Ax=b
residuals = []
x = mls.solve(b, tol=1e-10, residuals=residuals)
print "Number of iterations:  %d\n"%len(residuals)

# Output convergence
print_cycle_history(residuals, mls, verbose=True, plotting=True)
