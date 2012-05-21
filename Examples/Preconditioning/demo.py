# Examples of AMG as a preconditioner

import scipy
import numpy
from pyamg.gallery import linear_elasticity, poisson
from pyamg import smoothed_aggregation_solver, rootnode_solver

# Create test cases
trials = []
A,B = poisson((500,500), format='csr'), None
trials.append( ('Poisson',A,B) )
A,B = linear_elasticity((200,200), format='bsr')
trials.append( ('Elasticity',A,B) )
   
print "Show advantages of accleration for two example problems"
choice = input('\n Input Choice:\n' + \
            '1:  Run smoothed_aggregation_solver\n' + \
            '2:  Run rootnode_solver\n' )
if choice == 1:
    method = smoothed_aggregation_solver
elif choice == 2:
    method = rootnode_solver
else:
    raise ValueError("Enter a choice of 1 or 2")


for name,A,B in trials:
    # Construct solver using AMG based on Smoothed Aggregation (SA)
    mls = method(A, B=B)

    # Display hierarchy information
    print 'Matrix: %s' % name
    print mls

    # Create random right hand side
    b = scipy.rand(A.shape[0],1)
    
    # Solve Ax=b with no acceleration ('standalone' solver)
    standalone_residuals = []
    x = mls.solve(b, tol=1e-10, accel=None, residuals=standalone_residuals)
    
    # Solve Ax=b with Conjugate Gradient (AMG as a preconditioner to CG)
    accelerated_residuals = []
    x = mls.solve(b, tol=1e-10, accel='cg', residuals=accelerated_residuals)
    
    # Compute relative residuals
    standalone_residuals  = numpy.array(standalone_residuals)/standalone_residuals[0]  
    accelerated_residuals = numpy.array(accelerated_residuals)/accelerated_residuals[0]  
    
    # Plot convergence history
    import pylab
    pylab.figure()
    pylab.title('Convergence History (%s)' % name)
    pylab.xlabel('Iteration')
    pylab.ylabel('Relative Residual')
    pylab.semilogy(standalone_residuals,  label='Standalone',  linestyle='None', marker='.')
    pylab.semilogy(accelerated_residuals, label='Accelerated', linestyle='None', marker='.')
    pylab.legend()
    print "Close window for program to proceed.\n"
    pylab.show()

