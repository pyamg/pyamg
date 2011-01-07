#------------------------------------------------------------------
# Step 1: import scipy and pyamg packages
#------------------------------------------------------------------
from numpy import meshgrid, linspace
from scipy import rand, sin, pi
from scipy.linalg import norm
from pyamg import *
from pyamg.gallery import poisson

#------------------------------------------------------------------
# Step 2: setup up the system using pyamg.gallery
#------------------------------------------------------------------
n=200
X,Y = meshgrid(linspace(0,1,n),linspace(0,1,n))
A = poisson((n,n), format='csr')         # 2D Poisson problem on 500x500 grid
b = rand(A.shape[0])                     # pick a random right hand side

#------------------------------------------------------------------
# Step 3: setup of the multigrid hierarchy
#------------------------------------------------------------------
ml = smoothed_aggregation_solver(A)      # construct the multigrid hierarchy

#------------------------------------------------------------------
# Step 4: solve the system
#------------------------------------------------------------------
res1 = []
x = ml.solve(b, tol=1e-6, residuals=res1)# solve Ax=b to a tolerance of 1e-6

#------------------------------------------------------------------
# Step 5: print details
#------------------------------------------------------------------
print ml                                 # print hierarchy information
print "residual norm is", norm(b - A*x)  # compute norm of residual vector
print "\n\n\n\n\n"

# notice that there are 5 (or maybe 6) levels in the hierarchy
#
# we can look at the data in each of the levels
# e.g. the multigrid components on the finest (0) level
print dir(ml.levels[0])

# e.g. the multigrid components on the coarsest (4) level
print dir(ml.levels[-1])
# there are no interpoation operators (P,R) or smoothers on the coarsest level


# check the size and type of the fine level operators
print 'type = ',ml.levels[0].A.format
print '   A = ',ml.levels[0].A.shape
print '   P = ',ml.levels[0].P.shape
print '   R = ',ml.levels[0].R.shape
print "\n\n\n\n\n"


#------------------------------------------------------------------
# Step 6: change the hierarchy
#------------------------------------------------------------------
# we can also change the details of the hierarchy
ml = smoothed_aggregation_solver(A,        # the matrix
        B=X.reshape(n*n,1),                # the representation of the near null space
        BH=None,                           # the representation of the left near null space
        symmetry='hermitian',              # indicate that the matrix is Hermitian
        strength='ode',                    # change the strength of connection 
        aggregate='standard',              # use a standard aggregation method
        smooth=('jacobi', {'omega': 1.3333333333333333,'degree':2}),   # prolongation smoothing
        presmoother=('block_gauss_seidel', {'sweep': 'symmetric'}), 
        postsmoother=('block_gauss_seidel', {'sweep': 'symmetric'}), 
        Bimprove='default',                # use the default 5 sweeps of prerelaxing B at each level
        max_levels=10,                     # maximum number of levels
        max_coarse=5,                      # maximum number on a coarse level
        keep=False)                        # keep extra operators around in the hierarchy (memory)

#------------------------------------------------------------------
# Step 7: print details
#------------------------------------------------------------------
res2 = []                                # keep the residual history in the solve
x = ml.solve(b, tol=1e-12,residuals=res2)# solve Ax=b to a tolerance of 1e-12
print ml                                 # print hierarchy information
print "residual norm is", norm(b - A*x)  # compute norm of residual vector
print "\n\n\n\n\n"

#------------------------------------------------------------------
# Step 8: plot convergence history
#------------------------------------------------------------------
from pylab import *
semilogy(res1) 
hold(True)
semilogy(res2) 
show()
