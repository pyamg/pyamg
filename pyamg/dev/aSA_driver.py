import pdb

import time
import math
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from pyamg.gallery import poisson
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery.stencil import stencil_grid
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
from pyamg.aggregation.adaptive import adaptive_sa_solver
from pyamg.aggregation.new_adaptive import asa_solver
from pyamg.util.utils import symmetric_rescaling



# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# General multilevel parameters
# -----------------------------
max_levels 		   = 20 		# Max levels in hierarchy
max_coarse 		   = 100 		# Max points allowed on coarse grid
tol 			   = 1e-8		# Residual convergence tolerance
is_pdef 		   = True		# Assume matrix positive definite (only for aSA)
keep_levels 	   = False		# Also store SOC, aggregation, and tentative P operators
diagonal_dominance = False		# Avoid coarsening diagonally dominant rows 
coarse_solver = 'pinv'
accel = None

# AMG CF-splitting 
# ----------------
#
#
#
#
#
#
# CF = 

# Strength of connection 
# ----------------------
#	- symmetric, strong connection if |A[i,j]| >= theta * sqrt( |A[i,i]| * |A[j,j]| )
#		+ theta (0)
#	- classical, strong connection if |A[i,j]| >= theta * max( |A[i,k]| )
#		+ theta (0)
# 	- evolution
#		+ epsilon (4)- drop tolerance, > 1. Larger e -> denser matrix. 
#		+ k (2)- ODE num time steps, step size = 1/rho(DinvA)
#		+ block_flag (F)- True / False for block matrices
#		+ symmetrize_measure (T)- True / False, True --> Atilde = 0.5*(Atilde + Atilde.T)
#		+ proj_type (l2)- Define norm for constrained min prob, l2 or D_A
strength_connection = ('symmetric', {'theta': 0} )


# Aggregation 
# -----------
#	- standard
#	- naive
#		+ Differs from standard - "Each dof is considered. If it has been aggregated
# 		  skip over. Otherwise, put dof and any unaggregated neighbors in an aggregate.
#   	  Results in possibly much higher complexities than standard aggregation." 
#	- lloyd (don't know how this works...)
#		+ ratio (0.03)- fraction of the nodes which will be seeds.
#		+ maxiter (10)- maximum number iterations to perform
#		+ distance (unit)- edge weight of graph G used in Lloyd clustering.
#		  For each C[i,j]!=0,
#	    	~ unit - G[i,j] = 1
#	        ~ abs  - G[i,j] = abs(C[i,j])
#	        ~ inv  - G[i,j] = 1.0/abs(C[i,j])
#	        ~ same - G[i,j] = C[i,j]
#	        ~ sub  - G[i,j] = C[i,j] - min(C)
aggregation = ('standard')


# Interpolation smooother (Jacobi seems slow...)
# -----------------------
# 	- richardson
#		+ omega (4/3)- weighted Richardson w/ weight omega
#		+ degree (1)- number of Richardson iterations 
# 	- jacobi
#		+ omega (4/3)- weighted Jacobi w/ weight omega
#		+ degree (1)- number of Jacobi iterations 
#		+ filter (F)- True / False, filter smoothing matrix S w/ nonzero 
#		  indices of SOC matrix. Can greatly control complexity? (appears to slow convergence)
#		+ weighting (diagonal)- construction of diagonal preconditioning
#			~ local - local row-wise weight, avoids under-damping (appears to slow convergence)
#			~ diagonal - inverse of diagonal of A
#			~ block - block diagonal inverse for A, USE FOR BLOCK SYSTEMS
# 	- energy
#		+ krylov (cg)- descent method for energy minimization
#			~ cg - use cg for SPD systems. 
#			~ cgnr - use for nonsymmetric or indefinite systems.
#			  Only supports diagonal weighting.
#			~ gmres - use for nonsymmetric or indefinite systems.
#		+ degree (1)- sparsity pattern for P based on (Atilde^degree T).
#		+ maxiter (4)- number of energy minimization steps to apply to P.
#		+ weighting (local)- construction of diagonal preconditioning
#			~ local - local row-wise weight, avoids under-damping 
#			~ diagonal - inverse of diagonal of A
#			~ block - block diagonal inverse for A, USE FOR BLOCK SYSTEMS
# interp_smooth = ('jacobi', {'omega': 3.0/3.0, 'filter': True, 'weighting':'local'} )
# interp_smooth = ('jacobi', {'omega': 3.0/3.0 } )
interp_smooth = ('richardson', {'omega': 3.0/2.0, 'degree': 3} )


# Relaxation
# ---------- 
# 	- jacobi
#		+ omega (1.0)- damping parameter.
#		+ iterations (1)- number of iterations to perform.
# 	- gauss_seidel
#		+ iterations (1)- number of iterations to perform.
#		+ sweep (forward)- direction of relaxation sweep.
#			~ forward
#			~ backward
#			~ symmetric
# 	- sor
#		+ omega - damping parameter. If omega = 1.0, SOR <--> Gauss-Seidel.
#		+ iterations (1)- number of iterations to perform.
#		+ sweep (forward)- direction of relaxation sweep.
#			~ forward
#			~ backward
#			~ symmetric
# 	- block_jacobi
#		+ omega (1.0)- damping parameter.
#		+ iterations (1)- number of relaxation iterations.
#		+ blocksize (1)- block size of bsr matrix
#		+ Dinv (None)- Array holding block diagonal inverses of A.
#		  size (numBlocks, blocksize, blocksize)
#	- block_gauss_seidel
#		+ iterations (1)- number of relaxation iterations.
#		+ sweep (forward)- direction of relaxation sweep.
#			~ forward
#			~ backward
#			~ symmetric
#		+ blocksize (1)- block size of bsr matrix
#		+ Dinv (None)- Array holding block diagonal inverses of A.
#		  size (numBlocks, blocksize, blocksize)
#
# Note, Schwarz relaxation, polynomial relaxation, Cimmino relaxation,
# Kaczmarz relaxation, indexed Gauss-Seidel, and one other variant of 
# Gauss-Seidel are also available - see relaxation.py. 
# relaxation = ('jacobi', {'omega': 3.0/3.0, 'iterations': 1} )
relaxation = ('gauss_seidel', {'sweep': 'forward', 'iterations': 1} )
# relaxation = ('richardson', {'iterations': 1})


# Adaptive parameters
# -------------------
candidate_iters		= 5 	# number of smoothings/cycles used at each stage of adaptive process
num_candidates 		= 1		# number of near null space candidated to generate
improvement_iters 	= 2		# number of times a target bad guy is improved
target_convergence	= 0.3 	# target convergence factor, called epsilon in adaptive solver input
eliminate_local		= (False, {'Ca': 1.0})	# aSA, supposedly not useful I think

# New adaptive parameters
# -----------------------
weak_tol 		 	 = 15.0			# new aSA 
local_weak_tol 		 = 15.0			# new aSA
min_targets			 = 1
max_targets			 = 4
max_level_iterations = 5
coarse_size			 = 100


# from SA --> WHY WOULD WE DEFINE THIS TO BE DIFFERENT THAN THE RELAXATION SCHEME USED??
improve_candidates = ('gauss_seidel', {'sweep': 'symmetric', 'iterations': 4})
# improve_candidates = ('jacobi', {'omega': 3.0/3.0, 'iterations': 4})
# improve_candidates = ('richardson', {'omega': 3.0/2.0, 'iterations': 4} )

bad_guy 	= None


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Problem parameters and variables
# ---------------------------------

rand_guess 	= True
zero_rhs 	= True
problem_dim = 2
N 			= 100
epsilon 	= 0.1 				# 'Strength' of aniostropy (only for 2d)
theta 		= 4.0*math.pi/16.0	# Angle of anisotropy (only for 2d)

# Empty arrays to store residuals
sa_residuals = []
asa_residuals = []
new_asa_residuals = []

# 1d Poisson 
if problem_dim == 1:
	grid_dims = [N,1]
	A = poisson((N,), format='csr')
# 2d Poisson
elif problem_dim == 2:
	grid_dims = [N,N]
	stencil = diffusion_stencil_2d(epsilon,theta)
	A = stencil_grid(stencil, grid_dims, format='csr')

# # Vectors and additional variables
[d,d,A] = symmetric_rescaling(A)
vec_size = np.prod(grid_dims)

# Zero right hand side or sin(pi x)
if zero_rhs:
	b = np.zeros((vec_size,1))
	# If zero rhs and zero initial guess, throw error
	if not rand_guess:
		print "Zero rhs and zero initial guess converges trivially."
# Note, this vector probably doesn't make sense in 2d... 
else: 
	b = np.sin(math.pi*np.arange(0,vec_size)/(vec_size-1.0))

# Random vs. zero initial guess
if rand_guess:
	x0 = np.random.rand(vec_size,1)
else:
	x0 = np.zeros(vec_size,1)

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Classical SA solver
# -------------------

start = time.clock()
ml_sa = smoothed_aggregation_solver(A, B=bad_guy, strength=strength_connection, aggregate=aggregation,
						 			smooth=interp_smooth, max_levels=max_levels, max_coarse=max_coarse,
						 			presmoother=relaxation, postsmoother=relaxation,
						 			improve_candidates=improve_candidates, coarse_solver=coarse_solver,
						 			keep=keep_levels )

sa_sol = ml_sa.solve(b, x0, tol, residuals=sa_residuals)

end = time.clock()
sa_time = end-start
sa_conv_factors = np.zeros((len(sa_residuals)-1,1))
for i in range(0,len(sa_residuals)-1):
	sa_conv_factors[i] = sa_residuals[i]/sa_residuals[i-1]

print "Smoothed aggregation - ", sa_time, " seconds"
print sa_conv_factors

pdb.set_trace()

cc = ml_sa.cycle_complexity()
print "Cycle complexity = ",cc

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# Classical aSA solver
# --------------------

# start = time.clock()
# [ml_asa, work] = asa_solver(A, B=bad_guy, pdef=is_pdef, num_candidates=num_candidates,
# 									candidate_iters=candidate_iters, improvement_iters=improvement_iters,
# 									epsilon=target_convergence, max_levels=max_levels, max_coarse=max_coarse,
# 									aggregate=aggregation, prepostsmoother=relaxation, smooth=interp_smooth,
# 									strength=strength_connection, coarse_solver=coarse_solver,
# 									eliminate_local=(False, {'Ca': 1.0}), keep=keep_levels)

# asa_sol = ml_asa.solve(b, x0, tol, residuals=asa_residuals)

# end = time.clock()
# asa_time = end-start
# asa_conv_factors = np.zeros((len(asa_residuals)-1,1))
# for i in range(0,len(asa_residuals)-1):
# 	asa_conv_factors[i] = asa_residuals[i]/asa_residuals[i-1]

# print "Classical aSA - ", asa_time, " seconds"
# print asa_conv_factors

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# New aSA solver
# --------------

# start = time.clock()
# [ml_new_asa, work] = asa_solver(A, B=bad_guy,
# 			                    max_targets=max_targets,
# 	                            min_targets=min_targets,
# 	                            num_targets=num_candidates,
# 	                            targets_iters=candidate_iters, conv_tol=tol,
# 	                            weak_tol=weak_tol, local_weak_tol=local_weak_tol,
# 	                            max_coarse=max_coarse, max_levels=max_levels,
# 	                            max_level_iterations=max_level_iterations,
# 	                            prepostsmoother=relaxation,
# 	                            smooth=interp_smooth, strength=strength_connection, aggregate=aggregation,
# 	                            coarse_solver=coarse_solver, coarse_size=coarse_size)

# new_asa_sol = ml_new_asa.solve(b, x0, tol, residuals=new_asa_residuals)

# end = time.clock()
# new_asa_time = end-start
# new_asa_conv_factors = np.zeros((len(new_asa_residuals)-1,1))
# for i in range(0,len(new_asa_residuals)-1):
# 	new_asa_conv_factors[i] = new_asa_residuals[i]/new_asa_residuals[i-1]

# print "New  aSA - ", new_asa_time, " seconds"
# print new_asa_conv_factors


# pdb.set_trace()

