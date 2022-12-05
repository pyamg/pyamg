import numpy as np
from copy import deepcopy
import pdb, cProfile, pstats
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.sparse import csr_matrix, lil_matrix, isspmatrix_csr, isspmatrix_bsr
from scipy.io import mmwrite

from pyamg.gallery import poisson, linear_elasticity
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery.stencil import stencil_grid
from pyamg.util.utils import symmetric_rescaling
from pyamg.aggregation.aggregation import smoothed_aggregation_solver
from adaptive_pairwise import adaptive_pairwise_solver

# pyAMG diffusion stencil is rotated from what it looks like, i.e. each row of 3
# elements corresponds to a fixed x-value, and 3 y-values. This means that 
# the nodes are organized something like 
#   ind = iy + ix*Nx
# N gives interior DOF, so hx = 1/(Nx+1), hy = 1/(Ny+1). The inverse transform
# should look like
#   x = (floor(ind/Ny) + 1)hx
#   y = (rem(ind/Ny) + 1)hy
def ind_to_coordinates(i, Nx, Ny):

    ix = (1.0 + np.floor(i/Ny)) * 1.0/(Nx+1)
    iy = (1.0 + np.remainder(i,Ny)) * 1.0/(Ny+1)
    return [ix,iy] 

# Weight defined as follows:
#   - Get unit vector between i,j 
#   - Unit vector of direction theta given by [acos(theta), asin(theta)] 
#   - Take absolute value of dot product between two vectors = |cos(phi)|,
#     where phi is angle between two vectors. Maximum at parallel, 
#     minimum at orthogonal. 
# Takes in 
#   - Nodes - i,j
#   - Number of unknowns in x and y direction - Nx,Ny
#   - Angle of anisotropy - theta 
#   - power to raise weight to - 2
def get_weight(i, j, Nx, Ny, theta, dim=2, power=2):

    if dim > 2:
        raise ValueError("Only implemented for 2-dimensions.")

    if i == j:
        return 0.0
    else:
        [ix,iy] = ind_to_coordinates(i, Nx, Ny)
        [jx,jy] = ind_to_coordinates(j, Nx, Ny)
        if ix == jx:
            phi = 0.0
        else:
            phi = np.arctan( float(jy-iy)/(jx-ix) )
        
        w = np.abs( np.cos(phi)*np.cos(theta) + np.sin(phi)*np.sin(theta) )
        return w**power


def get_geometric_weights(A, theta, Nx, Ny):

    n = A.shape[0]
    if isspmatrix_csr(A):
        W = A.copy()
        # Get angle between nodes for (i,j)th entry in A
        for i in range(0,n):
            lower_ind = W.indptr[i]
            upper_ind = W.indptr[i+1]
            for j_ind in range(lower_ind,upper_ind):
                j = W.indices[j_ind]
                W.data[j_ind] = get_weight(i,j,Nx,Ny,theta)

    elif isspmatrix_bsr(A): 
        raise ValueError("Not implemented for BSR matrices yet.")

    return W


# Get geometric weights for angle between nodes in a neighborhood of degree 2. 
def get_geometric_weights_d2(A, theta, Nx, Ny):

    n = A.shape[0]
    # Get maximum possible number of paths of length 2 in A
    paths_d1 = np.max( A.indptr[1:(n+1)] - A.indptr[0:n] )
    max_paths = n*paths_d1*paths_d1
    # Preallocate empty sparse array
    rowptr = np.zeros((n+1,))
    data = np.zeros((max_paths,))
    colinds = np.zeros((max_paths,))
    next_ind = 0

    if isspmatrix_csr(A):
        # Loop over each node in A
        for i in range(0,n):
            lower_ind1 = A.indptr[i]
            upper_ind1 = A.indptr[i+1]
            neighbors = []

            # Loop over d1 and d2 neighbors for ith node, save neighbors in set.
            for d1_ind in range(lower_ind1,upper_ind1):
                d1_node = A.indices[d1_ind]
                lower_ind2 = A.indptr[d1_node]
                upper_ind2 = A.indptr[d1_node+1]
                for d2_ind in range(lower_ind2,upper_ind2):
                    d2_node = A.indices[d2_ind]
                    neighbors.append(d2_node)

            # Find unique set of neighbors, get weight w(i,j)
            neighbors = np.unique(neighbors)
            for node in neighbors:
                colinds[next_ind] = node
                data[next_ind] = get_weight(i,node,Nx,Ny,theta)
                next_ind += 1

            # Set row-pointer for ith row
            rowptr[i+1] = next_ind

    elif isspmatrix_bsr(A): 
        raise ValueError("Not implemented for BSR matrices yet.")

    W = csr_matrix((data[0:next_ind],colinds[0:next_ind],rowptr),shape=A.shape)
    return W


# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

N 			= 200
problem_dim = 2
epsilon 	= 0.01
theta 		= 3.0*np.pi/16
zero_rhs    = True
rand_guess  = True

# 1d Poisson 
if problem_dim == 1:
	grid_dims = [N,1]
	A = poisson((N,), format='csr')
# 2d Poisson
elif problem_dim == 2:
	grid_dims = [N,N]
	stencil = diffusion_stencil_2d(epsilon,theta)
	A = stencil_grid(stencil, grid_dims, format='csr')

# Scale out diagonal
[d,d,A]   = symmetric_rescaling(A)
vec_size  = A.shape[0]
# W = get_geometric_weights(A, theta, N, N)
# W.eliminate_zeros()

# Zero right hand side or sin(pi x)
if zero_rhs:
    b = np.zeros((vec_size,1))
    # If zero rhs and zero initial guess, throw error
    if not rand_guess:
        print "Zero rhs and zero initial guess converges trivially."
# Note, this vector probably doesn't make sense in 2d... 
else: 
    b = np.sin(math.pi*np.arange(0,vec_size)/(vec_size-1.0))
    # b = np.array([np.sin(math.pi*np.arange(0,vec_size)/(vec_size-1.0)),np.ones(vec_size)]).T

# Random vs. zero initial guess
if rand_guess:
    x0 = np.random.rand(vec_size,1)
else:
    x0 = np.zeros(vec_size,1)

asa_residuals = []

# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

# General multilevel parameters
# -----------------------------
max_levels         = 20         # Max levels in hierarchy
max_coarse         = 30         # Max points allowed on coarse grid
tol                = 1e-8       # Residual convergence tolerance
is_pdef            = True       # Assume matrix positive definite (only for aSA)
keep_levels        = False      # Also store SOC, aggregation, and tentative P operators
diagonal_dominance = False      # Avoid coarsening diagonally dominant rows 
coarse_solver = 'pinv'
accel = None

aggregate = ('pairwise', {'matchings': 1, 'algorithm': 'drake', 'initial_target' : 'ones'})
# interp_smooth = ('jacobi', {'omega': 4.0/3.0, 'degree':1 } )
interp_smooth = None
relaxation = ('gauss_seidel', {'sweep': 'forward', 'iterations': 1} )
# relaxation = ('jacobi', {'omega': 4.0/3.0} )
improve_candidates = [('gauss_seidel', {'sweep': 'forward', 'iterations': 5})]
# improve_candidates = None
test_iterations = 15
desired_convergence = 0.7
additive = False
reconstruct = False
use_ritz = False

# ----------------------------------------------------------------------------- #
# ------------------------------------------initial_targets----------------------------------- #

ml_asa = adaptive_pairwise_solver(A, B = None, symmetry = 'symmetric',
                                  desired_convergence = desired_convergence,
                                  test_iterations = test_iterations, 
                                  test_accel=accel,
                                  strength = None,
                                  smooth = interp_smooth,
                                  aggregate = aggregate,
                                  presmoother = relaxation,
                                  postsmoother = relaxation,
                                  max_levels = max_levels, max_coarse = max_coarse,
                                  coarse_solver=coarse_solver, additive=additive,
                                  reconstruct=reconstruct, use_ritz=use_ritz,
                                  improve_candidates=improve_candidates)
grid = ml_asa.operator_complexity()
cycle = ml_asa.cycle_complexity()

sol = ml_asa.solve(b, x0, tol=tol, residuals=asa_residuals, accel=accel, additive=additive)
asa_conv_factors = np.zeros((len(asa_residuals)-1,1))
for i in range(0,len(asa_residuals)-1):
  asa_conv_factors[i] = asa_residuals[i]/asa_residuals[i-1]

print "Adaptive SA/AMG - ", np.mean(asa_conv_factors[1:])
print " operator complexity - ", grid
print " cycle complexity - ", cycle, "\n"

# pdb.set_trace()



