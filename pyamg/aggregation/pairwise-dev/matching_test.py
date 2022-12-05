import numpy as np
import time
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
from pyamg.strength import classical_strength_of_connection,\
    symmetric_strength_of_connection, evolution_strength_of_connection,\
    energy_based_strength_of_connection, distance_strength_of_connection,\
    algebraic_distance
from pyamg.aggregation.aggregate import standard_aggregation, naive_aggregation,\
    lloyd_aggregation, pairwise_aggregation



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


# Optional input, figure and subplot indices [num_plots, i, j]
def plot_poisson_matching(AggOp, N, Cpts=None, fig=None, subplot=None, label=None):

    # AggOplot nodes in grid
    if fig == None:
        fig, ax = plt.subplots()
    else:
        ax = fig.add_subplot(subplot[0],subplot[1],subplot[2])

    # Note, this could be more efficient w/o constructing graph... 
    num_unknowns = N*N
    node_locations = np.zeros((num_unknowns,2))
    for i in range(0,num_unknowns):
        node_locations[i,:] = ind_to_coordinates(i=i, Nx=N, Ny=N)

    ax.scatter(node_locations[:,0],node_locations[:,1], color='darkred')
    if Cpts is not None:
        ax.scatter(node_locations[Cpts,0],node_locations[Cpts,1], color='black')

    # Plot pairwise aggregates
    AggOp = AggOp.tocsc()
    for agg in range(0,len(AggOp.indptr)-1):
        agg_size = AggOp.indptr[agg+1] - AggOp.indptr[agg]
        this_agg = np.zeros((agg_size, 2))
        first = AggOp.indptr[agg]
        for ind in range(0, agg_size):
            this_agg[ind,:] = node_locations[(AggOp.indices[first+ind]),:]

        this_agg = this_agg[this_agg[:,0].argsort(),:]
        this_agg = this_agg[this_agg[:,1].argsort(),:]
        if agg_size == 2:
            ax.plot([this_agg[0,0],this_agg[1,0]],[this_agg[0,1],this_agg[1,1]],color='blue')
        elif agg_size == 3:
            shape = Polygon(this_agg, fill=True, color='blue', zorder=0)
            ax.add_patch(shape)  
        elif agg_size == 4:
            shape = Polygon(this_agg, fill=True, color='blue', zorder=0)
            ax.add_patch(shape)            
            inds = [0,2,1,3]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)  
        elif agg_size == 5:
            shape = Polygon(this_agg, fill=True, color='blue', zorder=0)
            ax.add_patch(shape)    
            inds = [0,2,4,1,3]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)  
        elif agg_size == 6:
            shape = Polygon(this_agg, fill=True, color='blue', zorder=0)
            ax.add_patch(shape)    
            inds = [0,2,4,1,3,5]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)  
            inds = [0,4,1,5,2,3]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)  
        elif agg_size == 7:
            shape = Polygon(this_agg, fill=True, color='blue', zorder=0)
            ax.add_patch(shape)    
            inds = [0,2,4,6,1,3,5]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,4,1,5,2,6,3]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
        elif agg_size == 8:
            shape = Polygon(this_agg, fill=True, color='blue', zorder=0)
            ax.add_patch(shape)    
            inds = [0,2,4,6,1,3,5,7]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [3,0,4,1,7,2,5,6]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [5,0,6,2,3,7,4,1]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
        elif agg_size == 9:
            shape = Polygon(this_agg, fill=True, color='blue', zorder=0)
            ax.add_patch(shape)    
            inds = [0,2,4,6,8,1,3,5,7]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,3,6,2,8,5,1,7,4]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,5,2,7,3,8,4,1,6]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
        elif agg_size == 10:
            shape = Polygon(this_agg, fill=True, color='blue', zorder=0)
            ax.add_patch(shape)    
            inds = [0,2,4,6,8,1,3,5,7,9]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,3,6,9,1,4,7,2,5,8]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,4,8,2,6,1,5,9,3,7]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,5,4,9,2,7,3,8,1,6]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
        elif agg_size == 11:
            shape = Polygon(this_agg, fill=True, color='blue', zorder=0)
            ax.add_patch(shape)    
            inds = [0,2,4,6,8,10,1,3,5,7,9]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,3,6,9,1,4,7,10,2,5,8]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,4,8,1,5,9,2,6,10,3,7]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,5,10,4,9,3,8,2,7,1,6]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
        elif agg_size == 12:
            shape = Polygon(this_agg, fill=True, color='blue', zorder=0)
            ax.add_patch(shape)    
            inds = [0,2,4,6,8,10,1,3,5,7,9,11]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,3,6,9,5,8,11,1,4,7,2,10]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,4,8,3,7,11,2,6,10,1,5,9]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,5,10,3,8,1,6,11,2,9,4,7]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)
            inds = [0,6,5,11,4,10,7,1,9,3,2,8]
            shape = Polygon(this_agg[inds,:], fill=True, color='blue', zorder=0)
            ax.add_patch(shape)

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_title(label)


def plot_multiple_poisson(aggregations, labels, N, Cpts=True):

    num_plots = len(aggregations)
    if Cpts == True:
        if num_plots == 1:
            plot_poisson_matching(aggregations[0], N)
        elif num_plots == 2:
            fig = plt.figure()
            plot_poisson_matching(AggOp=aggregations[0][0], N=N, Cpts=aggregations[0][1], fig=fig, subplot=[1,2,1], label=labels[0])
            plot_poisson_matching(AggOp=aggregations[1][0], N=N, Cpts=aggregations[1][1], fig=fig, subplot=[1,2,2], label=labels[1])
        elif num_plots == 3:
            fig = plt.figure()
            plot_poisson_matching(AggOp=aggregations[0][0], N=N, Cpts=aggregations[0][1], fig=fig, subplot=[1,3,1], label=labels[0])
            plot_poisson_matching(AggOp=aggregations[1][0], N=N, Cpts=aggregations[1][1], fig=fig, subplot=[1,3,2], label=labels[1])
            plot_poisson_matching(AggOp=aggregations[2][0], N=N, Cpts=aggregations[2][1], fig=fig, subplot=[1,3,3], label=labels[2])
        elif num_plots == 4:
            fig = plt.figure()
            plot_poisson_matching(AggOp=aggregations[0][0], N=N, Cpts=aggregations[0][1], fig=fig, subplot=[2,2,1], label=labels[0])
            plot_poisson_matching(AggOp=aggregations[1][0], N=N, Cpts=aggregations[1][1], fig=fig, subplot=[2,2,2], label=labels[1])
            plot_poisson_matching(AggOp=aggregations[2][0], N=N, Cpts=aggregations[2][1], fig=fig, subplot=[2,2,3], label=labels[2])
            plot_poisson_matching(AggOp=aggregations[3][0], N=N, Cpts=aggregations[3][1], fig=fig, subplot=[2,2,4], label=labels[3])
        else:
            raise ValueError('Too many plots.')
    else:
        if num_plots == 1:
            plot_poisson_matching(aggregations[0], N)
        elif num_plots == 2:
            fig = plt.figure()
            plot_poisson_matching(AggOp=aggregations[0], N=N, fig=fig, subplot=[1,2,1], label=labels[0])
            plot_poisson_matching(AggOp=aggregations[1], N=N, fig=fig, subplot=[1,2,2], label=labels[1])
        elif num_plots == 3:
            fig = plt.figure()
            plot_poisson_matching(AggOp=aggregations[0], N=N, fig=fig, subplot=[1,3,1], label=labels[0])
            plot_poisson_matching(AggOp=aggregations[1], N=N, fig=fig, subplot=[1,3,2], label=labels[1])
            plot_poisson_matching(AggOp=aggregations[2], N=N, fig=fig, subplot=[1,3,3], label=labels[2])
        elif num_plots == 4:
            fig = plt.figure()
            plot_poisson_matching(AggOp=aggregations[0], N=N, fig=fig, subplot=[2,2,1], label=labels[0])
            plot_poisson_matching(AggOp=aggregations[1], N=N, fig=fig, subplot=[2,2,2], label=labels[1])
            plot_poisson_matching(AggOp=aggregations[2], N=N, fig=fig, subplot=[2,2,3], label=labels[2])
            plot_poisson_matching(AggOp=aggregations[3], N=N, fig=fig, subplot=[2,2,4], label=labels[3])
        else:
            raise ValueError('Too many plots.')
    plt.show()


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# Potential bugs - Preis behaves a little funny, should cross reference w./ 
# Panayot / Pasqua's code. Notay doesn't work right on negative angles, maybe
# due to the whole negative weight thing?
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#

N 			= 25
problem_dim = 2
epsilon 	= 0.00
theta 		= 3.0*np.pi/12

# 1d Poisson 
if problem_dim == 1:
	grid_dims = [N,1]
	A = poisson((N,), format='csr')
# 2d Poisson
elif problem_dim == 2:
	grid_dims = [N,N]
	stencil = diffusion_stencil_2d(epsilon,theta)
	A = stencil_grid(stencil, grid_dims, format='csr')
# Elasticity (don't plot right now, plotting designed for Poisson)
elif problem_dim == -1:
    grid_dims = [N,N]
    [A,B] = linear_elasticity(grid_dims)

[d,d,A]   = symmetric_rescaling(A)
# W = get_geometric_weights(A, theta, N, N)
# W.eliminate_zeros()
n = A.shape[0]

# mmwrite('./test.mtx', A)

# pdb.set_trace()

# ------------------------------------------------------------------------------#
# C = evolution_strength_of_connection(A, epsilon=4.0, k=2)
C = symmetric_strength_of_connection(A, theta=0.1)
# C = classical_strength_of_connection(A, theta=0.2)
AggOp, Cpts = standard_aggregation(C)

# ------------------------------------------------------------------------------#
matchings = 3
B = np.ones((n,1))
improve_candidates = ('gauss_seidel', {'sweep': 'forward', 'iterations': 4})
B = None
drake_w = pairwise_aggregation(A, B=B, get_weights=True, matchings=matchings, algorithm='drake', improve_candidates=improve_candidates)
drake = pairwise_aggregation(A, B=B, get_weights=False, matchings=matchings, algorithm='drake', improve_candidates=improve_candidates)
notay = pairwise_aggregation(A, B=B, get_weights=False, matchings=matchings, algorithm=('notay', {'beta':0.25}), improve_candidates=improve_candidates)

# ------------------------------------------------------------------------------#
# labels = ['Preis pairwise (n=%i)'%(num_matchings),'Notay pairwise (e=%1.2f,n=%i)'%(0.25,num_matchings),'Drake pairwise (n=%i)'%(num_matchings)]
# matchings = [preis_matching,notay_matching,drake_matching]

# labels = ['Classical SOC/aggregation','Notay pairwise (e=%1.2f,n=%i)'%(0.25,num_matchings),'Drake pairwise (n=%i)'%(num_matchings)]
# matchings = [ [AggOp,Cpts], [notay_matching, notay_Cpts] , [drake_matching, drake_Cpts] ]

labels = ['Standard','Drake weighted', 'Drake unweighted', 'Notay pairwise']
matchings = [ AggOp, drake_w, drake, notay ]

plot_multiple_poisson(matchings,labels,N, Cpts=False)


# Plot sparsity patterns
# fig = plt.figure()
# ax1 = fig.add_subplot(1,2,1);
# ax1.spy(AggOp.todense())
# ax2 = fig.add_subplot(1,2,2);
# ax2.spy(drake.todense())
# plt.show()

pdb.set_trace()



