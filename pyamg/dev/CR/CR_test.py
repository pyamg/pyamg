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
from pyamg.classical import CR


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


# Optional input, figure and subplot indices [num_plots, i, j]
def plot_poisson_matching(splitting, N, fig=None, subplot=None, label=None):

    # Plot nodes in grid
    if fig == None:
        fig, ax = plt.subplots()
    else:
        ax = fig.add_subplot(subplot[0],subplot[1],subplot[2])

    Cpts = np.where(splitting == 1)
    Fpts = np.where(splitting == 0)

    # Note, this could be more efficient w/o constructing graph... 
    num_unknowns = N*N
    node_locations = np.zeros((num_unknowns,2))
    for i in range(0,num_unknowns):
        node_locations[i,:] = ind_to_coordinates(i=i, Nx=N, Ny=N)

    ax.scatter(node_locations[Fpts,0],node_locations[Fpts,1], color='black', s=15)
    ax.scatter(node_locations[Cpts,0],node_locations[Cpts,1], color='darkred', s=50)

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_title(label)


def plot_multiple_poisson(splittings, labels, N):

    num_plots = len(splittings)
    if num_plots == 1:
        plot_poisson_matching(splittings[0], N)
    elif num_plots == 2:
        fig = plt.figure()
        plot_poisson_matching(splitting=splittings[0], N=N, fig=fig, subplot=[1,2,1], label=labels[0])
        plot_poisson_matching(splitting=splittings[1], N=N, fig=fig, subplot=[1,2,2], label=labels[1])
    elif num_plots == 3:
        fig = plt.figure()
        plot_poisson_matching(splitting=splittings[0], N=N, fig=fig, subplot=[1,3,1], label=labels[0])
        plot_poisson_matching(splitting=splittings[1], N=N, fig=fig, subplot=[1,3,2], label=labels[1])
        plot_poisson_matching(splitting=splittings[2], N=N, fig=fig, subplot=[1,3,3], label=labels[2])
    elif num_plots == 4:
        fig = plt.figure()
        plot_poisson_matching(splitting=splittings[0], N=N, fig=fig, subplot=[2,2,1], label=labels[0])
        plot_poisson_matching(splitting=splittings[1], N=N, fig=fig, subplot=[2,2,2], label=labels[1])
        plot_poisson_matching(splitting=splittings[2], N=N, fig=fig, subplot=[2,2,3], label=labels[2])
        plot_poisson_matching(splitting=splittings[3], N=N, fig=fig, subplot=[2,2,4], label=labels[3])
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
epsilon 	= 1.0
theta 		= 3.0*np.pi/4

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

# ------------------------------------------------------------------------------#

thetacs = [0.3,0.4]
thetacr = 0.5
nu = 3
maxiter = 20

h_split = CR(A, method='habituated', nu=nu, thetacr=thetacr, thetacs=thetacs, maxiter=maxiter)
c_split = CR(A, method='concurrent', nu=nu, thetacr=thetacr, thetacs=thetacs, maxiter=maxiter)


h_split_auto = CR(A, method='habituated', nu=nu, thetacr=thetacr, thetacs='auto', maxiter=maxiter)
c_split_auto = CR(A, method='concurrent', nu=nu, thetacr=thetacr, thetacs='auto', maxiter=maxiter)


# ------------------------------------------------------------------------------#

labels = ['Habituated','Concurrent', 'Habituated auto','Concurrent auto']
splittings = [h_split,c_split,h_split_auto,c_split_auto]

plot_multiple_poisson(splittings,labels,N)





