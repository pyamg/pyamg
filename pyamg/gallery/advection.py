import numpy as np
from .stencil import stencil_grid


def advection_2d(grid, theta=np.pi/4.0, l_bdry=1, b_bdry=1):
    """
    Generate matrix and right-hand side for upwind FD
    discretization of 2d advection:
        (cos(theta),sin(theta)) cdot nabla u = 0,
    with inflow boundaries on the left and bottom of
    the domain. Assume uniform grid spacing, dx=dy,
    even for grid[0] != grid[1].

    """
    grid = tuple(grid)
    if theta <= 0 or theta >= np.pi/2:
        raise ValueError("theta must be in (0,pi/2)")

    # First-order upwind FD for dx and dy in (cos(theta),sin(theta)) \nabla u.
    w1 = np.cos(theta)
    w2 = np.sin(theta)
    st = np.array([[0,0,0],[-w1,w1+w2,0],[0,-w2,0]])
    A = stencil_grid(st, grid).tocsr()

    # Assume left and bottom of domain to be in flow boundary
    # From 
    #   grid=(ny,nx)
    #   np.arange(np.prod(grid)).reshape((grid))
    # We get boundary DOFs
    l_bdofs = np.array([i*grid[1] for i in range(0,grid[0])])
    b_bdofs = np.array([grid[1]*(grid[0]-1)+i for i in range(0,grid[1])])
    all_bdofs = np.concatenate((l_bdofs,b_bdofs))
    int_dofs = [i for i in range(0,A.shape[0]) if i not in all_bdofs]

    # Convert boundary values to array
    if not hasattr(l_bdry, "__len__"):
        l_bdry = l_bdry*np.ones((grid[0],))
    elif l_bdry.shape[0] != grid[0]:
        raise ValueError("left boundary data does not match boundary size")

    if not hasattr(b_bdry, "__len__"):
        b_bdry = b_bdry*np.ones((grid[1],))
    elif b_bdry.shape[0] != grid[1]:
        raise ValueError("bottom boundary data does not match boundary size")

    # Eliminate boundary DOFs
    bdry = np.concatenate((l_bdry.flatten(),b_bdry.flatten()))
    rhs = -A[int_dofs,:][:,all_bdofs]*bdry
    A = A[int_dofs,:][:,int_dofs]

    return A, rhs


