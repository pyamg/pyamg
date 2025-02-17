"""Generate matrix and right-hand side for upwind FD discretization of 2D advection."""

import numpy as np
from .stencil import stencil_grid


def advection_2d(grid, theta=np.pi/4.0, l_bdry=1.0, b_bdry=1.0):
    """Generate matrix and RHS for upwind FD discretization of 2D advection.

    The 2D advection equation

        (cos(theta),sin(theta)) dot grad(u) = 0,

    with inflow boundaries on the left and bottom of the domain. Assume uniform
    grid spacing, dx=dy, even for grid[0] != grid[1].

    Parameters
    ----------
    grid : tuple
        Number of points in y and x, ``(ny, nx)``, note the ordering.
    theta : float, optional
        Rotation angle `theta` in radians defines direction of advection
        (cos(theta),sin(theta)).
    l_bdry : float, array
        Left boundary value. If float, then constant in-flow boundary value
        applied. If array, then length of array must be equal to ``ny=grid[0]``,
        and this array defines non-constant boundary value on the left.
    b_bdry : float, array
        Bottom boundary value. If float, then constant in-flow boundary value
        applied. If array, then length of array must be equal to ``nx=grid[1]``,
        and this array defines non-constant boundary value on the bottom.

    Returns
    -------
    csr_array
        Defines 2D FD upwind discretization, with boundary.
    array
        Defines right-hand-side with boundary contributions.

    See Also
    --------
    poisson

    Examples
    --------
    >>> from numpy import pi
    >>> from pyamg.gallery import advection_2d
    >>> A, rhs = advection_2d( (4,4), theta=pi/4)
    >>> print(A.toarray().round(4))
    [[ 1.4142  0.      0.     -0.7071  0.      0.      0.      0.      0.    ]
     [-0.7071  1.4142  0.      0.     -0.7071  0.      0.      0.      0.    ]
     [ 0.     -0.7071  1.4142  0.      0.     -0.7071  0.      0.      0.    ]
     [ 0.      0.      0.      1.4142  0.      0.     -0.7071  0.      0.    ]
     [ 0.      0.      0.     -0.7071  1.4142  0.      0.     -0.7071  0.    ]
     [ 0.      0.      0.      0.     -0.7071  1.4142  0.      0.     -0.7071]
     [ 0.      0.      0.      0.      0.      0.      1.4142  0.      0.    ]
     [ 0.      0.      0.      0.      0.      0.     -0.7071  1.4142  0.    ]
     [ 0.      0.      0.      0.      0.      0.      0.     -0.7071  1.4142]]

    """
    grid = tuple(grid)
    if len(grid) != 2:
        raise ValueError('grid must be a length 2 tuple, '
                         'describe number of points in x and y')
    if theta <= 0 or theta >= np.pi/2:
        raise ValueError('theta must be in (0, pi/2)')

    # First-order upwind FD for dx and dy in (cos(theta),sin(theta)) \nabla u.
    w1 = np.cos(theta)
    w2 = np.sin(theta)
    st = np.array([[0, 0, 0], [-w1, w1+w2, 0], [0, -w2, 0]])
    A = stencil_grid(st, grid).tocsr()

    # Assume left and bottom of domain to be in-flow boundary
    # From
    #   grid=(ny,nx)
    #   np.arange(np.prod(grid)).reshape((grid))
    # We get boundary DOFs
    l_bdofs = np.array([i*grid[1] for i in range(0, grid[0])])
    b_bdofs = np.array([grid[1]*(grid[0]-1)+i for i in range(0, grid[1])])
    all_bdofs = np.concatenate((l_bdofs, b_bdofs))
    int_dofs = [i for i in range(0, A.shape[0]) if i not in all_bdofs]

    # Convert boundary values to array
    if np.isscalar(l_bdry):
        l_bdry = np.full(grid[0], l_bdry)
    elif l_bdry.shape[0] != grid[0]:
        raise ValueError('left boundary data does not match boundary size')

    if np.isscalar(b_bdry):
        b_bdry = np.full(grid[0], b_bdry)
    elif b_bdry.shape[0] != grid[1]:
        raise ValueError('bottom boundary data does not match boundary size')

    # Eliminate boundary DOFs
    bdry = np.concatenate((l_bdry.flatten(), b_bdry.flatten()))
    rhs = -A[int_dofs, :][:, all_bdofs] @ bdry
    A = A[int_dofs, :][:, int_dofs]

    return A, rhs
