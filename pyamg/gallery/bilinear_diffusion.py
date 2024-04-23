"""Generate a coo-matrix.

Supports isotropic diffusion (FE), anisotropic diffusion (FE), and
rotated anisotropic diffusion (FE).
"""

import numpy as np
from scipy import sparse


def bilinear_fem(grid, kappa=None, c=None):
    """Bilinear finite element discretization of
        - div . kappa(x,y) grad u + c(x, y) u.

    Parameters
    ----------
    grid : tuple of integers
        grid dimensions in 2D, e.g. (100,100)
    kappa : function
        diffusion coefficient, kappa(x,y) with vector input
        returns a 2x2 matrix that transforms <grad u>
    c : function
        reaction term c(x,y)

    Returns
    -------
    matrix : scipy coo_matrix
        A nxn matrix, where n=grid[0]*grid[1]
    """
    if kappa is None:
        def kappa(x, y):
            return np.array([[1.0, 0.0], [0.0, 1.0]])

    if c is None:
        def c(x, y):
            return 1.0

    no_mesh_cols = grid[0]
    no_mesh_rows = grid[1]
    hx = 1/(no_mesh_cols-1)
    hy = 1/(no_mesh_rows-1)
    hxy = hx/hy
    hyx = hy/hx
    hxhy = hx*hy
    tot_dof = no_mesh_rows * no_mesh_cols
    tot_funcs = (no_mesh_rows-1) * (no_mesh_cols-1)
    data = np.zeros((16*tot_funcs,), )
    row = np.zeros((16*tot_funcs,), dtype=int)
    col = np.zeros((16*tot_funcs,), dtype=int)

    count_data = 0

    for j in range(no_mesh_cols-1):
        for i in range(no_mesh_rows-1):
            x_pos = j*hx + hx/2
            y_pos = i*hy + hy/2
            k = kappa(x_pos, y_pos)
            k11 = k[0, 0]
            k12 = k[0, 1]
            k22 = k[1, 1]
            val = hxhy*c(x_pos, y_pos)

            rr = j * no_mesh_rows + i
            rc_nos = np.array([rr, rr+no_mesh_rows, rr+1, rr+no_mesh_rows+1])
            row[count_data:count_data+16] = \
                np.kron(rc_nos, np.ones((4,), dtype=int))
            col[count_data:count_data+16] = \
                np.kron(np.ones((4,), dtype=int), rc_nos)
            x1 = k11*hyx/3 + k22*hxy/3 + k12/2 + val/9
            x2 = -k11*hyx/3 + k22*hxy/6 + val/18
            x3 = k11*hyx/6 - k22*hxy/3 + val/18
            x4 = -k11*hyx/6 - k22*hxy/6 - k12/2 + val/36
            x5 = k11*hyx/3 + k22*hxy/3 - k12/2 + val/9
            x6 = -k11*hyx/6 - k22*hxy/6 + k12/2 + val/36
            x7 = k11*hyx/6 - k22*hxy/3 + val/18
            x8 = k11*hyx/3 + k22*hxy/3 - k12/2 + val/9
            x9 = -k11*hyx/3 + k22*hxy/6 + val/18
            x10 = k11*hyx/3 + k22*hxy/3 + k12/2 + val/9
            data[count_data:count_data+16] = np.array([x1, x2, x3, x4,
                                                       x2, x5, x6, x7,
                                                       x3, x6, x8, x9,
                                                       x4, x7, x9, x10])
            count_data = count_data+16
    A = sparse.coo_matrix((data, (row, col)), shape=(tot_dof, tot_dof))
    A.sum_duplicates()
    return A
