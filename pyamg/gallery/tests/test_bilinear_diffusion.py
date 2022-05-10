'''Test bilinear fem example.'''

import numpy as np
from numpy.testing import assert_almost_equal

# from pyamg.gallery.bilinear_diffusion import bilinear_fem
from pyamg.gallery import stencil_grid
from pyamg.gallery.diffusion import diffusion_stencil_2d

from bilinear_diffusion import bilinear_fem

cparams = 0
for kparams in [(1e-6, np.pi/3), (1e-3, np.pi/6), (1e-1, np.pi/4), (1, 0)]:
    def kappa_1quad(x, y):
        aniso_stren, aniso_dir = kparams
        C = np.cos(aniso_dir)
        S = np.sin(aniso_dir)
        Q = np.array([[C, -S],
                      [S,  C]])
        A2 = np.array([[1, 0],
                       [0, aniso_stren]])
        return Q @ A2 @ Q.T

    def ccoef_1quad(x, y):
        return cparams

    no_nodes_1d_list = [16, 17, 20, 21]
    for no_nodes_1d in no_nodes_1d_list:
        grid = (no_nodes_1d, no_nodes_1d)
        A_bfem = bilinear_fem(grid, kappa=kappa_1quad, c=ccoef_1quad)
        row_charac_vec_A = np.asarray((A_bfem != 0).sum(1).T)[0]
        col_charac_vec_A = np.asarray((A_bfem != 0).sum(0))[0]
        nzo_elem_rows = np.where(row_charac_vec_A == 9)[0]
        nzo_elem_cols = np.where(col_charac_vec_A == 9)[0]
        A_bfem = A_bfem.tocsr()
        A_bfem = A_bfem[np.ix_(nzo_elem_rows, nzo_elem_cols)]

        grid = (no_nodes_1d-2, no_nodes_1d-2)
        epsilon, theta = kparams
        sten = diffusion_stencil_2d(epsilon, theta, type='FE')
        A_pyamg = stencil_grid(sten, grid, dtype=float, format='csr')
        assert_almost_equal(A_bfem.A, A_pyamg.A)
