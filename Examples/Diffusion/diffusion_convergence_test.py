"""
Test the convergence of a diffusion equation
"""
from numpy import array, random, mean, prod
from scipy import rand, log10, pi
from pyamg.gallery import stencil_grid
from pyamg.strength import classical_strength_of_connection
from pyamg.classical.classical import classical_strength_of_connection,\
        ruge_stuben_solver
from convergence_tools import print_cycle_history
from diffusion_stencil import diffusion_stencil

if __name__ == '__main__':
    n = 100
    nx = n
    ny = n

    # Rotated Anisotropic Diffusion
    stencil = diffusion_stencil('FE', eps=0.001, beta=pi/3)

    A = stencil_grid(stencil, (nx,ny), format='csr')
    S = classical_strength_of_connection(A, 0.0)

    random.seed(625)
    x = rand(A.shape[0])
    b = A*rand(A.shape[0])

    ml = ruge_stuben_solver(A, max_coarse=10)

    resvec = []
    x = ml.solve(b, x0=x, maxiter=20, tol=1e-14, residuals=resvec)

    print_cycle_history(resvec, ml, verbose=True, plotting=True)
