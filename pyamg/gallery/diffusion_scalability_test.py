"""
Test the scalability of AMG for the diffusion equation
"""
from numpy import array, random, mean, prod, zeros
from scipy import rand, log10, cos, sin, pi
from pyamg.gallery import stencil_grid
from pyamg.strength import classical_strength_of_connection
from pyamg.classical.classical import classical_strength_of_connection, ruge_stuben_solver
from convergence_tools import print_scalability
from diffusion_stencil import diffusion_stencil

if(__name__=="__main__"):
    nlist = [100,200,300,400,500,600]

    factors=zeros((len(nlist),1)).ravel()
    complexity=zeros((len(nlist),1)).ravel()
    nnz=zeros((len(nlist),1)).ravel()
    sizelist=zeros((len(nlist),1)).ravel()
    run=0

    for n in nlist:
        nx = n
        ny = n
        print "n = %-10d of %-10d"%(n,nlist[-1])

        # Rotated Anisotropic Diffusion
        stencil = diffusion_stencil('FE',eps=0.001,beta=pi/3)

        A = stencil_grid(stencil, (nx,ny), format='csr')

        S = classical_strength_of_connection(A, 0.0)

        random.seed(625)
        x = rand(A.shape[0])
        b = A*rand(A.shape[0])

        ml = ruge_stuben_solver(A, max_coarse=10)

        x, resvec = ml.solve(b, x0=x, maxiter=200, tol=1e-8, return_residuals=True)
        factors[run] = (resvec[-1]/resvec[0])**(1.0/len(resvec))
        complexity[run] = ml.operator_complexity()
        nnz[run] = A.nnz
        sizelist[run]=A.shape[0]
        run +=1

    print_scalability(factors, complexity, nnz, sizelist, plotting=True)
