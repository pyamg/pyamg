"""
Test the scalability of AMG for the anisotropic diffusion equation
"""
import numpy
import scipy

from pyamg.gallery import stencil_grid
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.strength import classical_strength_of_connection
from pyamg.classical.classical import classical_strength_of_connection, ruge_stuben_solver

from convergence_tools import print_scalability

if(__name__=="__main__"):
    nlist = [100,200,300,400,500,600]

    factors=numpy.zeros((len(nlist),1)).ravel()
    complexity=numpy.zeros((len(nlist),1)).ravel()
    nnz=numpy.zeros((len(nlist),1)).ravel()
    sizelist=numpy.zeros((len(nlist),1)).ravel()
    run=0

    for n in nlist:
        nx = n
        ny = n
        print "n = %-10d of %-10d"%(n,nlist[-1])

        # Rotated Anisotropic Diffusion
        stencil = diffusion_stencil_2d(type='FE',epsilon=0.001,theta=scipy.pi/3)

        A = stencil_grid(stencil, (nx,ny), format='csr')

        S = classical_strength_of_connection(A, 0.0)

        numpy.random.seed(625)
        x = scipy.rand(A.shape[0])
        b = A*scipy.rand(A.shape[0])

        ml = ruge_stuben_solver(A, max_coarse=10)

        resvec = []
        x = ml.solve(b, x0=x, maxiter=200, tol=1e-8, residuals=resvec)
        factors[run] = (resvec[-1]/resvec[0])**(1.0/len(resvec))
        complexity[run] = ml.operator_complexity()
        nnz[run] = A.nnz
        sizelist[run]=A.shape[0]
        run +=1

    print_scalability(factors, complexity, nnz, sizelist, plotting=True)
