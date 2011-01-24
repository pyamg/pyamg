"""
Simple example of one dimensional diffusion that makes use of 
the included one dimensional visualization tools for a 
stand-along SA solver.

Usage
-----
$ python oneD_diffusion.py  npts

"""
import numpy
import scipy
from pyamg import smoothed_aggregation_solver
from pyamg.gallery import poisson
from oneD_tools import *
import pylab
import sys

if(__name__=="__main__"):

    # Ensure repeatability of tests
    numpy.random.seed(625)

    # Generate system and solver
    if len(sys.argv)<2:
        n=10
    else:
        n = int(sys.argv[1])

    # setup 1D Poisson problem
    A = poisson((n,), format='csr')
    ml=smoothed_aggregation_solver(A, max_coarse=5, coarse_solver='pinv2',keep=True)
        
    # Profile this solver for 5 iterations
    oneD_profile(ml, grid=scipy.linspace(0,1,n), x0=scipy.rand(n,), \
                 b=numpy.zeros((n,)), iter=10)
    
    # Plot the fine level's aggregates
    oneD_coarse_grid_vis(ml, fig_num=20, level=0)
    
    if True:
        # Only plot the basis functions in P if n is small, e.g. 20
        oneD_P_vis(ml, fig_num=30, level=0, interp=False)

    pylab.show()
