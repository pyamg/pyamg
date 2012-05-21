"""
Test the scalability of SA for rotated diffusion while 
highlighting the performance of different strength measures.
Try different values for classic_theta and evolution_theta.
"""
import numpy
import scipy

from pyamg import smoothed_aggregation_solver, rootnode_solver
from pyamg.gallery import stencil_grid
from pyamg.gallery.diffusion import diffusion_stencil_2d

from convergence_tools import print_scalability

if(__name__=="__main__"):

    # Ensure repeatability of tests
    numpy.random.seed(625)
    
    # Grid sizes to test
    #nlist = [40,70,100,130,160]
    nlist = [100,200,300,400]

    factors_classic    = numpy.zeros((len(nlist),1)).ravel()
    complexity_classic = numpy.zeros((len(nlist),1)).ravel()
    nnz_classic        = numpy.zeros((len(nlist),1)).ravel()
    sizelist_classic   = numpy.zeros((len(nlist),1)).ravel()
        
    factors_ode    = numpy.zeros((len(nlist),1)).ravel()
    complexity_ode = numpy.zeros((len(nlist),1)).ravel()
    nnz_ode        = numpy.zeros((len(nlist),1)).ravel()
    sizelist_ode   = numpy.zeros((len(nlist),1)).ravel()
    
    factors_ode_root    = numpy.zeros((len(nlist),1)).ravel()
    complexity_ode_root = numpy.zeros((len(nlist),1)).ravel()
    nnz_ode_root        = numpy.zeros((len(nlist),1)).ravel()
    sizelist_ode_root   = numpy.zeros((len(nlist),1)).ravel()

    run=0

    # Smoothed Aggregation Parameters
    theta = scipy.pi/8.0                                # Angle of rotation
    epsilon = 0.001                                     # Anisotropic coefficient
    mcoarse = 10                                        # Max coarse grid size
    prepost = ('gauss_seidel',                          # pre/post smoother
               {'sweep':'symmetric', 'iterations':1})   
    smooth = ('energy', {'maxiter' : 9, 'degree':3})    # Prolongation Smoother
    classic_theta = 0.0                                 # Classic Strength Measure
                                                        #    Drop Tolerance
    evolution_theta = 4.0                                     # evolution Strength Measure
                                                        #    Drop Tolerance

    for n in nlist:
        nx = n
        ny = n
        print "Running Grid = (%d x %d)" % (nx, ny)

        # Rotated Anisotropic Diffusion Operator
        stencil = diffusion_stencil_2d(type='FE',epsilon=epsilon,theta=theta)
        A = stencil_grid(stencil, (nx,ny), format='csr')

        # Random initial guess, zero RHS
        x0 = scipy.rand(A.shape[0])
        b = numpy.zeros((A.shape[0],))

        # Classic SA strength measure
        ml = smoothed_aggregation_solver(A, max_coarse=mcoarse, coarse_solver='pinv2', 
                                         presmoother=prepost, postsmoother=prepost, smooth=smooth,
                                         strength=('symmetric', {'theta' : classic_theta}) )
        resvec = []
        x = ml.solve(b, x0=x0, maxiter=100, tol=1e-8, residuals=resvec)
        factors_classic[run]    = (resvec[-1]/resvec[0])**(1.0/len(resvec))
        complexity_classic[run] = ml.operator_complexity()
        nnz_classic[run]        = A.nnz
        sizelist_classic[run]   = A.shape[0]

        # Evolution strength measure
        ml = smoothed_aggregation_solver(A, max_coarse=mcoarse, coarse_solver='pinv2', 
                                         presmoother=prepost, postsmoother=prepost, smooth=smooth,
                                         strength=('evolution', {'epsilon' : evolution_theta, 'k' : 2}) )
        resvec = []
        x = ml.solve(b, x0=x0, maxiter=100, tol=1e-8, residuals=resvec)
        factors_ode[run]    = (resvec[-1]/resvec[0])**(1.0/len(resvec))
        complexity_ode[run] = ml.operator_complexity()
        nnz_ode[run]        = A.nnz
        sizelist_ode[run]   = A.shape[0]

        # Evolution strength measure
        ml = rootnode_solver(A, max_coarse=mcoarse, coarse_solver='pinv2', 
                                presmoother=prepost, postsmoother=prepost, smooth=smooth,
                                strength=('evolution', {'epsilon' : evolution_theta, 'k' : 2}) )
        resvec = []
        x = ml.solve(b, x0=x0, maxiter=100, tol=1e-8, residuals=resvec)
        factors_ode_root[run]    = (resvec[-1]/resvec[0])**(1.0/len(resvec))
        complexity_ode_root[run] = ml.operator_complexity()
        nnz_ode_root[run]        = A.nnz
        sizelist_ode_root[run]   = A.shape[0]


        run +=1

    # Print Problem Description
    print "\nAMG Scalability Study for Ax = 0, x_init = rand\n"
    print "Emphasis on Robustness of Evolution Strength "
    print "Measure and Root-Node Solver\n"
    print "Rotated Anisotropic Diffusion in 2D"
    print "Anisotropic Coefficient = %1.3e" % epsilon
    print "Rotation Angle = %1.3f" % theta

    # Print Tables
    print_scalability(factors_classic, complexity_classic, 
         nnz_classic, sizelist_classic, plotting=False, 
         title='Classic SA\nClassic Strength Measure DropTol = %1.2f' % classic_theta)
    print_scalability(factors_ode, complexity_ode, nnz_ode, sizelist_ode, 
         plotting=False, title='Classic SA\nEvolution Strength Measure DropTol = %1.2f' % evolution_theta)
    print_scalability(factors_ode_root, complexity_ode_root, nnz_ode_root, sizelist_ode_root, 
         plotting=False, title='Root-Node Solver\nEvolution Strength Measure DropTol = %1.2f' % evolution_theta)


