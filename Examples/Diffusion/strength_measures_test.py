"""
Test the scalability of SA for rotated diffusion while 
highlighting the performance of different strength measures.
Try different values for classic_theta and ode_theta.
"""
from numpy import array, random, zeros, ravel
from scipy import rand, pi
from pyamg import smoothed_aggregation_solver
from pyamg.gallery import stencil_grid
from convergence_tools import print_scalability
from diffusion_stencil import diffusion_stencil

if(__name__=="__main__"):

    # Ensure repeatability of tests
    random.seed(625)
    
    # Grid sizes to test
    nlist = [40,70,100,130,160]
    #nlist = [100,200,300,400,500,600]

    factors_classic    =zeros((len(nlist),1)).ravel()
    complexity_classic =zeros((len(nlist),1)).ravel()
    nnz_classic        =zeros((len(nlist),1)).ravel()
    sizelist_classic   =zeros((len(nlist),1)).ravel()
        
    factors_ode    =zeros((len(nlist),1)).ravel()
    complexity_ode =zeros((len(nlist),1)).ravel()
    nnz_ode        =zeros((len(nlist),1)).ravel()
    sizelist_ode   =zeros((len(nlist),1)).ravel()
    
    run=0

    # Smoothed Aggregation Parameters
    beta = pi/8.0                                       # Angle of rotation
    epsilon = 0.001                                     # Anisotropic coefficient
    mcoarse = 10                                        # Max coarse grid size
    prepost = ('gauss_seidel',                          # pre/post smoother
               {'sweep':'symmetric', 'iterations':2})   
    smooth = ('energy', {'maxiter' : 6})                # Prolongation Smoother
    classic_theta = 0.0                                 # Classic Strength Measure
                                                        #    Drop Tolerance
    ode_theta = 4.0                                     # ODE Strength Measure
                                                        #    Drop Tolerance

    for n in nlist:
        nx = n
        ny = n
        print "Running Grid = (%d x %d)" % (nx, ny)

        # Rotated Anisotropic Diffusion Operator
        stencil = diffusion_stencil('FE',eps=epsilon,beta=beta)
        A = stencil_grid(stencil, (nx,ny), format='csr')

        # Random initial guess, zero RHS
        x0 = rand(A.shape[0])
        b = zeros((A.shape[0],))

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

        # ODE strength measure
        ml = smoothed_aggregation_solver(A, max_coarse=mcoarse, coarse_solver='pinv2', 
                                         presmoother=prepost, postsmoother=prepost, smooth=smooth,
                                         strength=('ode', {'epsilon' : ode_theta, 'k' : 2}) )
        resvec = []
        x = ml.solve(b, x0=x0, maxiter=100, tol=1e-8, residuals=resvec)
        factors_ode[run]    = (resvec[-1]/resvec[0])**(1.0/len(resvec))
        complexity_ode[run] = ml.operator_complexity()
        nnz_ode[run]        = A.nnz
        sizelist_ode[run]   = A.shape[0]

        run +=1

    # Print Problem Description
    print "\nAMG Scalability Study for Ax = 0, x = rand"
    print "Emphasis on Robustness of Strength Measures and Drop Tolerances"
    print "Rotated Anisotropic Diffusion in 2D"
    print "Anisotropic Coefficient = %1.3e" % epsilon
    print "Rotation Angle = %1.3f" % beta

    # Print Tables
    print_scalability(factors_classic, complexity_classic, 
         nnz_classic, sizelist_classic, plotting=False, 
         title='Classic Strength Measure DropTol = %1.2f' % classic_theta)
    print_scalability(factors_ode, complexity_ode, nnz_ode, sizelist_ode, 
         plotting=False, title='ODE Strength Measure DropTol = %1.2f' % ode_theta)


