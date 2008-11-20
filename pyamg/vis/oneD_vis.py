"""
For use with pylab and 1D problems solved with a multilevel method.
Routines here allow you to visualized aggregates, nullspace vectors
and columns of P.  This is possible on any level
"""
__docformat__ = "restructuredtext en"

import warnings

__all__ = ['oneD_P_vis', 'oneD_coarse_grid_vis', 'oneD_nullspace_vis', 'oneD_profile']

import pylab
from numpy import ravel, zeros, ones, min, abs, array, max, pi
from scipy import imag, real, linspace, exp, rand
from scipy.sparse import bsr_matrix
from scipy.linalg import solve 
from pyamg.util.linalg import norm

def oneD_profile(mg, grid=None, x0=None, b=None, soln=None, iter=1, cycle='V', fig_num=1):
    '''
    Profile mg on the problem defined by x0 and b.  
    Default problem is x0=rand, b = 0.

    Parameters
    ----------
    mg : pyamg multilevel hierarchy
        Hierarchy to profile
    grid : array
        if None, default grid is assumed to be on [0,1]
    x0 : array
        initial guess to linear system, default is a random
    b : array
        right hand side to linear system, default is all zeros
        Note that if b is not all zeros and soln is not provided,
        A must be inverted in order to plot the error
    soln: array
        soln to the linear system
    iter : int
        number of cycle iterations, default is 1
    cycle : {'V', 'W', 'F'}
        solve with a V, W or F cycle
    fig_num : int
        figure number from which to begin plotting
    
    Returns
    -------
    The error, residual and residual ratio history are sent to the plotter.
    To see plots, type pyamg.show()

    Notes
    -----

    Examples
    --------
    >>>from pyamg import *
    >>>from pyamg.vis.oneD_vis import *
    >>>import pylab
    >>>from scipy import rand, zeros
    >>>A = poisson( (128,), format='csr')
    >>>ml=smoothed_aggregation_solver(A, max_coarse=5)
    >>>oneD_profile(ml);                                         pylab.show()
    >>>oneD_profile(ml, iter=3);                                 pylab.show()
    >>>oneD_profile(ml, iter=3, cycle='W');                      pylab.show()
    >>>oneD_profile(ml, b=rand(128,), x0=zeros((128,)), iter=5); pylab.show()
    '''
    
    A = mg.levels[0].A
    ndof = mg.levels[0].A.shape[0]

    # Default regular grid on 0 to 1
    if grid == None:
        grid = linspace(0,1,ndof)
    elif ravel(grid).shape[0] != ndof:
        raise ValueError("Grid must be of size %d" % ndof)

    # Default initial guess is random
    if x0 == None:
        x0 = rand(ndof,)
        if A.dtype == complex:
            x0 += 1.0j*rand(ndof,)
    elif ravel(x0).shape[0] != ndof:
        raise ValueError("Initial guess must be of size %d" % ndof)
    else:
        x0 = ravel(x0)

    # Default RHS is all zero
    if b == None:
        b = zeros((ndof,), dtype=A.dtype)
    elif ravel(b).shape[0] != ndof:
        raise ValueError("RHS must be of size %d" % ndof)
    else:
        b = ravel(b)
   
    # Must invert A to find soln, if RHS is not all zero
    if soln == None:
        if b.any():
            soln = solve(A.todense(), b)
        else:
            soln = zeros((ndof,), dtype=A.dtype)
    elif ravel(soln).shape[0] != ndof:
        raise ValueError("Soln must be of size %d" % ndof)
    else:
        soln = ravel(soln)

    # solve system with mg
    res = []
    guess = mg.solve(b, x0=x0, tol=1e-8, maxiter=iter, cycle=cycle, residuals=res)
    res = array(res)
    resratio = res[1:]/res[0:-1]
    r = b - A*guess
    e = soln - guess

    # plot results
    if iter > 1:
        pylab.figure(fig_num)
        pylab.plot(array(range(1,resratio.shape[0]+1)), resratio)
        title = pylab.title('Residual Ratio History')
        title.set_fontsize(18)
        xlabel = pylab.xlabel('Iteration')
        xlabel.set_fontsize(18)
        ylabel = pylab.ylabel('||r_{i+1}|| / ||r_{i}||')
        ylabel.set_fontsize(18)

    pylab.figure(fig_num+1)
    pylab.plot(grid, r)
    title = pylab.title('Final Residual')
    title.set_fontsize(18)
    xlabel = pylab.xlabel('X')
    xlabel.set_fontsize(18)
    ylabel = pylab.ylabel('b - Ax')
    ylabel.set_fontsize(18)
    
    pylab.figure(fig_num+2)
    pylab.plot(grid, e)
    title = pylab.title('Final Error')
    title.set_fontsize(18)
    xlabel = pylab.xlabel('X')
    xlabel.set_fontsize(18)
    ylabel = pylab.ylabel('soln - x')
    ylabel.set_fontsize(18)

def oneD_nullspace_vis(mg, level=0, interp=False, fig_num=1, x=None):
    '''
    Plot the near nullspace modes from which P is built
    
    Parameters
    ----------
    mg : pyamg multilevel hierarchy
        visualize the components of mg
    
    level : int
        level of mg on which to visualize

    interp : {False, True}
        Should the modes at level=level be interpolated 
        to the finest grid before plotting?
 
    fig_num : int
        figure number from which to begin plotting

    x : array
        grid for the level on which to visualize
        if None, then an evenly space grid on [0,1] is assumed

    Returns
    -------
    Plots of the nullspace vectors on level=level are sent to the plotter
    To see plots, type pyamg.show()

    Notes
    -----

    Examples
    --------
    >>>from pyamg import *
    >>>from pyamg.vis.oneD_vis import *
    >>>import pylab
    >>>A = poisson( (64,), format='csr')
    >>>ml=smoothed_aggregation_solver(A, max_coarse=5)
    >>>oneD_nullspace_vis(ml, level=0)
    >>>pylab.show()
    >>>oneD_nullspace_vis(ml, level=1)
    >>>pylab.show()
    >>>oneD_nullspace_vis(ml, level=1, interp=True)
    >>>pylab.show()
    '''
    
    if level > (len(mg.levels)-1):
        raise ValueError("Level %d has no Nullspace Candidates" % level)

    pylab.figure(fig_num)
    colors = ['b-o', 'r-o', 'k-o', 'g-o', 'm-o', 'c-o', 'y-o', '#00ff7f', '#006400' ]
    
    if interp == False:
        ndof = mg.levels[level].A.shape[0]
        Btemp = mg.levels[level].B
    else:     
        # We are visualizing coarse grid basis 
        # functions interpolated up to the finest level
        ndof = mg.levels[0].A.shape[0]
        Btemp = mg.levels[level].B
        for i in range(level-1,-1,-1):
            Btemp = mg.levels[i].P*Btemp

    # Default regular grid on 0 to 1
    if x == None:
        x = linspace(0,1,ndof)
    
    # Plot modes
    for i in range(Btemp.shape[1]):
        pylab.plot(x, ravel(real(Btemp[:,i])), colors[i], label=("Mode %d" % i))
        title_string = 'Level ' + str(level) + ' Real Components of Null Space Modes'
        if interp and (level != 0):
            title_string += '\nInterpolated to Finest Level'
        title = pylab.title(title_string)
        title.set_fontsize(18)
        xlabel = pylab.xlabel('X')
        xlabel.set_fontsize(18)
        ylabel = pylab.ylabel('real(mode)')
        ylabel.set_fontsize(18)

    ax = array(pylab.axis())
    ax[2] = min(real(ravel(Btemp[:,:])))*0.9
    ax[3] = max(real(ravel(Btemp[:,:])))*1.1
    ax = pylab.axis(ax)
    
    if Btemp.dtype == complex:
        pylab.figure(fig_num+1)
        for i in range(Btemp.shape[1]):
            pylab.plot(x, ravel(imag(Btemp[:,i])), colors[i], label=("Mode %d" % i))
            title_string = 'Level ' + str(level) + ' Imag Components of Null Space Modes'
            if interp and (level != 0):
                title_string += '\nInterpolated to Finest Level'
            title = pylab.title(title_string)
            title.set_fontsize(18)
            xlabel = pylab.xlabel('X')
            xlabel.set_fontsize(18)
            ylabel = pylab.ylabel('imag(mode)')
            ylabel.set_fontsize(18)
        bottom = min(imag(Btemp[:,:]))*0.9
        top = max(imag(Btemp[:,:]))*1.1
        pylab.axis([min(x), max(x), bottom, top])


def oneD_coarse_grid_vis(mg, fig_num=1, x=None, level=0):
    '''
    Visualize the aggregates on level=level in terms of 
    the aggregates' fine grid representation
    
    Parameters
    ----------
    mg : pyamg multilevel hierarchy
         visualize the components of mg
    fig_num : int
        figure number from which to begin plotting
    x : array
        grid for the level on which to visualize
        if None, then an evenly space grid on [0,1] is assumed
    level : int
        level on which to visualize

    Returns
    -------
    A plot of the aggregates on level=level is sent to the plotter
    The aggregates are always interpolated to the finest level
    Use pylab.show() to see the plot

    Notes
    -----

    Examples
    --------
    >>>from pyamg import *
    >>>from pyamg.vis.oneD_vis import *
    >>>import pylab
    >>>A = poisson( (64,), format='csr')
    >>>ml=smoothed_aggregation_solver(A, max_coarse=5)
    >>>oneD_coarse_grid_vis(ml, level=0)
    >>>pylab.show()
    >>>oneD_coarse_grid_vis(ml, level=1)
    >>>pylab.show()

    '''

    if level > (len(mg.levels)-2):
        raise ValueError("Level %d has no AggOp" % level)

    ndof = mg.levels[0].A.shape[0]
    AggOp = mg.levels[level].AggOp
    for i in range(level-1,-1,-1):
        AggOp = mg.levels[i].AggOp*AggOp

    AggOp = AggOp.tocsc()

    # Default regular grid on 0 to 1
    if x == None:
        x = linspace(0,1,ndof)

    pylab.figure(fig_num)
    for i in range(AggOp.shape[1]):
        aggi = AggOp[:,i].indices
        pylab.plot(x[aggi], i*ones((aggi.shape[0],)), marker='o')
    
    title_string='Level ' + str(level) + ' Aggregates'
    if level != 0:
        title_string += '\nInterpolated to Finest Level'
    title = pylab.title(title_string)
    title.set_fontsize(18)
    xlabel=pylab.xlabel('X')
    xlabel.set_fontsize(18)
    ylabel = pylab.ylabel('Aggregate Number')
    ylabel.set_fontsize(18)
    pylab.axis([min(x)-.05, max(x)+.05, -1, AggOp.shape[1]])
    

def oneD_P_vis(mg, fig_num=1, x=None, level=0, interp=False):
    '''
    Visualize the basis functions of P (i.e. columns) from level=level
    
    Parameters
    ----------
    mg : pyamg multilevel hierarchy
        visualize the components of mg
    fig_num : int
        figure number from which to begin plotting
    x : array like
        grid for the level on which to visualize
        if None, then an evenly space grid on [0,1] is assumed
    level : int
        level on which to visualize
    interp : {True, False}
        Should the columns of P be interpolated to the finest 
        level before plotting? i.e., plot the columns of 
        (P_1*P_2*...*P_level) or just columns of P_level

    Returns
    -------
    A plot of the columns of P on level=level is sent to the plotter
    Use pylab.show() to see the plot

    Notes
    -----
    These plots are only useful for small grids as all columns 
    of P are actually printed

    Examples
    --------
    >>>from pyamg import *
    >>>from pyamg.vis.oneD_vis import *
    >>>import pylab
    >>>A = poisson( (64,), format='csr')
    >>>ml=smoothed_aggregation_solver(A, max_coarse=5)
    >>>oneD_P_vis(ml, level=0, interp=False)
    >>>pylab.show()
    >>>oneD_P_vis(ml, level=1, interp=False)
    >>>pylab.show()
    >>>oneD_P_vis(ml, level=1, interp=True)
    >>>pylab.show()
    '''
    
    if level > (len(mg.levels)-2):
        raise ValueError("Level %d has no P" % level)

    if interp == False:
        ndof = mg.levels[level].A.shape[0]
        P = mg.levels[level].P
    else:     
        # We are visualizing coarse grid basis 
        # functions interpolated up to the finest level
        ndof = mg.levels[0].A.shape[0]
        P = mg.levels[level].P
        for i in range(level-1,-1,-1):
            P = mg.levels[i].P*P

    blocks = P.blocksize[1]
    P = P.tocsc()

    # Default regular grid on 0 to 1
    if x == None:
        x = linspace(0,1,ndof)
    
    # Grab and plot each aggregate's part of a basis function together
    for i in range(0, P.shape[1], blocks):
        #extract aggregate i's basis functions
        p = P[:, i:(i+blocks)].todense()
        for j in range(blocks):
            pylab.figure(fig_num+j)
            p2 = ravel(p[:,j])
            pylab.plot(x[p2!=0], real(p2[p2!=0.0]), marker='o')
            axline = pylab.axhline(color='k')
            title_string = ('Level ' + str(level) + ' Real Components of Basis Function %d' % j)
            if interp and (level != 0):
                title_string += '\nInterpolated to Finest Level'
            title = pylab.title(title_string)
            title.set_fontsize(18)
            xlabel = pylab.xlabel('X')
            xlabel.set_fontsize(18)
            ylabel = pylab.ylabel('real(mode)')
            ylabel.set_fontsize(18)
            ax = array(pylab.axis()); ax[0] = min(x); ax[1] = max(x)
            ax = pylab.axis(ax)
            
            if P.dtype == complex:
                pylab.figure(fig_num+10+j)
                p2 = ravel(p[:,j])
                pylab.plot(x[p2!=0], imag(p2[p2!=0.0]), marker='o')
                axline = pylab.axhline(color='k')
                title_string = ('Level ' + str(level) + ' Imag Components of Basis Function %d' % j)
                if interp and (level != 0):
                    title_string += '\nInterpolated to Finest Level'
                title = pylab.title(title_string)
                title.set_fontsize(18)
                xlabel = pylab.xlabel('X')
                xlabel.set_fontsize(18)
                ylabel = pylab.ylabel('imag(mode)')
                ylabel.set_fontsize(18)
                ax = array(pylab.axis()); ax[0] = min(x)
                ax[1] = max(x); ax = pylab.axis(ax)

