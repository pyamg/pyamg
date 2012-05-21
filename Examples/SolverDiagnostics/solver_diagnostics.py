import scipy
import numpy
from scipy import rand, zeros, hstack, vstack, mat, sparse, log10, argsort, inf
from numpy import ones, ravel, arange, mod, array, abs, kron, eye, random
from scipy.sparse import csr_matrix, isspmatrix_bsr, isspmatrix_csr
from scipy.io import savemat, loadmat

from pyamg.aggregation import smoothed_aggregation_solver, rootnode_solver
from pyamg.util.linalg import norm, _approximate_eigenvalues, ishermitian
from pyamg.util.utils import print_table 

def solver_diagnostics(A, solver=smoothed_aggregation_solver, 
        fname='solver_diagnostic', definiteness=None,
        symmetry=None, strength_list=None, aggregate_list=None,
        smooth_list=None, Bimprove_list=None, max_levels_list=None,
        cycle_list=None, krylov_list=None, prepostsmoother_list=None,
        B_list=None, coarse_size_list=None):
    ''' 
    Try many different different parameter combinations for
    smoothed_aggregation_solver(...).  The goal is to find appropriate SA
    parameter settings for the arbitrary matrix problem A x = 0 using a 
    random initial guess.
    
    Every combination of the input parameter lists is used to construct and
    test an SA solver.  Thus, be wary of the total number of solvers possible!
    For example for an SPD CSR matrix, the default parameter lists generate 60
    different smoothed aggregation solvers.

    Symmetry and definiteness are automatically detected, but it is safest to
    manually set these parameters through the ``definiteness" and ``symmetry"
    parameters.
    
    Parameters
    ----------
    A : {csr_matrix, bsr_matrix}
        Sparse NxN matrix in CSR or BSR format
    
    solver : {smoothed_aggregation_solver, rootnode_solver}
        Solver to run diagnostic on.  Currently, these two solvers are supported.

    fname : {string}
        File name where the diagnostic results are dumped

        Default: solver_diagnostic.txt
    
    definiteness : {string}
        'positive' denotes positive definiteness
        'indefinite' denotes indefiniteness

        Default: detected with a few iterations of Arnoldi iteration
    
    symmetry : {string}
        'hermitian' or 'nonsymmetric', denoting the symmetry of the matrix

        Default: detected by testing if A induces an inner-product
    
    strength_list : {list} 
        List of various parameter choices for the strength argument sent to solver(...)
        
        Default:  [('symmetric', {'theta' : 0.0}), 
                   ('evolution', {'k':2, 'proj_type':'l2', 'epsilon':2.0}),
                   ('evolution', {'k':2, 'proj_type':'l2', 'epsilon':4.0})]
    
    aggregate_list : {list} 
        List of various parameter choices for the aggregate argument sent to solver(...)

        Default: ['standard']
    
    smooth_list : {list} 
        List of various parameter choices for the smooth argument sent to solver(...)

        Default depends on the symmetry and definiteness parameters:
        if definiteness == 'positive' and (symmetry=='hermitian' or symmetry=='symmetric'):
            ['jacobi', ('jacobi', {'filter' : True, 'weighting' : 'local'}),
            ('energy',{'krylov':'cg','maxiter':2, 'degree':1, 'weighting':'local'}),
            ('energy',{'krylov':'cg','maxiter':3, 'degree':2, 'weighting':'local'}),
            ('energy',{'krylov':'cg','maxiter':4, 'degree':3, 'weighting':'local'})]
        if definiteness == 'indefinite' or symmetry=='nonsymmetric':
           [('energy',{'krylov':'gmres','maxiter':2,'degree':1,'weighting':'local'}),
            ('energy',{'krylov':'gmres','maxiter':3,'degree':2,'weighting':'local'}),
            ('energy',{'krylov':'gmres','maxiter':3,'degree':3,'weighting':'local'})] 

    Bimprove_list : {list} 
        List of various parameter choices for the Bimprove argument sent to solver(...)

        Default: ['default', None]

    max_levels_list : {list} 
        List of various parameter choices for the max_levels argument sent to solver(...)
        
        Default: [25]
    
    cycle_list : {list} 
        List of various parameter choices for the cycle argument sent to solver.solve() 
        
        Default: ['V', 'W']
    
    krylov_list : {list} 
        List of various parameter choices for the krylov argument sent to
        solver.solve().  Basic form is (string, dict), where the string is a
        Krylov descriptor, e.g., 'cg' or 'gmres', and dict is a dictionary of
        parameters like tol and maxiter.  The dictionary dict may be empty.
      
        Default depends on the symmetry and definiteness parameters:
        if symmetry == 'nonsymmetric' or definiteness == 'indefinite':     
            [('gmres', {'tol':1e-8, 'maxiter':300})]
        else:
            [('cg', {'tol':1e-8, 'maxiter':300})]

    prepostsmoother_list : {list} 
        List of various parameter choices for the presmoother and postsmoother
        arguments sent to solver(...).  Basic form is 
        [ (presmoother_descriptor, postsmoother_descriptor), ...].
        
        Default depends on the symmetry parameter:
        if symmetry == 'nonsymmetric' or definiteness == 'indefinite':
            [ (('gauss_seidel_nr', {'sweep':'symmetric', 'iterations':2}),
               ('gauss_seidel_nr', {'sweep':'symmetric', 'iterations':2})) ]
        else:
            [ (('block_gauss_seidel',{'sweep':'symmetric','iterations':1}),
               ('block_gauss_seidel',{'sweep':'symmetric','iterations':1})) ]
        
    B_list : {list} 
        List of various B parameter choices for the B and BH arguments sent to
        solver(...).  Basic form is [ (B, BH, string), ...].  B is a vector of
        left near null-space modes used to generate prolongation, BH is a
        vector of right near null-space modes used to generate restriction, and
        string is a python command(s) that can generate your particular B and
        BH choice.  B and BH must have a row-size equal to the dimensionality
        of A.  string is only used in the automatically generated test script.

        Default depends on whether A is BSR:
        if A is CSR:
            B_list = [(ones((A.shape[0],1)), ones((A.shape[0],1)), 'B, BH are all ones')]
        if A is BSR:
            bsize = A.blocksize[0]
            B_list = [(ones((A.shape[0],1)), ones((A.shape[0],1)), 'B, BH are all ones'),
                      (kron(ones((A.shape[0]/bsize,1)), numpy.eye(bsize)), 
                       kron(ones((A.shape[0]/bsize,1)), numpy.eye(bsize)),
                       'B = kron(ones((A.shape[0]/A.blocksize[0],1), dtype=A.dtype), 
                                 eye(A.blocksize[0])); BH = B.copy()')]

    coarse_size_list : {list} 
        List of various tuples containing pairs of the (max_coarse, coarse_solver)
        parameters sent to solver(...).  

        Default: [ (300, 'pinv') ]

    Notes
    -----
    Only smoothed_aggregation_solver(...) and rootnode_solver(...) are
    supported.  The Ruge-Stuben solver framework is not used.
    
    60 total solvers are generated by the defaults for CSR SPD matrices.  For
    BSR SPD matrices, 120 total solvers are generated by the defaults.  A
    somewhat smaller number of total solvers is generated if the matrix is
    indefinite or nonsymmetric.  Every combination of the parameter lists is
    attempted.

    Generally, there are two types of parameter lists passed to this function.  
    Type 1 includes: cycle_list, strength_list, aggregate_list, smooth_list, 
                     krylov_list, Bimprove_list, max_levels_list
                     -------------------------------------------
                     Here, you pass in a list of different parameters, e.g., 
                     cycle_list=['V','W'].

    Type 2 includes: B_list, coarse_size_list, prepostsmoother_list 
                     -------------------------------------------
                     This is similar to Type 1, only these represent lists of
                     pairs of parameters, e.g., 
                     coarse_size_list=[ (300, 'pinv'), (5000, 'splu')], 
                     where coarse size_list is of the form 
                     [ (max_coarse, coarse_solver), ...].

    For detailed info on each of these parameter lists, see above.

    Returns
    -------
    Two files are written:
    (1) fname + ".py"
        Use the function defined here to generate and run the best 
        smoothed aggregation method found.  The only argument taken
        is a BSR/CSR matrix.
    (2) fname + ".txt"
        This file outputs the solver profile for each method 
        tried in a sorted table listing the best solver first.
        The detailed solver descriptions then follow the table.
    
    See Also
    --------
    smoothed_aggregation_solver

    Examples
    --------
    >>> from pyamg import gallery
    >>> from solver_diagnostics import *
    >>> A = gallery.poisson( (50,50), format='csr') 
    >>> solver_diagnostics(A, fname='isotropic_diffusion_diagnostics.txt', cycle_list=['V'])
    
    '''
    
    ##
    # Preprocess A
    if not (isspmatrix_csr(A) or isspmatrix_bsr(A)):
        try:
            A = csr_matrix(A)
            print 'Implicit conversion of A to CSR'
        except:
            raise TypeError('Argument A must have type csr_matrix or bsr_matrix,\
                             or be convertible to csr_matrix')
    #
    A = A.asfptype()
    #
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')
    
    print "\nSearching for optimal smoothed aggregation method for (%d,%d) matrix"%A.shape
    print "    ..."
    
    ##
    # Detect symmetry
    if symmetry == None:
        if ishermitian(A, fast_check=True):
            symmetry = 'hermitian'
        else:
            symmetry = 'nonsymmetric'
        ##
        print "    Detected a " + symmetry + " matrix"
    else:
        print "    User specified a " + symmetry + " matrix"


    ##
    # Detect definiteness
    if definiteness == None:
        [EVect, Lambda, H, V, breakdown_flag] = _approximate_eigenvalues(A, 1e-6, 40)
        if Lambda.min() < 0.0:
            definiteness = 'indefinite'
            print "    Detected indefiniteness"
        else:
            definiteness = 'positive'
            print "    Detected positive definiteness"
    else:
        print "    User specified definiteness as " + definiteness 

    ##
    # Default B are (1) a vector of all ones, and 
    # (2) if A is BSR, the constant for each variable
    if B_list == None:
        B_list = [(ones((A.shape[0],1), dtype=A.dtype), 
                   ones((A.shape[0],1), dtype=A.dtype), 
                   'B = ones((A.shape[0],1), dtype=A.dtype); BH = B.copy()')]

        if isspmatrix_bsr(A) and A.blocksize[0] > 1:
            bsize = A.blocksize[0]
            B_list.append( (kron(ones((A.shape[0]/bsize,1), dtype=A.dtype),eye(bsize)), 
              kron(ones((A.shape[0]/bsize,1), dtype=A.dtype),eye(bsize)),
              'B = kron(ones((A.shape[0]/A.blocksize[0],1), dtype=A.dtype), eye(A.blocksize[0])); BH = B.copy()'))
    
    ##
    # Default is to try V- and W-cycles
    if cycle_list == None:
        cycle_list = ['V', 'W']

    ##
    # Default strength of connection values
    if strength_list == None:
        strength_list = [('symmetric', {'theta' : 0.0}),
                         ('evolution', {'k':2, 'proj_type':'l2', 'epsilon':2.0}),
                         ('evolution', {'k':2, 'proj_type':'l2', 'epsilon':4.0})]

    ##
    # Default aggregation strategies
    if aggregate_list is None:
        aggregate_list = ['standard']
    
    ##
    # Default prolongation smoothers
    if smooth_list == None:
        if definiteness == 'positive' and (symmetry=='hermitian' or symmetry=='symmetric'):
            if solver.func_name == 'smoothed_aggregation_solver':
                smooth_list = ['jacobi', ('jacobi', {'filter' : True, 'weighting' : 'local'})]
            else:
                smooth_list = []
            ##
            smooth_list.append( ('energy',{'krylov':'cg','maxiter':2,'degree':1,'weighting':'local'}) )
            smooth_list.append( ('energy',{'krylov':'cg','maxiter':3,'degree':2,'weighting':'local'}) )
            smooth_list.append( ('energy',{'krylov':'cg','maxiter':4,'degree':3,'weighting':'local'}) )
        elif definiteness == 'indefinite' or symmetry=='nonsymmetric':
            smooth_list =[('energy',{'krylov':'gmres','maxiter':2,'degree':1,'weighting':'local'}),
                          ('energy',{'krylov':'gmres','maxiter':3,'degree':2,'weighting':'local'}),
                          ('energy',{'krylov':'gmres','maxiter':4,'degree':3,'weighting':'local'})]
        else:
            raise ValueError('invalid string for definiteness and/or symmetry')

    ##
    # Default pre- and postsmoothers
    if prepostsmoother_list == None:
        if symmetry == 'nonsymmetric' or definiteness == 'indefinite':
            prepostsmoother_list = [ (('gauss_seidel_nr', {'sweep':'symmetric', 'iterations':2}),
                                      ('gauss_seidel_nr', {'sweep':'symmetric', 'iterations':2})) ]
        else:
            prepostsmoother_list= [ (('block_gauss_seidel',{'sweep':'symmetric','iterations':1}),
                                     ('block_gauss_seidel',{'sweep':'symmetric','iterations':1})) ]
    
    ##
    # Default Krylov wrapper
    if krylov_list == None:
        if symmetry == 'nonsymmetric' or definiteness == 'indefinite':
            krylov_list = [('gmres', {'tol':1e-8, 'maxiter':300})]
        else:
            krylov_list = [('cg', {'tol':1e-8, 'maxiter':300})]

    ##
    # Default Bimprove
    if Bimprove_list == None:
        Bimprove_list = ['default', None]

    ##
    # Default basic solver parameters
    if max_levels_list == None:
        max_levels_list = [25]
    if coarse_size_list == None:
        coarse_size_list = [ (300, 'pinv') ]
   
    ##
    # Setup for ensuing numerical tests
    # The results array will hold in each row, three values: 
    # iterations, operator complexity, and work per digit of accuracy
    num_test = len(cycle_list)*len(strength_list)*len(aggregate_list)*len(smooth_list)* \
               len(krylov_list)*len(Bimprove_list)*len(max_levels_list)*len(B_list)* \
               len(coarse_size_list)*len(prepostsmoother_list)
    results = zeros( (num_test,3) )
    solver_descriptors = []
    solver_args = []
    
    ##
    # Zero RHS and random initial guess
    random.seed(0)
    b = zeros( (A.shape[0],1), dtype=A.dtype)
    x0 = rand( A.shape[0], 1)
    if A.dtype == complex:
        x0 += 1.0j*rand( A.shape[0], 1)

    ##
    # Begin loops over parameter choices
    print "    ..."
    counter = -1
    for cycle in cycle_list:
        for krylov in krylov_list:
            for max_levels in max_levels_list:
                for max_coarse,coarse_solver in coarse_size_list:
                    for presmoother,postsmoother in prepostsmoother_list:
                        for B_index in range(len(B_list)): 
                            for strength in strength_list:
                                for aggregate in aggregate_list:
                                    for smooth in smooth_list:
                                        for Bimprove in Bimprove_list:
                                            
                                            counter += 1
                                            print "    Test %d out of %d"%(counter+1,num_test)
                                            
                                            ##
                                            # Grab B vectors
                                            B,BH,Bdescriptor = B_list[B_index]
                                            
                                            ##
                                            # Store this solver setup
                                            if krylov[1].has_key('tol'):
                                                tol = krylov[1]['tol']
                                            else:
                                                tol = 1e-6
                                            if krylov[1].has_key('maxiter'):
                                                maxiter = krylov[1]['maxiter']
                                            else:
                                                maxiter = 300
                                            ##
                                            descriptor = '  Solve phase arguments:' + '\n' \
                                                '    cycle = ' + str(cycle) + '\n' \
                                                '    krylov accel = ' + str(krylov[0]) + '\n' \
                                                '    tol = ' + str(tol) + '\n' \
                                                '    maxiter = ' + str(maxiter)+'\n'\
                                                '  Setup phase arguments:' + '\n' \
                                                '    max_levels = ' + str(max_levels) + '\n' \
                                                '    max_coarse = ' + str(max_coarse) + '\n' \
                                                '    coarse_solver = ' + str(coarse_solver)+'\n'\
                                                '    presmoother = ' + str(presmoother) + '\n' \
                                                '    postsmoother = ' + str(postsmoother) + '\n'\
                                                '    ' + Bdescriptor + '\n' \
                                                '    strength = ' + str(strength) + '\n' \
                                                '    aggregate = ' + str(aggregate) + '\n' \
                                                '    smooth = ' + str(smooth) + '\n' \
                                                '    Bimprove = ' + str(Bimprove) 
                                            solver_descriptors.append(descriptor)
                                            solver_args.append( {'cycle' : cycle, 
                                                'accel' : str(krylov[0]),
                                                'tol' : tol, 'maxiter' : maxiter, 
                                                'max_levels' : max_levels, 'max_coarse' : max_coarse,
                                                'coarse_solver' : coarse_solver, 'B_index' : B_index,
                                                'presmoother' : presmoother, 
                                                'postsmoother' : postsmoother,
                                                'strength' : strength, 'aggregate' : aggregate,
                                                'smooth' : smooth, 'Bimprove' : Bimprove} )
                                            
                                            ##
                                            # Construct solver
                                            try:
                                                sa = solver(A, B=B, BH=BH,
                                                        strength=strength,
                                                        smooth=smooth,
                                                        Bimprove=Bimprove,
                                                        aggregate=aggregate,
                                                        presmoother=presmoother,
                                                        max_levels=max_levels,
                                                        postsmoother=postsmoother,
                                                        max_coarse=max_coarse,
                                                        coarse_solver=coarse_solver)
                                            
                                                ##
                                                # Solve system
                                                residuals = []
                                                x = sa.solve(b, x0=x0, accel=krylov[0], 
                                                  cycle=cycle, tol=tol, maxiter=maxiter, 
                                                  residuals=residuals)

                                                ##
                                                # Store results: iters, operator complexity, and
                                                # work per digit-of-accuracy
                                                results[counter,0] = len(residuals)
                                                results[counter,1] = sa.operator_complexity()
                                                resid_rate = (residuals[-1]/residuals[0])**\
                                                             (1.0/(len(residuals)-1.))
                                                results[counter,2] = sa.cycle_complexity()/ \
                                                                     abs(log10(resid_rate))

                                            except:
                                                descriptor_indented = '      ' + \
                                                  descriptor.replace('\n', '\n      ')
                                                print"    --> Failed this test"
                                                print"    --> Solver descriptor is..."
                                                print descriptor_indented
                                                results[counter,:] = inf
    ##
    # Sort results and solver_descriptors according to work-per-doa
    indys = argsort(results[:,2])
    results = results[indys,:]
    solver_descriptors = list(array(solver_descriptors)[indys])
    solver_args = list(array(solver_args)[indys])

    ##
    # Create table from results and print to file
    table = [ ['solver #', 'iters', 'op complexity', 'work per DOA'] ]
    for i in range(results.shape[0]):
        if (results[i,:] == inf).all() == True:
            # in this case the test failed...
            table.append(['%d'%(i+1), 'err', 'err', 'err'])
        else:
            table.append(['%d'%(i+1),'%d'%results[i,0],'%1.1f'%results[i,1],'%1.1f'%results[i,2]])
    #
    fptr = open(fname+'.txt', 'w')
    fptr.write('****************************************************************\n' + \
               '*                Begin Solver Diagnostic Results               *\n' + \
               '*                                                              *\n' + \
               '*        \'\'solver #\'\' refers to below solver descriptors       *\n' + \
               '*                                                              *\n' + \
               '*        \'\'iters\'\' refers to iterations taken                  *\n' + \
               '*                                                              *\n' + \
               '*        \'\'op complexity\'\' refers to operator complexity       *\n' + \
               '*                                                              *\n' + \
               '*        \'\'work per DOA\'\' refers to work per digit of          *\n' + \
               '*          accuracy to solve the algebraic system, i.e. it     *\n' + \
               '*          measures the overall efficiency of the solver       *\n' + \
               '****************************************************************\n\n')
    fptr.write(print_table(table))

    ##
    # Now print each solver descriptor to file
    fptr.write('\n****************************************************************\n' + \
                 '*                 Begin Solver Descriptors                     \n' + \
                 '*       Solver used is ' + solver.func_name + '( )             \n' + \
                 '****************************************************************\n\n')

    for i in range(len(solver_descriptors)):
        fptr.write('Solver Descriptor %d\n'%(i+1))
        fptr.write(solver_descriptors[i])
        fptr.write(' \n \n')
    
    fptr.close()
    
    ##
    # Now write a function definition file that generates the "best" solver
    fptr = open(fname + '.py', 'w')
    # Helper function for file writing
    def to_string(a):
        if type(a) == type((1,)):   return(str(a))
        elif type(a) == type('s'):  return("\"%s\""%a)
        else: return str(a)
    #
    fptr.write('#######################################################################\n')
    fptr.write('# Function definition automatically generated by solver_diagnostics.py\n')
    fptr.write('#\n')
    fptr.write('# Use the function defined here to generate and run the best\n')
    fptr.write('# smoothed aggregation method found by solver_diagnostics(...).\n')
    fptr.write('# The only argument taken is a CSR/BSR matrix.\n')
    fptr.write('#\n')
    fptr.write('# To run:  >>> # User must load/generate CSR/BSR matrix A\n')
    fptr.write('#          >>> from ' + fname + ' import ' + fname + '\n' )
    fptr.write('#          >>> ' + fname + '(A)' + '\n')
    fptr.write('#######################################################################\n\n')
    fptr.write('from pyamg import ' + solver.func_name + '\n')
    fptr.write('from pyamg.util.linalg import norm\n') 
    fptr.write('from numpy import ones, array, arange, zeros, abs, random\n') 
    fptr.write('from scipy import rand, ravel, log10, kron, eye\n') 
    fptr.write('from scipy.io import loadmat\n') 
    fptr.write('from scipy.sparse import isspmatrix_bsr, isspmatrix_csr\n') 
    fptr.write('import pylab\n\n')
    fptr.write('def ' + fname + '(A):\n') 
    fptr.write('    ##\n    # Generate B\n')
    fptr.write('    ' + B_list[B_index][2] + '\n\n')
    fptr.write('    ##\n    # Random initial guess, zero right-hand side\n')
    fptr.write('    random.seed(0)\n')
    fptr.write('    b = zeros((A.shape[0],1))\n')
    fptr.write('    x0 = rand(A.shape[0],1)\n\n')
    fptr.write('    ##\n    # Create solver\n')
    fptr.write('    ml = ' + solver.func_name + '(A, B=B, BH=BH,\n' + \
               '        strength=%s,\n'%to_string(solver_args[0]['strength']) + \
               '        smooth=%s,\n'%to_string(solver_args[0]['smooth']) + \
               '        Bimprove=%s,\n'%to_string(solver_args[0]['Bimprove']) + \
               '        aggregate=%s,\n'%to_string(solver_args[0]['aggregate']) + \
               '        presmoother=%s,\n'%to_string(solver_args[0]['presmoother']) + \
               '        postsmoother=%s,\n'%to_string(solver_args[0]['postsmoother']) + \
               '        max_levels=%s,\n'%to_string(solver_args[0]['max_levels']) + \
               '        max_coarse=%s,\n'%to_string(solver_args[0]['max_coarse']) + \
               '        coarse_solver=%s)\n\n'%to_string(solver_args[0]['coarse_solver']) ) 
    fptr.write('    ##\n    # Solve system\n')
    fptr.write('    res = []\n')
    fptr.write('    x = ml.solve(b, x0=x0, tol=%s, residuals=res, accel=%s, maxiter=%s, cycle=%s)\n'%\
              (to_string(solver_args[0]['tol']),
               to_string(solver_args[0]['accel']),
               to_string(solver_args[0]['maxiter']),
               to_string(solver_args[0]['cycle'])) ) 
    fptr.write('    res_rate = (res[-1]/res[0])**(1.0/(len(res)-1.))\n')
    fptr.write('    normr0 = norm(ravel(b) - ravel(A*x0))\n')
    fptr.write('    print " "\n')
    fptr.write('    print ml\n')
    fptr.write("    print \"System size:                \" + str(A.shape)\n")
    fptr.write("    print \"Avg. Resid Reduction:       %1.2f\"%res_rate\n")
    fptr.write("    print \"Iterations:                 %d\"%len(res)\n")
    fptr.write("    print \"Operator Complexity:        %1.2f\"%ml.operator_complexity()\n")
    fptr.write("    print \"Work per DOA:               %1.2f\"%(ml.cycle_complexity()/abs(log10(res_rate)))\n")
    fptr.write("    print \"Relative residual norm:     %1.2e\"%(norm(ravel(b) - ravel(A*x))/normr0)\n\n")
    fptr.write('    ##\n    # Plot residual history\n')
    fptr.write('    pylab.semilogy(array(res)/normr0)\n') 
    fptr.write('    pylab.title(\'Residual Histories\')\n')
    fptr.write('    pylab.xlabel(\'Iteration\')\n')
    fptr.write('    pylab.ylabel(\'Relative Residual Norm\')\n')
    fptr.write('    pylab.show()\n\n')
    # Close file pointer
    fptr.close()

    print "    ..."
    print "    --> Diagnostic Results located in " + fname + '.txt'
    print "    ..."
    print "    --> See automatically generated function definition\n" + \
          "        ./" + fname + ".py.\n\n" + \
          "        Use the function defined here to generate and run the best\n" + \
          "        smoothed aggregation method found.  The only argument taken\n" + \
          "        is a CSR/BSR matrix.\n\n" + \
          "        To run: >>> # User must load/generate CSR/BSR matrix A\n" + \
          "                >>> from " + fname + " import " + fname + "\n" + \
          "                >>> " + fname + "(A)"


