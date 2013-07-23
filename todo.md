- fix wiki pages
- move wiki to pages (org or project?)
- update pyamg.org
- specific points 
- point to code
- point to examples
- analytics (google or bitdeli?)
- update install scripts
- clearly block out strength, aggregation, interp routines
- move BImprove before strength?  (test before/after)
- clean up symmetry flag

- change Bimprove to improve_candidates  (consistent with adaptive SA solver)
----> Double check this...run solver diagnostics and other examples
bash-3.2$ egrep "Bimprove" ./*/*.py
./0STARTHERE/demo.py:        Bimprove='default',                # use the default 5 sweeps of prerelaxing B at each level
./ComplexSymmetric/demo_two_D_helmholtz.py:    # Bimprove[k] -- stipulates the relaxation method on level k used to "improve" B
./ComplexSymmetric/demo_two_D_helmholtz.py:    Bimprove = [ ('gauss_seidel', {'iterations' : 2, 'sweep':'forward'}), 
./ComplexSymmetric/demo_two_D_helmholtz.py:               aggregate=aggregate, Bimprove=Bimprove, presmoother=smoother,
./ComplexSymmetric/demo_two_D_helmholtz.py:                aggregate=aggregate, Bimprove=Bimprove, presmoother=smoother,
./ComplexSymmetric/smoothed_aggregation_helmholtz_solver.py:from pyamg.aggregation.aggregation import extend_hierarchy, preprocess_Bimprove, \
./ComplexSymmetric/smoothed_aggregation_helmholtz_solver.py:        Bimprove='default', max_levels = 10, max_coarse = 100, **kwargs):
./ComplexSymmetric/smoothed_aggregation_helmholtz_solver.py:    Bimprove : {list} : default [('block_gauss_seidel', {'sweep':'symmetric'}), None]
./ComplexSymmetric/smoothed_aggregation_helmholtz_solver.py:    Bimprove = preprocess_Bimprove(Bimprove, A, max_levels)
./ComplexSymmetric/smoothed_aggregation_helmholtz_solver.py:            if Bimprove[0] is not None:
./ComplexSymmetric/smoothed_aggregation_helmholtz_solver.py:                Bcoarse2 = relaxation_as_linear_operator(Bimprove[0], A, zeros_0)*Bcoarse2
./ComplexSymmetric/smoothed_aggregation_helmholtz_solver.py:            if Bimprove[len(levels)-1] is not None:
./ComplexSymmetric/smoothed_aggregation_helmholtz_solver.py:                Bcoarse2 =relaxation_as_linear_operator(Bimprove[len(levels)-1],A_l,zeros_l)*Bcoarse2
./Diffusion/demo_local_disc_galerkin_diffusion.py:    Bimprove = [('block_gauss_seidel', {'sweep':'symmetric', 'iterations':p}),
./Diffusion/demo_local_disc_galerkin_diffusion.py:    sa = smoothed_aggregation_solver(A, B=B, smooth=smooth, Bimprove=Bimprove,\
./SolverDiagnostics/solver_diagnostics.py:        smooth_list=None, Bimprove_list=None, max_levels_list=None,
./SolverDiagnostics/solver_diagnostics.py:    Bimprove_list : {list} 
./SolverDiagnostics/solver_diagnostics.py:        List of various parameter choices for the Bimprove argument sent to solver(...)
./SolverDiagnostics/solver_diagnostics.py:                     krylov_list, Bimprove_list, max_levels_list
./SolverDiagnostics/solver_diagnostics.py:    # Default Bimprove
./SolverDiagnostics/solver_diagnostics.py:    if Bimprove_list == None:
./SolverDiagnostics/solver_diagnostics.py:        Bimprove_list = ['default', None]
./SolverDiagnostics/solver_diagnostics.py:               len(krylov_list)*len(Bimprove_list)*len(max_levels_list)*len(B_list)* \
./SolverDiagnostics/solver_diagnostics.py:                                        for Bimprove in Bimprove_list:
./SolverDiagnostics/solver_diagnostics.py:                                                '    Bimprove = ' + str(Bimprove) 
./SolverDiagnostics/solver_diagnostics.py:                                                'smooth' : smooth, 'Bimprove' : Bimprove} )
./SolverDiagnostics/solver_diagnostics.py:                                                        Bimprove=Bimprove,
./SolverDiagnostics/solver_diagnostics.py:               '        Bimprove=%s,\n'%to_string(solver_args[0]['Bimprove']) + \

- move default option from preprocess_Bimprove somewhere else, just set the
  default to ('gauss_seidel', {'sweep':'symmetric', 'iterations':4}), but then
  you have to worry about symmetry.
- change 'scheme' to consistent usage
- rename preprocess_Bimprove, preprocess_str_or_agg, preprocess_smooth to levelize -- 
- fix returns for preprocess
- preprocess stuff needs full doc-strings

- As you change the preprocess names: also update test_utils, __all__ in utils and rootnode and smoothed_aggregation_solver, examples directory, 
  test_rootnode, test_smoothed_aggregation_solver, and just egrep in examples and pyamg
