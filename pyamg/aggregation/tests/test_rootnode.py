import numpy as np
import scipy as sp
import scipy.sparse
from scipy.sparse import csr_matrix, SparseEfficiencyWarning

from pyamg.util.utils import diag_sparse
from pyamg.gallery import poisson, linear_elasticity, gauge_laplacian,\
    load_example

from pyamg.aggregation.rootnode import rootnode_solver

from numpy.testing import TestCase, assert_approx_equal,\
    assert_array_almost_equal

import warnings
warnings.simplefilter('ignore', SparseEfficiencyWarning)


class TestParameters(TestCase):
    def setUp(self):
        self.cases = []

        self.cases.append((poisson((100,), format='csr'), None))
        self.cases.append((poisson((10, 10), format='csr'), None))
        self.cases.append(linear_elasticity((7, 7), format='bsr'))

    def run_cases(self, opts):
        for A, B in self.cases:
            ml = rootnode_solver(A, B, max_coarse=5, **opts)

            np.random.seed(0)  # make tests repeatable

            x = sp.rand(A.shape[0])
            b = A * sp.rand(A.shape[0])

            residuals = []
            x_sol = ml.solve(b, x0=x, maxiter=30, tol=1e-10,
                             residuals=residuals)
            del x_sol
            convergence_ratio =\
                (residuals[-1] / residuals[0]) ** (1.0 / len(residuals))
            assert(convergence_ratio < 0.9)

    def test_strength_of_connection(self):
        for strength in ['symmetric', 'evolution']:
            self.run_cases({'strength': strength})

    def test_aggregation_method(self):
        for aggregate in ['standard', 'lloyd']:
            self.run_cases({'aggregate': aggregate})

    def test_prolongation_smoother(self):
        for smooth in ['energy']:
            self.run_cases({'smooth': smooth})

    def test_smoothers(self):
        smoothers = []
        smoothers.append('gauss_seidel')
        # smoothers.append( ('sor',{'omega':0.9}) )
        smoothers.append(('gauss_seidel', {'sweep': 'symmetric'}))

        for pre in smoothers:
            for post in smoothers:
                self.run_cases({'presmoother': pre, 'postsmoother': post})

    def test_coarse_solvers(self):
        solvers = []
        solvers.append('splu')
        solvers.append('lu')
        solvers.append('cg')
        solvers.append('gauss_seidel')
        solvers.append('block_gauss_seidel')
        solvers.append('gauss_seidel_nr')
        solvers.append('jacobi')

        for solver in solvers:
            self.run_cases({'coarse_solver': solver})

    def test_diagonal_dominance(self):
        diagonal_dominance = []
        diagonal_dominance.append((False, {'theta': 1.1}))
        diagonal_dominance.append((True, {'theta': 1.1}))
        diagonal_dominance.append(True)
        diagonal_dominance.append(False)

        for dd in diagonal_dominance:
            self.run_cases({'diagonal_dominance': dd})


class TestComplexParameters(TestCase):
    def setUp(self):
        self.cases = []

        # Consider "Helmholtz" like problems with an imaginary shift so that
        # the operator should still be SPD in a sense and SA should perform
        # well.  There are better near nullspace vectors than the default, but
        # a constant should give a convergent solver, nonetheless.
        A = poisson((100,), format='csr')
        A = A + 1.0j * scipy.sparse.eye(A.shape[0], A.shape[1])
        self.cases.append((A, None))
        A = poisson((10, 10), format='csr')
        A = A + 1.0j * scipy.sparse.eye(A.shape[0], A.shape[1])
        self.cases.append((A, None))

    def run_cases(self, opts):
        for A, B in self.cases:
            ml = rootnode_solver(A, B, max_coarse=5, **opts)

            np.random.seed(0)  # make tests repeatable

            x = sp.rand(A.shape[0]) + 1.0j * sp.rand(A.shape[0])
            b = A * sp.rand(A.shape[0])
            residuals = []

            x_sol = ml.solve(b, x0=x, maxiter=30, tol=1e-10,
                             residuals=residuals)
            del x_sol
            convergence_ratio =\
                (residuals[-1] / residuals[0]) ** (1.0 / len(residuals))
            assert(convergence_ratio < 0.9)

    def test_strength_of_connection(self):
        for strength in ['classical', 'symmetric']:
            self.run_cases({'strength': strength})

    def test_aggregation_method(self):
        for aggregate in ['standard', 'lloyd']:
            self.run_cases({'aggregate': aggregate})

    def test_prolongation_smoother(self):
        for smooth in [('energy', {'krylov': 'cgnr'}),
                       ('energy', {'krylov': 'gmres'})]:
            self.run_cases({'smooth': smooth})

    def test_smoothers(self):
        smoothers = []
        smoothers.append('gauss_seidel')
        smoothers.append(('gauss_seidel', {'sweep': 'symmetric'}))
        smoothers.append(('gauss_seidel_ne', {'sweep': 'symmetric'}))
        smoothers.append(('gauss_seidel_nr', {'sweep': 'symmetric'}))

        for pre in smoothers:
            for post in smoothers:
                self.run_cases({'presmoother': pre, 'postsmoother': post})

    def test_coarse_solvers(self):
        solvers = []
        solvers.append('splu')
        solvers.append('lu')
        solvers.append('cg')
        solvers.append('pinv2')

        for solver in solvers:
            self.run_cases({'coarse_solver': solver})

    def test_diagonal_dominance(self):
        diagonal_dominance = []
        diagonal_dominance.append((False, {'theta': 1.1}))
        diagonal_dominance.append((True, {'theta': 1.1}))
        diagonal_dominance.append(True)
        diagonal_dominance.append(False)

        for dd in diagonal_dominance:
            self.run_cases({'diagonal_dominance': dd})


class TestSolverPerformance(TestCase):
    def setUp(self):
        self.cases = []

        A = poisson((5000,), format='csr')
        self.cases.append((A, None, 0.4, 'symmetric',
                          ('energy', {'krylov': 'cg'})))
        self.cases.append((A, None, 0.4, 'symmetric',
                          ('energy', {'krylov': 'gmres'})))

        A = poisson((75, 75), format='csr')
        self.cases.append((A, None, 0.26, 'symmetric',
                          ('energy', {'krylov': 'cg'})))
        self.cases.append((A, None, 0.30, 'symmetric',
                          ('energy', {'krylov': 'cgnr'})))

        A, B = linear_elasticity((50, 50), format='bsr')
        self.cases.append((A, B, 0.3, 'symmetric',
                          ('energy', {'krylov': 'cg'})))
        self.cases.append((A, B, 0.3, 'symmetric',
                          ('energy', {'krylov': 'cgnr'})))
        self.cases.append((A, B, 0.3, 'symmetric',
                          ('energy', {'krylov': 'gmres'})))
        # TODO add unstructured tests

    def test_basic(self):
        """check that method converges at a reasonable rate"""

        for A, B, c_factor, symmetry, smooth in self.cases:
            ml = rootnode_solver(A, B, symmetry=symmetry, smooth=smooth,
                                 max_coarse=10)

            np.random.seed(0)  # make tests repeatable

            x = sp.rand(A.shape[0])
            b = A * sp.rand(A.shape[0])

            residuals = []
            x_sol = ml.solve(b, x0=x, maxiter=20, tol=1e-10,
                             residuals=residuals)
            del x_sol

            avg_convergence_ratio =\
                (residuals[-1] / residuals[0]) ** (1.0 / len(residuals))
            # print "Real Test:   %1.3e,  %1.3e,  %d,  %1.3e" % \
            #   (avg_convergence_ratio, c_factor, len(ml.levels),
            #    ml.operator_complexity())
            assert(avg_convergence_ratio < c_factor)

    def test_DAD(self):
        A = poisson((50, 50), format='csr')

        x = sp.rand(A.shape[0])
        b = sp.rand(A.shape[0])

        D = diag_sparse(1.0 / np.sqrt(10**(12 * sp.rand(A.shape[0]) - 6)))
        D = D.tocsr()
        D_inv = diag_sparse(1.0 / D.data)

        # DAD = D * A * D

        B = np.ones((A.shape[0], 1))

        # TODO force 2 level method and check that result is the same
        kwargs = {'max_coarse': 1, 'max_levels': 2, 'coarse_solver': 'splu'}

        sa = rootnode_solver(D * A * D, D_inv * B, **kwargs)

        residuals = []
        x_sol = sa.solve(b, x0=x, maxiter=10, tol=1e-12, residuals=residuals)
        del x_sol

        avg_convergence_ratio =\
            (residuals[-1] / residuals[0]) ** (1.0 / len(residuals))

        # print "Diagonal Scaling Test:   %1.3e,  %1.3e" %
        # (avg_convergence_ratio, 0.4)
        assert(avg_convergence_ratio < 0.4)

    def test_improve_candidates(self):
        # test improve_candidates for the Poisson problem and elasticity, where
        # rho_scale is the amount that each successive improve_candidates
        # option should improve convergence over the previous
        # improve_candidates option.
        improve_candidates_list = [None,
                                   [('block_gauss_seidel',
                                     {'iterations': 4, 'sweep': 'symmetric'})]]
        # make tests repeatable
        np.random.seed(0)

        cases = []
        A_elas, B_elas = linear_elasticity((60, 60), format='bsr')
        #                Matrix                       Candidates    rho_scale
        cases.append((poisson((75, 75), format='csr'),
                      np.ones((75 * 75, 1)), 0.9))
        cases.append((A_elas, B_elas, 0.9))
        for (A, B, rho_scale) in cases:
            last_rho = -1.0
            x0 = sp.rand(A.shape[0], 1)
            b = sp.rand(A.shape[0], 1)
            for improve_candidates in improve_candidates_list:
                ml = rootnode_solver(A, B, max_coarse=10,
                                     improve_candidates=improve_candidates)
                residuals = []
                x_sol = ml.solve(b, x0=x0, maxiter=20, tol=1e-10,
                                 residuals=residuals)
                del x_sol
                rho = (residuals[-1] / residuals[0]) ** (1.0 / len(residuals))
                if last_rho == -1.0:
                    last_rho = rho
                else:
                    # each successive improve_candidates option should be an
                    # improvement on the previous print "\nimprove_candidates
                    # Test: %1.3e, %1.3e,
                    # %d\n"%(rho,rho_scale*last_rho,A.shape[0])
                    assert(rho < rho_scale * last_rho)
                    last_rho = rho

    def test_symmetry(self):
        # Test that a basic V-cycle yields a symmetric linear operator.  Common
        # reasons for failure are problems with using the same rho for the
        # pres/post-smoothers and using the same block_D_inv for
        # pre/post-smoothers.

        n = 500
        A = poisson((n,), format='csr')
        smoothers = [('gauss_seidel', {'sweep': 'symmetric'}),
                     ('schwarz', {'sweep': 'symmetric'}),
                     ('block_gauss_seidel', {'sweep': 'symmetric'}),
                     'jacobi', 'block_jacobi']
        Bs = [np.ones((n, 1)),
              sp.hstack((np.ones((n, 1)),
                         np.arange(1, n + 1, dtype='float').reshape(-1, 1)))]

        for smoother in smoothers:
            for B in Bs:
                ml = rootnode_solver(A, B, max_coarse=10,
                                     presmoother=smoother,
                                     postsmoother=smoother)
                P = ml.aspreconditioner()
                x = sp.rand(n,)
                y = sp.rand(n,)
                assert_approx_equal(np.dot(P * x, y), np.dot(x, P * y))

    def test_nonsymmetric(self):
        # problem data
        data = load_example('recirc_flow')
        A = data['A'].tocsr()
        B = data['B']
        np.random.seed(625)
        x0 = sp.rand(A.shape[0])
        b = A * sp.rand(A.shape[0])
        # solver parameters
        smooth = ('energy', {'krylov': 'gmres'})
        SA_build_args = {'max_coarse': 25, 'coarse_solver': 'pinv2',
                         'symmetry': 'nonsymmetric'}
        SA_solve_args = {'cycle': 'V', 'maxiter': 20, 'tol': 1e-8}
        strength = [('evolution', {'k': 2, 'epsilon': 8.0})]
        smoother = ('gauss_seidel_nr', {'sweep': 'symmetric', 'iterations': 1})
        improve_candidates = [('gauss_seidel_nr', {'sweep': 'symmetric',
                                                   'iterations': 4}), None]
        # Construct solver with nonsymmetric parameters
        sa = rootnode_solver(A, B=B, smooth=smooth,
                             improve_candidates=improve_candidates,
                             strength=strength,
                             presmoother=smoother,
                             postsmoother=smoother, **SA_build_args)
        residuals = []
        # stand-alone solve
        x = sa.solve(b, x0=x0, residuals=residuals, **SA_solve_args)
        residuals = np.array(residuals)
        avg_convergence_ratio =\
            (residuals[-1] / residuals[0]) ** (1.0 / len(residuals))
        # print "Test 1  %1.3e,  %1.3e" % (avg_convergence_ratio, 0.7)
        assert(avg_convergence_ratio < 0.7)
        # accelerated solve
        residuals = []
        x = sa.solve(b, x0=x0, residuals=residuals, accel='gmres',
                     **SA_solve_args)
        del x
        residuals = np.array(residuals)
        avg_convergence_ratio =\
            (residuals[-1] / residuals[0]) ** (1.0 / len(residuals))
        # print "Test 2  %1.3e,  %1.3e" % (avg_convergence_ratio, 0.45)
        assert(avg_convergence_ratio < 0.45)

        # test that nonsymmetric parameters give the same result as symmetric
        # parameters for Poisson problem
        A = poisson((15, 15), format='csr')
        strength = 'symmetric'
        SA_build_args['symmetry'] = 'nonsymmetric'
        sa_nonsymm = rootnode_solver(A, B=np.ones((A.shape[0], 1)),
                                     smooth=smooth,
                                     strength=strength,
                                     presmoother=smoother,
                                     postsmoother=smoother,
                                     improve_candidates=None, **SA_build_args)
        SA_build_args['symmetry'] = 'symmetric'
        sa_symm = rootnode_solver(A, B=np.ones((A.shape[0], 1)), smooth=smooth,
                                  strength=strength, presmoother=smoother,
                                  postsmoother=smoother,
                                  improve_candidates=None, **SA_build_args)
        for (symm_lvl, nonsymm_lvl) in zip(sa_nonsymm.levels, sa_symm.levels):
            assert_array_almost_equal(symm_lvl.A.todense(),
                                      nonsymm_lvl.A.todense())

    def test_coarse_solver_opts(self):
        # these tests are meant to test whether coarse solvers are correctly
        # passed parameters

        A = poisson((30, 30), format='csr')
        b = sp.rand(A.shape[0], 1)

        # for each pair, the first entry should yield an SA solver that
        # converges in fewer iterations for a basic Poisson problem
        coarse_solver_pairs = [(('jacobi', {'iterations': 30}), 'jacobi')]
        coarse_solver_pairs.append((('gauss_seidel',
                                     {'iterations': 30}), 'gauss_seidel'))
        coarse_solver_pairs.append(('gauss_seidel', 'jacobi'))
        coarse_solver_pairs.append(('cg', ('cg', {'tol': 10.0})))
        coarse_solver_pairs.append(('pinv2', ('pinv2', {'cond': 1.0})))

        for coarse1, coarse2 in coarse_solver_pairs:
            r1 = []
            r2 = []
            sa1 = rootnode_solver(A, coarse_solver=coarse1, max_coarse=500)
            sa2 = rootnode_solver(A, coarse_solver=coarse2, max_coarse=500)
            x1 = sa1.solve(b, residuals=r1)
            x2 = sa2.solve(b, residuals=r2)
            del x1, x2
            assert((len(r1) + 5) < len(r2))

    def test_matrix_formats(self):

        # Do dense, csr, bsr and csc versions of A all yield the same solver
        A = poisson((7, 7), format='csr')
        cases = [A.tobsr(blocksize=(1, 1))]
        cases.append(A.tocsc())
        cases.append(A.todense())

        sa_old = rootnode_solver(A, max_coarse=10)
        for AA in cases:
            sa_new = rootnode_solver(AA, max_coarse=10)
            dff = sa_old.levels[-1].A.todense() - sa_new.levels[-1].A.todense()
            assert(np.abs(np.ravel(dff)).max() < 0.01)
            sa_old = sa_new


class TestComplexSolverPerformance(TestCase):
    ''' Imaginary tests from
        'Algebraic Multigrid Solvers for Complex-Valued Matrices",
            Maclachlan, Oosterlee,
         Vol. 30, SIAM J. Sci. Comp, 2008
    '''

    def setUp(self):
        self.cases = []

        # Test 1
        A = poisson((5000,), format='csr')
        Ai = A + 1.0j * scipy.sparse.eye(A.shape[0], A.shape[1])
        self.cases.append((Ai, None, 0.12, 'symmetric',
                           ('energy', {'krylov': 'gmres'})))

        # Test 2
        A = poisson((71, 71), format='csr')
        Ai = A + (0.625 / 0.01) * 1.0j *\
            scipy.sparse.eye(A.shape[0], A.shape[1])
        self.cases.append((Ai, None, 1e-3, 'symmetric',
                           ('energy', {'krylov': 'cgnr'})))

        # Test 3
        A = poisson((60, 60), format='csr')
        Ai = 1.0j * A
        self.cases.append((Ai, None, 0.35, 'symmetric',
                           ('energy', {'krylov': 'cgnr', 'maxiter': 8})))
        self.cases.append((Ai, None, 0.35, 'symmetric',
                           ('energy', {'krylov': 'gmres', 'maxiter': 8})))

        # Test 4 Use an "inherently" imaginary problem, the Gauge Laplacian in
        # 2D from Quantum Chromodynamics,
        A = gauge_laplacian(70, spacing=1.0, beta=0.41)
        self.cases.append((A, None, 0.25, 'hermitian',
                           ('energy', {'krylov': 'cg'})))

    def test_basic(self):
        """check that method converges at a reasonable rate"""

        for A, B, c_factor, symmetry, smooth in self.cases:
            A = csr_matrix(A)

            ml = rootnode_solver(A, B, symmetry=symmetry, smooth=smooth,
                                 max_coarse=10)

            np.random.seed(0)  # make tests repeatable

            x = sp.rand(A.shape[0]) + 1.0j * sp.rand(A.shape[0])
            b = A * sp.rand(A.shape[0])
            residuals = []

            x_sol = ml.solve(b, x0=x, maxiter=20, tol=1e-10,
                             residuals=residuals)
            del x_sol

            avg_convergence_ratio =\
                (residuals[-1] / residuals[0]) ** (1.0 / len(residuals))

            # print "Complex Test:   %1.3e,  %1.3e,  %d,  %1.3e" % \
            #    (avg_convergence_ratio, c_factor, len(ml.levels),
            #     ml.operator_complexity())
            assert(avg_convergence_ratio < c_factor)

    def test_nonhermitian(self):
        # problem data
        data = load_example('helmholtz_2D')
        A = data['A'].tocsr()
        B = data['B']
        np.random.seed(625)
        x0 = sp.rand(A.shape[0]) + 1.0j * sp.rand(A.shape[0])
        b = A * sp.rand(A.shape[0]) + 1.0j * (A * sp.rand(A.shape[0]))
        # solver parameters
        smooth = ('energy', {'krylov': 'gmres'})
        SA_build_args = {'max_coarse': 25, 'coarse_solver': 'pinv2',
                         'symmetry': 'symmetric'}
        SA_solve_args = {'cycle': 'V', 'maxiter': 20, 'tol': 1e-8}
        strength = [('evolution', {'k': 2, 'epsilon': 2.0})]
        smoother = ('gauss_seidel_nr', {'sweep': 'symmetric', 'iterations': 1})
        # Construct solver with nonsymmetric parameters
        sa = rootnode_solver(A, B=B, smooth=smooth,
                             strength=strength, presmoother=smoother,
                             postsmoother=smoother, **SA_build_args)
        residuals = []
        # stand-alone solve
        x = sa.solve(b, x0=x0, residuals=residuals, **SA_solve_args)
        residuals = np.array(residuals)
        avg_convergence_ratio =\
            (residuals[-1] / residuals[0]) ** (1.0 / len(residuals))
        # print "Test 3  %1.3e,  %1.3e" % (avg_convergence_ratio, 0.92)
        assert(avg_convergence_ratio < 0.92)
        # accelerated solve
        residuals = []
        x = sa.solve(b, x0=x0, residuals=residuals, accel='gmres',
                     **SA_solve_args)
        del x
        residuals = np.array(residuals)
        avg_convergence_ratio =\
            (residuals[-1] / residuals[0]) ** (1.0 / len(residuals))
        # print "Test 4  %1.3e,  %1.3e" % (avg_convergence_ratio, 0.8)
        assert(avg_convergence_ratio < 0.8)

        # test that nonsymmetric parameters give the same result as symmetric
        # parameters for the complex-symmetric matrix A
        strength = 'symmetric'
        SA_build_args['symmetry'] = 'nonsymmetric'
        sa_nonsymm = rootnode_solver(A, B=np.ones((A.shape[0], 1)),
                                     smooth=smooth,
                                     strength=strength, presmoother=smoother,
                                     postsmoother=smoother,
                                     improve_candidates=None, **SA_build_args)
        SA_build_args['symmetry'] = 'symmetric'
        sa_symm = rootnode_solver(A, B=np.ones((A.shape[0], 1)),
                                  smooth=smooth,
                                  strength=strength, presmoother=smoother,
                                  postsmoother=smoother,
                                  improve_candidates=None, **SA_build_args)
        for (symm_lvl, nonsymm_lvl) in zip(sa_nonsymm.levels, sa_symm.levels):
            assert_array_almost_equal(symm_lvl.A.todense(),
                                      nonsymm_lvl.A.todense())
