from pyamg.krylov import bicgstab, cg, cgne, cgnr, cr, fgmres, gmres
from pyamg.krylov._gmres_householder import gmres_householder
from pyamg.krylov._gmres_mgs import gmres_mgs
from numpy import array, zeros, ones
from scipy import mat, random
from scipy.linalg import solve
from pyamg.util.linalg import norm
import pyamg

from numpy.testing import TestCase, assert_array_almost_equal, assert_equal


class TestKrylov(TestCase):
    def setUp(self):
        self.cases = []
        self.spd_cases = []
        self.symm_cases = []

        # self.oblique = [gmres, fgmres, cgnr,
        #                 krylov._gmres_householder.gmres_householder,
        #                 krylov._gmres_mgs.gmres_mgs]
        self.oblique = [gmres_householder, gmres_mgs, gmres, fgmres, cgnr]
        self.symm_oblique = [cr]
        self.orth = [cgne]
        self.inexact = [bicgstab]
        self.spd_orth = [cg]

        # 1x1
        A = mat([[1.2]])
        b = array([3.9]).reshape(-1, 1)
        x0 = zeros((1, 1))
        self.cases.append({'A': A, 'b': b, 'x0': x0, 'tol': 1e-16,
                           'maxiter': 1, 'reduction_factor': 1e-10})
        self.spd_cases.append({'A': A, 'b': b, 'x0': x0, 'tol': 1e-16,
                               'maxiter': 1, 'reduction_factor': 1e-10})
        self.symm_cases.append({'A': A, 'b': b, 'x0': x0, 'tol': 1e-16,
                                'maxiter': 1, 'reduction_factor': 1e-10})

        # 4x4
        A = mat([[1.2, 0., 0., 0.],
                 [0., 4., 2., 6.],
                 [0., 0., 9.3, -2.31],
                 [-4., 0., 0., -11.]])
        b = array([1., 3.9, 0., -1.23]).reshape(-1, 1)
        x0 = zeros((4, 1))
        self.cases.append({'A': A, 'b': b, 'x0': x0, 'tol': 1e-16,
                           'maxiter': 4, 'reduction_factor': 1e-10})
        self.spd_cases.append({'A': A.T*A, 'b': b, 'x0': x0, 'tol': 1e-16,
                               'maxiter': 4, 'reduction_factor': 1e-10})
        self.symm_cases.append({'A': A.T + A, 'b': b, 'x0': x0, 'tol': 1e-16,
                                'maxiter': 4, 'reduction_factor': 1e-10})

        # 4x4 Imaginary
        A = mat(A, dtype=complex)
        A[0, 0] += 3.1j
        A[3, 3] -= 1.34j
        A[1, 3] *= 1.0j
        A[1, 2] += 1.0j
        b = array([1. - 1.0j, 2.0 - 3.9j, 0., -1.23]).reshape(-1, 1)
        x0 = ones((4, 1))
        self.cases.append({'A': A, 'b': b, 'x0': x0, 'tol': 1e-16,
                           'maxiter': 4, 'reduction_factor': 1e-10})
        self.spd_cases.append({'A': A.H*A, 'b': b, 'x0': x0, 'tol': 1e-16,
                               'maxiter': 4, 'reduction_factor': 1e-10})
        self.symm_cases.append({'A': A.H + A, 'b': b, 'x0': x0, 'tol': 1e-16,
                                'maxiter': 4, 'reduction_factor': 1e-10})

        # 10x10
        A = mat([[-1.1, 0., 0., 0., 3.9, 0., 0., 11., -1., 0.],
                 [0., 4., 2.9, 0., 0., 6.8, 0., 0., 0., 0.],
                 [0., 0., 9.0, 0., 0., 0.8, 1., -2.2, 0., 9.],
                 [-4., 0., 0.0, 0., 0., 0.0, 2., 2.2, 0., 0.],
                 [0., 0., 0.0, 21., 0., 0.1, 0., 0., 0., 0.],
                 [0., 0., 0.0, 0., -4.7, 0.0, 0., 0., 0., 0.],
                 [2.1, 7., 22.0, 0., 0., 0.0, 0., 0., 0., 0.],
                 [0., 0., 0.0, 34., 0., 0.0, 0., 0., -12.3, 0.],
                 [0., 3.4, 0.0, 0., 0., -0.3, 0., 0., 0., 0.],
                 [9., 0., 0.0, 0., 87., 0.0, 0., 0., 0., -11.2]])
        b = array([1., 0., 0.2, 8., 0., -1.9,
                   11.3, 0.0, 0.1, 0.0]).reshape(-1, 1)
        x0 = zeros((10, 1))
        x0[4] = 11.1
        x0[7] = -2.
        self.cases.append({'A': A, 'b': b, 'x0': x0, 'tol': 1e-16,
                           'maxiter': 2, 'reduction_factor': 0.98})
        self.symm_cases.append({'A': A + A.T, 'b': b, 'x0': x0, 'tol': 1e-16,
                                'maxiter': 2, 'reduction_factor': 0.98})
        self.spd_cases.append({'A':
                               mat(pyamg.gallery.poisson((10,)).todense()),
                               'b': b, 'x0': x0, 'tol': 1e-16, 'maxiter': 2,
                               'reduction_factor': 0.98})

    def test_gmres(self):
        # Ensure repeatability
        random.seed(0)

        #  For these small matrices, Householder and MGS GMRES should give the
        #  same result, and for symmetric (but possibly indefinite) matrices CR
        #  and GMRES should give same result
        for maxiter in [1, 2, 3]:
            for case, symm_case in zip(self.cases, self.symm_cases):
                A = case['A']
                b = case['b']
                x0 = case['x0']
                A_symm = symm_case['A']
                b_symm = symm_case['b']
                x0_symm = symm_case['x0']

                # Test agreement between Householder and GMRES
                (x, flag) = gmres_householder(A, b, x0=x0,
                                              maxiter=min(A.shape[0], maxiter))
                (x2, flag2) = gmres_mgs(A, b, x0=x0, maxiter=min(A.shape[0],
                                        maxiter))
                err_msg = ('Householder GMRES and MGS GMRES gave '
                           'different results for small matrix')
                assert_array_almost_equal(x/norm(x), x2/norm(x2),
                                          err_msg=err_msg)

                err_msg = ('Householder GMRES and MGS GMRES returned '
                           'different convergence flags for small matrix')
                assert_equal(flag, flag2, err_msg=err_msg)

                # Test agreement between GMRES and CR
                if A_symm.shape[0] > 1:
                    residuals2 = []
                    (x2, flag2) = gmres_mgs(A_symm, b_symm, x0=x0_symm,
                                            maxiter=min(A.shape[0], maxiter),
                                            residuals=residuals2)
                    residuals3 = []
                    (x3, flag2) = cr(A_symm, b_symm, x0=x0_symm,
                                     maxiter=min(A.shape[0], maxiter),
                                     residuals=residuals3)
                    residuals2 = array(residuals2)
                    residuals3 = array(residuals3)

                    err_msg = 'CR and GMRES yield different residual vectors'
                    assert_array_almost_equal(residuals3/norm(residuals3),
                                              residuals2/norm(residuals2),
                                              err_msg=err_msg)

                    err_msg = 'CR and GMRES yield different answers'
                    assert_array_almost_equal(x2/norm(x2), x3/norm(x3),
                                              err_msg=err_msg)

    def test_krylov(self):
        # Oblique projectors reduce the residual
        for method in self.oblique:
            for case in self.cases:
                A = case['A']
                b = case['b']
                x0 = case['x0']
                (xNew, flag) = method(A, b, x0=x0, tol=case['tol'],
                                      maxiter=case['maxiter'])
                xNew = xNew.reshape(-1, 1)
                assert_equal((norm(b - A*xNew)/norm(b - A*x0)) <
                             case['reduction_factor'], True,
                             err_msg='Oblique Krylov Method Failed Test')

        # Oblique projectors reduce the residual, here we consider oblique
        # projectors for symmetric matrices
        for method in self.symm_oblique:
            for case in self.symm_cases:
                A = case['A']
                b = case['b']
                x0 = case['x0']
                (xNew, flag) = method(A, b, x0=x0, tol=case['tol'],
                                      maxiter=case['maxiter'])
                xNew = xNew.reshape(-1, 1)
                assert_equal((norm(b - A*xNew)/norm(b - A*x0)) <
                             case['reduction_factor'], True,
                             err_msg='Symmetric oblique Krylov Method Failed')

        # Orthogonal projectors reduce the error
        for method in self.orth:
            for case in self.cases:
                A = case['A']
                b = case['b']
                x0 = case['x0']
                (xNew, flag) = method(A, b, x0=x0, tol=case['tol'],
                                      maxiter=case['maxiter'])
                xNew = xNew.reshape(-1, 1)
                soln = solve(A, b)
                assert_equal((norm(soln - xNew)/norm(soln - x0)) <
                             case['reduction_factor'], True,
                             err_msg='Orthogonal Krylov Method Failed Test')

        # SPD Orthogonal projectors reduce the error
        for method in self.spd_orth:
            for case in self.spd_cases:
                A = case['A']
                b = case['b']
                x0 = case['x0']
                (xNew, flag) = method(A, b, x0=x0, tol=case['tol'],
                                      maxiter=case['maxiter'])
                xNew = xNew.reshape(-1, 1)
                soln = solve(A, b)
                assert_equal((norm(soln - xNew)/norm(soln - x0)) <
                             case['reduction_factor'], True,
                             err_msg='Orthogonal Krylov Method Failed Test')

        # Assume that Inexact Methods reduce the residual for these examples
        for method in self.inexact:
            for case in self.cases:
                A = case['A']
                b = case['b']
                x0 = case['x0']
                (xNew, flag) = method(A, b, x0=x0, tol=case['tol'],
                                      maxiter=A.shape[0])
                xNew = xNew.reshape(-1, 1)
                assert_equal((norm(b - A*xNew)/norm(b - A*x0)) < 0.35, True,
                             err_msg='Inexact Krylov Method Failed Test')
