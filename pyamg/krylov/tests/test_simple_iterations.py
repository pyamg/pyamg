from pyamg.krylov import minimal_residual, steepest_descent
import numpy as np
from pyamg.util.linalg import norm
import pyamg

from numpy.testing import TestCase


class TestSimpleIterations(TestCase):
    def setUp(self):
        self.definite_cases = []
        self.spd_cases = []

        # 1x1
        A = np.mat([[1.2]])
        b = np.array([3.9]).reshape(-1, 1)
        x0 = np.zeros((1, 1))
        self.definite_cases.append({'A': A, 'b': b, 'x0': x0, 'maxiter': 1,
                                    'reduction_factor': 1e-10})
        self.spd_cases.append({'A': A, 'b': b, 'x0': x0, 'maxiter': 1,
                               'reduction_factor': 1e-10})

        # 2x2
        A = np.mat([[4.5, -1.2],
                   [3.4, 6.7]])
        b = np.array([-3.2, 5.6]).reshape(-1, 1)
        x0 = np.zeros((2, 1))
        self.definite_cases.append({'A': A, 'b': b, 'x0': x0, 'maxiter': 4,
                                    'reduction_factor': 0.1})
        self.spd_cases.append({'A': A.T*A, 'b': b, 'x0': x0, 'maxiter': 4,
                               'reduction_factor': 0.1})

        # 2x2 Imaginary
        A = np.mat([[4.5, -1.2j],
                    [3.4+1.2j, 6.7]])
        b = np.array([-3.2, 5.6]).reshape(-1, 1)
        x0 = np.zeros((2, 1))
        self.spd_cases.append({'A': A.T*A, 'b': b, 'x0': x0, 'maxiter': 4,
                               'reduction_factor': 0.1})

        # 4x4
        A = np.mat([[1.2, 0., 0., 0.],
                    [0., 10., 2., 6.],
                    [0., 0., 9.3, -2.31],
                    [-4., 0., 0., 11.]])
        b = np.array([1., 3.9, 0., -1.23]).reshape(-1, 1)
        x0 = np.zeros((4, 1))
        self.definite_cases.append({'A': A, 'b': b, 'x0': x0, 'maxiter': 4,
                                    'reduction_factor': 0.3})
        self.spd_cases.append({'A': A.T*A, 'b': b, 'x0': x0, 'maxiter': 4,
                               'reduction_factor': 0.3})

        # 4x4 Imaginary
        A = np.mat(A, dtype=complex)
        A[0, 0] += 3.1j
        A[3, 3] -= 1.34j
        A[1, 3] *= 1.0j
        A[1, 2] += 1.0j
        b = np.array([1. - 1.0j, 2.0 - 3.9j, 0., -1.23]).reshape(-1, 1)
        x0 = np.ones((4, 1))
        self.spd_cases.append({'A': A.H*A, 'b': b, 'x0': x0, 'maxiter': 4,
                               'reduction_factor': 0.3})

        # 10x10
        A = np.mat([[91.1, 0., 0., 0., 3.9, 0., 0., 11., -1., 0.],
                    [0., 45., 2.9, 0., 0., 6.8, 0., 0., 0., 0.],
                    [0., 0., 19.0, 0., 0., 0.8, 1., -2.2, 0., 9.],
                    [-4., 0., 0.0, 4., 0., 0.0, 2., 2.2, 0., 0.],
                    [0., 0., 0.0, 21., 30., 0.1, 0., 0., 0., 0.],
                    [0., 0., 0.0, 0., -4.7, 7.0, 0., 0., 0., 0.],
                    [2.1, 7., 2.0, 0., 0., 0.0, 8., 0., 0., 0.],
                    [0., 0., 0.0, 34., 0., 0.0, 0., 87.1, -12.3, 0.],
                    [0., 3.4, 0.0, 0., 0., -0.3, 0., 0., 7., 0.],
                    [9., 0., 0.0, 0., 8.7, 0.0, 0., 0., 0., 11.2]])
        b = np.array([1., 0., 0.2, 8., 0., -1.9,
                      11.3, 0.0, 0.1, 0.0]).reshape(-1, 1)
        x0 = np.zeros((10, 1))
        x0[4] = 11.1
        x0[7] = -2.1
        self.definite_cases.append({'A': A, 'b': b, 'x0': x0, 'maxiter': 2,
                                    'reduction_factor': 0.98})
        self.spd_cases.append({'A':
                               np.mat(pyamg.gallery.poisson((10,)).todense()),
                               'b': b, 'x0': x0, 'maxiter': 2,
                               'reduction_factor': 0.5})

    def test_steepest_descent(self):
        # Ensure repeatability
        np.random.seed(0)

        for case in self.spd_cases:
            A = case['A']
            b = case['b']
            x0 = case['x0']
            maxiter = case['maxiter']
            reduction_factor = case['reduction_factor']

            # This function should always decrease
            fvals = []

            def callback(x):
                fvals.append(0.5*np.dot(np.ravel(x),
                                        np.ravel(A*x.reshape(-1, 1))) -
                             np.dot(np.ravel(b), np.ravel(x)))

            (x, flag) = steepest_descent(A, b, x0=x0, tol=1e-16,
                                         maxiter=maxiter, callback=callback)
            actual_factor = (norm(np.ravel(b) - np.ravel(A*x.reshape(-1, 1))) /
                             norm(np.ravel(b) - np.ravel(A*x0.reshape(-1, 1))))
            assert(actual_factor < reduction_factor)

            if A.dtype != complex:
                for i in range(len(fvals)-1):
                    assert(fvals[i+1] <= fvals[i])

        # Test preconditioning
        A = pyamg.gallery.poisson((10, 10), format='csr')
        b = np.random.rand(A.shape[0], 1)
        x0 = np.random.rand(A.shape[0], 1)
        fvals = []

        def callback(x):
            fvals.append(0.5*np.dot(np.ravel(x),
                                    np.ravel(A*x.reshape(-1, 1))) -
                         np.dot(np.ravel(b), np.ravel(x)))

        resvec = []
        sa = pyamg.smoothed_aggregation_solver(A)
        (x, flag) = steepest_descent(A, b, x0, tol=1e-8, maxiter=20,
                                     residuals=resvec, M=sa.aspreconditioner(),
                                     callback=callback)
        assert(resvec[-1]/resvec[0] < 1e-8)
        for i in range(len(fvals)-1):
            assert(fvals[i+1] <= fvals[i])

    def test_minimal_residual(self):
        # Ensure repeatability
        np.random.seed(0)

        self.definite_cases.extend(self.spd_cases)

        for case in self.definite_cases:
            A = case['A']
            maxiter = case['maxiter']
            x0 = np.random.rand(A.shape[0],)
            b = np.zeros_like(x0)
            reduction_factor = case['reduction_factor']
            if A.dtype != complex:

                # This function should always decrease (assuming zero RHS)
                fvals = []

                def callback(x):
                    fvals.append(np.sqrt(np.dot(np.ravel(x),
                                 np.ravel(A*x.reshape(-1, 1)))))
                #
                (x, flag) = minimal_residual(A, b, x0=x0,
                                             tol=1e-16, maxiter=maxiter,
                                             callback=callback)
                actual_factor = (norm(np.ravel(b) -
                                 np.ravel(A * x.reshape(-1, 1))) /
                                 norm(np.ravel(b) -
                                 np.ravel(A * x0.reshape(-1, 1))))
                assert(actual_factor < reduction_factor)
                if A.dtype != complex:
                    for i in range(len(fvals)-1):
                        assert(fvals[i+1] <= fvals[i])

        # Test preconditioning
        A = pyamg.gallery.poisson((10, 10), format='csr')
        x0 = np.random.rand(A.shape[0], 1)
        b = np.zeros_like(x0)
        fvals = []

        def callback(x):
            fvals.append(np.sqrt(np.dot(np.ravel(x),
                                        np.ravel(A*x.reshape(-1, 1)))))
        #
        resvec = []
        sa = pyamg.smoothed_aggregation_solver(A)
        (x, flag) = minimal_residual(A, b, x0, tol=1e-8, maxiter=20,
                                     residuals=resvec, M=sa.aspreconditioner(),
                                     callback=callback)
        assert(resvec[-1]/resvec[0] < 1e-8)
        for i in range(len(fvals)-1):
            assert(fvals[i+1] <= fvals[i])
