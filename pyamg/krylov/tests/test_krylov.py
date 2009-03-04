from pyamg.testing import *
from pyamg.krylov import *
from numpy import array, zeros, ones
from scipy import mat
from scipy.linalg import solve
from pyamg.util.linalg import norm
import pyamg


class TestKrylov(TestCase):
    def setUp(self):
        self.cases=[]
        self.spd_cases=[]

        self.oblique = [gmres, fgmres, cgnr]
        self.orth = [cgne]
        self.inexact = [bicgstab]
        self.spd_orth = [cg]
        
        # 1x1 
        A = mat([[  1.2]])
        b = array([3.9]).reshape(-1,1)
        x0 = zeros((1,1))
        self.cases.append({'A' : A, 'b' : b, 'x0' : x0, 'tol' : 1e-16, 'maxiter' : 1, 'reduction_factor' : 1e-10})
        self.spd_cases.append({'A' : A, 'b' : b, 'x0' : x0, 'tol' : 1e-16, 'maxiter' : 1, 'reduction_factor' : 1e-10})


        # 4x4 
        A = mat([[  1.2,    0.,   0.,     0.],
                 [  0.,     4.,   2.,     6.],
                 [  0.,     0.,   9.3,  -2.31],
                 [ -4.,     0.,   0.,    -11.]])
        b = array([1., 3.9, 0., -1.23]).reshape(-1,1)
        x0 = zeros((4,1))
        self.cases.append({'A' : A, 'b' : b, 'x0' : x0, 'tol' : 1e-16, 'maxiter' : 4, 'reduction_factor' : 1e-10})
        self.spd_cases.append({'A' : A.T*A, 'b' : b, 'x0' : x0, 'tol' : 1e-16, 'maxiter' : 4, 'reduction_factor' : 1e-10})
        
        # 4x4 Imaginary
        A = mat(A, dtype=complex)
        A[0,0] += 3.1j
        A[3,3] -= 1.34j
        A[1,3] *= 1.0j
        A[1,2] += 1.0j
        b = array([1. - 1.0j, 2.0 - 3.9j, 0., -1.23]).reshape(-1,1)
        x0 = ones((4,1))
        self.cases.append({'A' : A, 'b' : b, 'x0' : x0, 'tol' : 1e-16, 'maxiter' : 4, 'reduction_factor' : 1e-10})
        self.spd_cases.append({'A' : A.H*A, 'b' : b, 'x0' : x0, 'tol' : 1e-16, 'maxiter' : 4, 'reduction_factor' : 1e-10})

        # 10x10 
        A = mat([[-1.1,    0.,   0.,   0.,  3.9,  0.,   0.,  11.,  -1.,  0.],
                 [  0.,    4.,   2.9,  0.,   0.,  6.8,  0.,  0.,    0.,  0.],
                 [  0.,    0.,   9.0,  0.,   0.,  0.8,  1., -2.2,   0.,  9.],
                 [ -4.,    0.,   0.0,  0.,   0.,  0.0,  2.,  2.2,   0.,  0.],
                 [  0.,    0.,   0.0, 21.,   0.,  0.1,  0.,   0.,   0.,  0.],
                 [  0.,    0.,   0.0,  0.,  -4.7, 0.0,  0.,   0.,   0.,  0.],
                 [  2.1,   7.,  22.0,  0.,   0.,  0.0,  0.,   0.,   0.,  0.],
                 [  0.,    0.,   0.0, 34.,   0.,  0.0,  0.,   0.,  -12.3,0.],
                 [  0.,   3.4,   0.0,  0.,   0., -0.3,  0.,   0.,   0.,  0.],
                 [  9.,    0.,   0.0,  0.,  87.,  0.0,  0.,   0.,   0.,-11.2]])
        b = array([1., 0., 0.2, 8., 0., -1.9, 11.3, 0.0, 0.1, 0.0]).reshape(-1,1)
        x0 = zeros((10,1))
        x0[4] = 11.1; x0[7] = -2.1
        self.cases.append({'A' : A, 'b' : b, 'x0' : x0, 'tol' : 1e-16, 'maxiter' : 2, 'reduction_factor' : 0.98})
        self.spd_cases.append({'A' : mat(pyamg.gallery.poisson((10,)).todense()), 'b' : b, 'x0' : x0, 'tol' : 1e-16, 'maxiter' : 2, 'reduction_factor' : 0.98})


    def test_krylov(self):

        # Oblique projectors reduce the residual
        for method in self.oblique:
            for case in self.cases:
                A = case['A']; b = case['b']; x0 = case['x0']
                (xNew, flag) = method(A, b, x0=x0, tol=case['tol'], maxiter=case['maxiter'])
                xNew = xNew.reshape(-1,1)
                assert_equal( (norm(b - A*xNew)/norm(b - A*x0)) < case['reduction_factor'], True, err_msg='Oblique Krylov Method Failed Test')
    
        # Orthogonal projectors reduce the error
        for method in self.orth:
            for case in self.cases:
                A = case['A']; b = case['b']; x0 = case['x0']
                (xNew, flag) = method(A, b, x0=x0, tol=case['tol'], maxiter=case['maxiter'])
                xNew = xNew.reshape(-1,1)
                soln = solve(A,b)
                assert_equal( (norm(soln - xNew)/norm(soln - x0)) < case['reduction_factor'], True, err_msg='Orthogonal Krylov Method Failed Test')
    
        # SPD Orthogonal projectors reduce the error
        for method in self.spd_orth:
            for case in self.spd_cases:
                A = case['A']; b = case['b']; x0 = case['x0']
                (xNew, flag) = method(A, b, x0=x0, tol=case['tol'], maxiter=case['maxiter'])
                xNew = xNew.reshape(-1,1)
                soln = solve(A,b)
                assert_equal( (norm(soln - xNew)/norm(soln - x0)) < case['reduction_factor'], True, err_msg='Orthogonal Krylov Method Failed Test')

        # Assume that Inexact Methods reduce the residual for these examples
        for method in self.inexact:
            for case in self.cases:
                A = case['A']; b = case['b']; x0 = case['x0']
                (xNew, flag) = method(A, b, x0=x0, tol=case['tol'], maxiter=A.shape[0])
                xNew = xNew.reshape(-1,1)
                assert_equal( (norm(b - A*xNew)/norm(b - A*x0)) < 0.15, True, err_msg='Inexact Krylov Method Failed Test')
    


