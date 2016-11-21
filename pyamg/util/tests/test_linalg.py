from numpy import matrix, array, abs, ravel, zeros_like, dot
from scipy import rand, linalg, mat, random
from scipy.sparse import csr_matrix
from scipy.linalg import svd, eigvals

from pyamg.util.linalg import approximate_spectral_radius,\
    infinity_norm, norm, condest, cond,\
    ishermitian, pinv_array

from pyamg import gallery

from numpy.testing import TestCase, assert_almost_equal, assert_equal,\
    assert_array_almost_equal


class TestLinalg(TestCase):
    def test_norm(self):
        cases = []

        cases.append(4)
        cases.append(-1)
        cases.append(2.5)
        cases.append(3 + 5j)
        cases.append(7 - 2j)
        cases.append([1 + 3j, 6])
        cases.append([1 + 3j, 6 - 2j])

        for A in cases:
            assert_almost_equal(norm(A), linalg.norm(A))

    def test_approximate_spectral_radius(self):
        cases = []

        cases.append(matrix([[-4]]))
        cases.append(array([[-4]]))

        cases.append(array([[2, 0], [0, 1]]))
        cases.append(array([[-2, 0], [0, 1]]))

        cases.append(array([[100, 0, 0], [0, 101, 0], [0, 0, 99]]))

        for i in range(1, 5):
            cases.append(rand(i, i))

        # method should be almost exact for small matrices
        for A in cases:
            A = A.astype(float)
            Asp = csr_matrix(A)

            [E, V] = linalg.eig(A)
            E = abs(E)
            largest_eig = (E == E.max()).nonzero()[0]
            expected_eig = E[largest_eig]
            expected_vec = V[:, largest_eig]

            assert_almost_equal(approximate_spectral_radius(A), expected_eig)
            assert_almost_equal(approximate_spectral_radius(Asp), expected_eig)
            vec = approximate_spectral_radius(A, return_vector=True)[1]
            minnorm = min(norm(expected_vec + vec), norm(expected_vec - vec))
            diff = minnorm / norm(expected_vec)
            assert_almost_equal(diff, 0.0, decimal=4)
            vec = approximate_spectral_radius(Asp, return_vector=True)[1]
            minnorm = min(norm(expected_vec + vec), norm(expected_vec - vec))
            diff = minnorm / norm(expected_vec)
            assert_almost_equal(diff, 0.0, decimal=4)

        # try symmetric matrices
        for A in cases:
            A = A + A.transpose()
            A = A.astype(float)
            Asp = csr_matrix(A)

            [E, V] = linalg.eig(A)
            E = abs(E)
            largest_eig = (E == E.max()).nonzero()[0]
            expected_eig = E[largest_eig]
            expected_vec = V[:, largest_eig]

            assert_almost_equal(approximate_spectral_radius(A), expected_eig)
            assert_almost_equal(approximate_spectral_radius(Asp), expected_eig)
            vec = approximate_spectral_radius(A, return_vector=True)[1]
            minnorm = min(norm(expected_vec + vec), norm(expected_vec - vec))
            diff = minnorm / norm(expected_vec)
            assert_almost_equal(diff, 0.0, decimal=4)
            vec = approximate_spectral_radius(Asp, return_vector=True)[1]
            minnorm = min(norm(expected_vec + vec), norm(expected_vec - vec))
            diff = minnorm / norm(expected_vec)
            assert_almost_equal(diff, 0.0, decimal=4)

        # test a larger matrix, and various parameter choices
        cases = []
        A1 = gallery.poisson((50, 50), format='csr')
        cases.append((A1, 7.99241331495))
        A2 = gallery.elasticity.linear_elasticity((32, 32), format='bsr')[0]
        cases.append((A2, 536549.922189))
        for A, expected in cases:
            # test that increasing maxiter increases accuracy
            ans1 = approximate_spectral_radius(A, tol=1e-16, maxiter=5,
                                               restart=0)
            del A.rho
            ans2 = approximate_spectral_radius(A, tol=1e-16, maxiter=15,
                                               restart=0)
            del A.rho
            assert_equal(abs(ans2 - expected) < 0.5*abs(ans1 - expected), True)
            # test that increasing restart increases accuracy
            ans1 = approximate_spectral_radius(A, tol=1e-16, maxiter=10,
                                               restart=0)
            del A.rho
            ans2 = approximate_spectral_radius(A, tol=1e-16, maxiter=10,
                                               restart=1)
            del A.rho
            assert_equal(abs(ans2 - expected) < 0.8*abs(ans1 - expected), True)
            # test tol
            ans1 = approximate_spectral_radius(A, tol=0.1, maxiter=15,
                                               restart=5)
            del A.rho
            assert_equal(abs(ans1 - expected)/abs(expected) < 0.1, True)
            ans2 = approximate_spectral_radius(A, tol=0.001, maxiter=15,
                                               restart=5)
            del A.rho
            assert_equal(abs(ans2 - expected)/abs(expected) < 0.001, True)
            assert_equal(abs(ans2 - expected) < 0.1*abs(ans1 - expected), True)

    def test_infinity_norm(self):
        A = matrix([[-4]])
        assert_equal(infinity_norm(csr_matrix(A)), 4)

        A = matrix([[1, 0, -5], [-2, 5, 0]])
        assert_equal(infinity_norm(csr_matrix(A)), 7)

        A = matrix([[0, 1], [0, -5]])
        assert_equal(infinity_norm(csr_matrix(A)), 5)

        A = matrix([[1.3, -4.7, 0], [-2.23, 5.5, 0], [9, 0, -2]])
        assert_equal(infinity_norm(csr_matrix(A)), 11)


class TestComplexLinalg(TestCase):
    def test_approximate_spectral_radius(self):
        cases = []

        cases.append(matrix([[-4-4.0j]]))
        cases.append(matrix([[-4+8.2j]]))

        cases.append(matrix([[2.0-2.9j, 0], [0, 1.5]]))
        cases.append(matrix([[-2.0-2.4j, 0], [0, 1.21]]))

        cases.append(matrix([[100+1.0j, 0, 0],
                             [0, 101-1.0j, 0],
                             [0, 0, 99+9.9j]]))

        for i in range(1, 6):
            cases.append(matrix(rand(i, i)+1.0j*rand(i, i)))

        # method should be almost exact for small matrices
        for A in cases:
            Asp = csr_matrix(A)
            [E, V] = linalg.eig(A)
            E = abs(E)
            largest_eig = (E == E.max()).nonzero()[0]
            expected_eig = E[largest_eig]
            # expected_vec = V[:, largest_eig]

            assert_almost_equal(approximate_spectral_radius(A), expected_eig)
            assert_almost_equal(approximate_spectral_radius(Asp), expected_eig)
            vec = approximate_spectral_radius(A, return_vector=True)[1]
            Avec = A * vec
            Avec = ravel(Avec)
            vec = ravel(vec)
            rayleigh = abs(dot(Avec, vec) / dot(vec, vec))
            assert_almost_equal(rayleigh, expected_eig, decimal=4)
            vec = approximate_spectral_radius(Asp, return_vector=True)[1]
            Aspvec = Asp * vec
            Aspvec = ravel(Aspvec)
            vec = ravel(vec)
            rayleigh = abs(dot(Aspvec, vec) / dot(vec, vec))
            assert_almost_equal(rayleigh, expected_eig, decimal=4)

            AA = mat(A).H*mat(A)
            AAsp = csr_matrix(AA)
            [E, V] = linalg.eig(AA)
            E = abs(E)
            largest_eig = (E == E.max()).nonzero()[0]
            expected_eig = E[largest_eig]
            # expected_vec = V[:, largest_eig]

            assert_almost_equal(approximate_spectral_radius(AA),
                                expected_eig)
            assert_almost_equal(approximate_spectral_radius(AAsp),
                                expected_eig)
            vec = approximate_spectral_radius(AA, return_vector=True)[1]
            AAvec = AA * vec
            AAvec = ravel(AAvec)
            vec = ravel(vec)
            rayleigh = abs(dot(AAvec, vec) / dot(vec, vec))
            assert_almost_equal(rayleigh, expected_eig, decimal=4)
            vec = approximate_spectral_radius(AAsp, return_vector=True)[1]
            AAspvec = AAsp * vec
            AAspvec = ravel(AAspvec)
            vec = ravel(vec)
            rayleigh = abs(dot(AAspvec, vec) / dot(vec, vec))
            assert_almost_equal(rayleigh, expected_eig, decimal=4)

    def test_infinity_norm(self):
        A = matrix([[-4-3.0j]])
        assert_equal(infinity_norm(csr_matrix(A)), 5.0)

        A = matrix([[1, 0, 4.0-3.0j], [-2, 5, 0]])
        assert_equal(infinity_norm(csr_matrix(A)), 7)

        A = matrix([[0, 1], [0, -4.0+3.0j]])
        assert_equal(infinity_norm(csr_matrix(A)), 5.0)

    def test_cond(self):
        # make tests repeatable
        random.seed(0)

        # Should be exact
        cases = []
        A = mat(array([2.14]))
        cases.append(A)
        A = mat(array([2.14j]))
        cases.append(A)
        A = mat(array([-1.2 + 2.14j]))
        cases.append(A)
        for i in range(1, 6):
            A = mat(rand(i, i))
            cases.append(A)
            cases.append(1.0j*A)
            A = mat(A + 1.0j*rand(i, i))
            cases.append(A)

        for A in cases:
            U, Sigma, Vh = svd(A)
            exact = max(Sigma)/min(Sigma)
            c = cond(A)
            assert_almost_equal(exact, c)

    def test_condest(self):
        # make tests repeatable
        random.seed(0)

        # Should be exact for small matrices
        cases = []
        A = mat(array([2.14]))
        cases.append(A)
        A = mat(array([2.14j]))
        cases.append(A)
        A = mat(array([-1.2 + 2.14j]))
        cases.append(A)
        for i in range(1, 6):
            A = mat(rand(i, i))
            cases.append(A)
            cases.append(1.0j*A)
            A = mat(A + 1.0j*rand(i, i))
            cases.append(A)

        for A in cases:
            eigs = eigvals(A)
            exact = max(abs(eigs))/min(abs(eigs))
            c = condest(A)
            assert_almost_equal(exact, c)

    def test_ishermitian(self):
        # make tests repeatable
        random.seed(0)
        casesT = []
        casesF = []
        # 1x1
        casesT.append(mat(rand(1, 1)))
        casesF.append(mat(1.0j*rand(1, 1)))
        # 2x2
        A = array([[1.0, 0.0], [2.0, 1.0]])
        Ai = 1.0j*A
        casesF.append(A)
        casesF.append(Ai)
        A = A + Ai
        casesF.append(A)
        casesT.append(A + A.conjugate().T)
        # 3x3
        A = mat(rand(3, 3))
        Ai = 1.0j*rand(3, 3)
        casesF.append(A)
        casesF.append(Ai)
        A = A + Ai
        casesF.append(A)
        casesT.append(A + A.H)

        for A in casesT:
            # dense arrays
            assert_equal(ishermitian(A, fast_check=False), True)
            assert_equal(ishermitian(A, fast_check=True), True)

            # csr arrays
            A = csr_matrix(A)
            assert_equal(ishermitian(A, fast_check=False), True)
            assert_equal(ishermitian(A, fast_check=True), True)

        for A in casesF:
            # dense arrays
            assert_equal(ishermitian(A, fast_check=False), False)
            assert_equal(ishermitian(A, fast_check=True), False)

            # csr arrays
            A = csr_matrix(A)
            assert_equal(ishermitian(A, fast_check=False), False)
            assert_equal(ishermitian(A, fast_check=True), False)

    def test_pinv_array(self):
        from scipy.linalg import pinv2

        tests = []
        tests.append(rand(1, 1, 1))
        tests.append(rand(3, 1, 1))
        tests.append(rand(1, 2, 2))
        tests.append(rand(3, 2, 2))
        tests.append(rand(1, 3, 3))
        tests.append(rand(3, 3, 3))
        A = rand(1, 3, 3)
        A[0, 0, :] = A[0, 1, :]
        tests.append(A)

        tests.append(rand(1, 1, 1) + 1.0j*rand(1, 1, 1))
        tests.append(rand(3, 1, 1) + 1.0j*rand(3, 1, 1))
        tests.append(rand(1, 2, 2) + 1.0j*rand(1, 2, 2))
        tests.append(rand(3, 2, 2) + 1.0j*rand(3, 2, 2))
        tests.append(rand(1, 3, 3) + 1.0j*rand(1, 3, 3))
        tests.append(rand(3, 3, 3) + 1.0j*rand(3, 3, 3))
        A = rand(1, 3, 3) + 1.0j*rand(1, 3, 3)
        A[0, 0, :] = A[0, 1, :]
        tests.append(A)

        for test in tests:
            pinv_test = zeros_like(test)
            for i in range(pinv_test.shape[0]):
                pinv_test[i] = pinv2(test[i])

            pinv_array(test)
            assert_array_almost_equal(test, pinv_test, decimal=4)
