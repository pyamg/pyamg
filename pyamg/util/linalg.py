"""Linear Algebra Helper Routines."""


from warnings import warn
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg import lapack, get_blas_funcs, eig, svd

from .params import set_tol


def norm(x, pnorm='2'):
    """2-norm of a vector.

    Parameters
    ----------
    x : array_like
        Vector of complex or real values

    pnorm : string
        '2' calculates the 2-norm
        'inf' calculates the infinity-norm

    Returns
    -------
    n : float
        2-norm of a vector

    Notes
    -----
    - currently 1+ order of magnitude faster than scipy.linalg.norm(x), which
      calls sqrt(numpy.sum(real((conjugate(x)*x)),axis=0)) resulting in an
      extra copy
    - only handles the 2-norm and infinity-norm for vectors

    See Also
    --------
    scipy.linalg.norm : scipy general matrix or vector norm

    """
    x = np.ravel(x)

    if pnorm == '2':
        return np.sqrt(np.inner(x.conj(), x).real)

    if pnorm == 'inf':
        return np.max(np.abs(x))

    raise ValueError('Only the 2-norm and infinity-norm are supported')


def infinity_norm(A):
    """Infinity norm of a matrix (maximum absolute row sum).

    Parameters
    ----------
    A : csr_matrix, csc_matrix, sparse, or numpy matrix
        Sparse or dense matrix

    Returns
    -------
    n : float
        Infinity norm of the matrix

    Notes
    -----
    - This serves as an upper bound on spectral radius.
    - csr and csc avoid a deep copy
    - dense calls scipy.linalg.norm

    See Also
    --------
    scipy.linalg.norm : dense matrix norms

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import spdiags
    >>> from pyamg.util.linalg import infinity_norm
    >>> n=10
    >>> e = np.ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n)
    >>> print(infinity_norm(A))
    4.0

    """
    if sparse.isspmatrix_csr(A) or sparse.isspmatrix_csc(A):
        # avoid copying index and ptr arrays
        abs_A = A.__class__((np.abs(A.data), A.indices, A.indptr),
                            shape=A.shape)
        return (abs_A * np.ones((A.shape[1]), dtype=A.dtype)).max()

    if sparse.isspmatrix(A):
        return (abs(A) * np.ones((A.shape[1]), dtype=A.dtype)).max()

    return np.dot(np.abs(A), np.ones((A.shape[1],), dtype=A.dtype)).max()


def axpy(x, y, a=1.0):
    """Quick level-1 call to BLAS y = a*x+y.

    Parameters
    ----------
    x : array_like
        nx1 real or complex vector
    y : array_like
        nx1 real or complex vector
    a : float
        real or complex scalar

    Returns
    -------
    y : array_like
        Input variable y is rewritten

    Notes
    -----
    The call to get_blas_funcs automatically determines the prefix for the blas
    call.

    """
    fn = get_blas_funcs(['axpy'], [x, y])[0]
    fn(x, y, a)

# def approximate_spectral_radius(A, tol=0.1, maxiter=10, symmetric=False):
#    """approximate the spectral radius of a matrix
#
#    Parameters
#    ----------
#
#    A : {dense or sparse matrix}
#        E.g. csr_matrix, csc_matrix, ndarray, etc.
#    tol : {scalar}
#        Tolerance of approximation
#    maxiter : {integer}
#        Maximum number of iterations to perform
#    symmetric : {boolean}
#        True if A is symmetric, False otherwise (default)
#
#    Returns
#    -------
#        An approximation to the spectral radius of A
#
#    """
#    if symmetric:
#        method = eigen_symmetric
#    else:
#        method = eigen
#
#    return norm( method(A, k=1, tol=0.1, which='LM', maxiter=maxiter,
#    return_eigenvectors=False) )


def _approximate_eigenvalues(A, maxiter, symmetric=None, initial_guess=None):
    """Apprixmate eigenvalues.

    Used by approximate_spectral_radius and condest.

    Returns [W, E, H, V, breakdown_flag], where W and E are the eigenvectors
    and eigenvalues of the Hessenberg matrix H, respectively, and V is the
    Krylov space.  breakdown_flag denotes whether Lanczos/Arnoldi suffered
    breakdown.  E is therefore the approximate eigenvalues of A.

    To obtain approximate eigenvectors of A, compute V*W.

    """
    A = aslinearoperator(A)  # A could be dense or sparse, or something weird

    # Choose tolerance for deciding if break-down has occurred
    breakdown = set_tol(A.dtype)
    breakdown_flag = False

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    maxiter = min(A.shape[0], maxiter)

    if initial_guess is None:
        v0 = np.random.rand(A.shape[1], 1)
        if A.dtype == complex:
            v0 = v0 + 1.0j * np.random.rand(A.shape[1], 1)
    else:
        v0 = initial_guess

    v0 /= norm(v0)

    # Important to type H based on v0, so that a real nonsymmetric matrix, can
    # have an imaginary initial guess for its Arnoldi Krylov space
    H = np.zeros((maxiter+1, maxiter),
                 dtype=np.find_common_type([v0.dtype, A.dtype], []))

    V = [v0]

    beta = 0.0
    for j in range(maxiter):
        w = A * V[-1]

        if symmetric:
            if j >= 1:
                H[j-1, j] = beta
                w -= beta * V[-2]

            alpha = np.dot(np.conjugate(w.ravel()), V[-1].ravel())
            H[j, j] = alpha
            w -= alpha * V[-1]  # axpy(V[-1],w,-alpha)

            beta = norm(w)
            H[j+1, j] = beta

            if (H[j+1, j] < breakdown):
                breakdown_flag = True
                break

            w /= beta

            V.append(w)
            V = V[-2:]  # retain only last two vectors

        else:
            # orthogonalize against Vs
            for i, v in enumerate(V):
                H[i, j] = np.dot(np.conjugate(v.ravel()), w.ravel())
                w = w - H[i, j]*v

            H[j+1, j] = norm(w)

            if (H[j+1, j] < breakdown):
                breakdown_flag = True
                if H[j+1, j] != 0.0:
                    w = w/H[j+1, j]
                V.append(w)
                break

            w = w/H[j+1, j]
            V.append(w)

            # if upper 2x2 block of Hessenberg matrix H is almost symmetric,
            # and the user has not explicitly specified symmetric=False,
            # then switch to symmetric Lanczos algorithm
            # if symmetric is not False and j == 1:
            #    if abs(H[1,0] - H[0,1]) < 1e-12:
            #        #print("using symmetric mode")
            #        symmetric = True
            #        V = V[1:]
            #        H[1,0] = H[0,1]
            #        beta = H[2,1]

    # print("Approximated spectral radius in %d iterations" % (j + 1))

    Eigs, Vects = eig(H[:j+1, :j+1], left=False, right=True)

    return (Vects, Eigs, H, V, breakdown_flag)


def approximate_spectral_radius(A, tol=0.01, maxiter=15, restart=5,
                                symmetric=None, initial_guess=None,
                                return_vector=False):
    """Approximate the spectral radius of a matrix.

    Parameters
    ----------
    A : {dense or sparse matrix}
        E.g. csr_matrix, csc_matrix, ndarray, etc.
    tol : {scalar}
        Relative tolerance of approximation, i.e., the error divided
        by the approximate spectral radius is compared to tol.
    maxiter : {integer}
        Maximum number of iterations to perform
    restart : {integer}
        Number of restarted Arnoldi processes.  For example, a value of 0 will
        run Arnoldi once, for maxiter iterations, and a value of 1 will restart
        Arnoldi once, using the maximal eigenvector from the first Arnoldi
        process as the initial guess.
    symmetric : {boolean}
        True  - if A is symmetric Lanczos iteration is used (more efficient)
        False - if A is non-symmetric Arnoldi iteration is used (less efficient)
    initial_guess : {array|None}
        If n x 1 array, then use as initial guess for Arnoldi/Lanczos.
        If None, then use a random initial guess.
    return_vector : {boolean}
        True - return an approximate dominant eigenvector and the spectral radius.
        False - Do not return the approximate dominant eigenvector

    Returns
    -------
    An approximation to the spectral radius of A, and
    if return_vector=True, then also return the approximate dominant
    eigenvector

    Notes
    -----
    The spectral radius is approximated by looking at the Ritz eigenvalues.
    Arnoldi iteration (or Lanczos) is used to project the matrix A onto a
    Krylov subspace: H = Q* A Q.  The eigenvalues of H (i.e. the Ritz
    eigenvalues) should represent the eigenvalues of A in the sense that the
    minimum and maximum values are usually well matched (for the symmetric case
    it is true since the eigenvalues are real).

    References
    ----------
    .. [1] Z. Bai, J. Demmel, J. Dongarra, A. Ruhe, and H. van der Vorst,
       editors.  "Templates for the Solution of Algebraic Eigenvalue Problems:
       A Practical Guide", SIAM, Philadelphia, 2000.

    Examples
    --------
    >>> from pyamg.util.linalg import approximate_spectral_radius
    >>> import numpy as np
    >>> from scipy.linalg import eigvals, norm
    >>> A = np.array([[1.,0.],[0.,1.]])
    >>> sr = approximate_spectral_radius(A,maxiter=3)
    >>> print(f'{sr:2.6}')
    1.0
    >>> print(max([norm(x) for x in eigvals(A)]))
    1.0

    """
    if not hasattr(A, 'rho') or return_vector:
        # somehow more restart causes a nonsymmetric case to fail...look at
        # this what about A.dtype=int?  convert somehow?

        # The use of the restart vector v0 requires that the full Krylov
        # subspace V be stored.  So, set symmetric to False.
        symmetric = False

        if maxiter < 1:
            raise ValueError('expected maxiter > 0')
        if restart < 0:
            raise ValueError('expected restart >= 0')
        if A.dtype == int:
            raise ValueError('expected A to be float (complex or real)')
        if A.shape[0] != A.shape[1]:
            raise ValueError('expected square A')

        if initial_guess is None:
            v0 = np.random.rand(A.shape[1], 1)
            if A.dtype == complex:
                v0 = v0 + 1.0j * np.random.rand(A.shape[1], 1)
        else:
            if initial_guess.shape[0] != A.shape[0]:
                raise ValueError('initial_guess and A must have same shape')
            if (len(initial_guess.shape) > 1) and (initial_guess.shape[1] > 1):
                raise ValueError('initial_guess must be an (n,1) or\
                                  (n,) vector')
            v0 = initial_guess.reshape(-1, 1)
            v0 = np.array(v0, dtype=A.dtype)

        for j in range(restart+1):
            [evect, ev, H, V, breakdown_flag] =\
                _approximate_eigenvalues(A, maxiter, symmetric, initial_guess=v0)
            # Calculate error in dominant eigenvector
            nvecs = ev.shape[0]
            max_index = np.abs(ev).argmax()
            error = H[nvecs, nvecs-1] * evect[-1, max_index]

            # error is a fast way of calculating the following line
            # error2 = ( A - ev[max_index]*sp.mat(
            #           sp.eye(A.shape[0],A.shape[1])) )*\
            #           ( sp.mat(sp.hstack(V[:-1]))*\
            #           evect[:,max_index].reshape(-1,1) )
            # print(str(error) + "    " + str(sp.linalg.norm(e2)))

            v0 = np.dot(np.hstack(V[:-1]), evect[:, max_index].reshape(-1, 1))

            if np.abs(error)/np.abs(ev[max_index]) < tol:
                # halt if below relative tolerance
                break

            if breakdown_flag:
                warn(f'Breakdown occured in step {j}')
                break
        # end j-loop

        rho = np.abs(ev[max_index])
        if sparse.isspmatrix(A):
            A.rho = rho

        if return_vector:
            return (rho, v0)

        return rho

    return A.rho


def condest(A, maxiter=25, symmetric=False):
    r"""Estimates the condition number of A.

    Parameters
    ----------
    A   : {dense or sparse matrix}
        e.g. array, matrix, csr_matrix, ...
    maxiter: {int}
        Max number of Arnoldi/Lanczos iterations
    symmetric : {bool}
        If symmetric use the far more efficient Lanczos algorithm,
        Else use Arnoldi.
        If hermitian, use symmetric=True.
        If complex symmetric, use symmetric=False.

    Returns
    -------
    Estimate of cond(A) with \|lambda_max\| / \|lambda_min\| or simga_max / sigma_min
    through the use of Arnoldi or Lanczos iterations, depending on
    the symmetric flag

    Notes
    -----
    The condition number measures how large of a change in the
    the problems solution is caused by a change in problem's input.
    Large condition numbers indicate that small perturbations
    and numerical errors are magnified greatly when solving the system.

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.util.linalg import condest
    >>> c = condest(np.array([[1.,0.],[0.,2.]]))
    >>> print(f'{c:2.6}')
    2.0

    """
    C = aslinearoperator(A)
    power = 1
    if not symmetric:
        def matvec(v):
            return C.rmatvec(C.A @ v)
        C.matvec = matvec
        power = 0.5

    [evect, ev, H, V, breakdown_flag] =\
        _approximate_eigenvalues(C, maxiter, symmetric)
    del evect, H, V, breakdown_flag

    return (np.max([norm(x) for x in ev])/min(norm(x) for x in ev))**power


def cond(A):
    """Return condition number of A.

    Parameters
    ----------
    A   : {dense or sparse matrix}
        e.g. array, matrix, csr_matrix, ...

    Returns
    -------
    2-norm condition number through use of the SVD
    Use for small to moderate sized dense matrices.
    For large sparse matrices, use condest.

    Notes
    -----
    The condition number measures how large of a change in
    the problems solution is caused by a change in problem's input.
    Large condition numbers indicate that small perturbations
    and numerical errors are magnified greatly when solving the system.

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.util.linalg import condest
    >>> c = condest(np.array([[1.0,0.],[0.,2.0]]))
    >>> print(f'{c:2.6}')
    2.0

    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    if sparse.isspmatrix(A):
        A = A.toarray()

    U, Sigma, Vh = svd(A)
    del U, Vh

    # 2-Norm Condition Number
    return np.max(Sigma)/min(Sigma)


def ishermitian(A, fast_check=True, tol=1e-6, verbose=False):
    r"""Return True if A is Hermitian to within tol.

    Parameters
    ----------
    A   : {dense or sparse matrix}
        e.g. array, matrix, csr_matrix, ...
    fast_check : {bool}
        If True, use the heuristic < Ax, y> = < x, Ay>
        for random vectors x and y to check for conjugate symmetry.
        If False, compute A - A.conj().T.
    tol : {float}
        Symmetry tolerance

    verbose: {bool}
        prints
        max( \|A - A.conj().T\| ) if nonhermitian and fast_check=False..
        \| <Ax, y> - <x, Ay> ) \| / sqrt( \| <Ax, y> * <x, Ay> \| )
        if nonhermitian and fast_check=True

    Returns
    -------
    True                        if hermitian
    False                       if nonhermitian

    Notes
    -----
    This function applies a simple test of conjugate symmetry

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.util.linalg import ishermitian
    >>> ishermitian(np.array([[1,2],[1,1]]))
    False

    >>> from pyamg.gallery import poisson
    >>> ishermitian(poisson((10,10)))
    True

    """
    # convert to array type
    if not sparse.isspmatrix(A):
        A = np.asarray(A)

    if fast_check:
        x = np.random.rand(A.shape[0], 1)
        y = np.random.rand(A.shape[0], 1)
        if A.dtype == complex:
            x = x + 1.0j*np.random.rand(A.shape[0], 1)
            y = y + 1.0j*np.random.rand(A.shape[0], 1)
        xAy = np.dot((A.dot(x)).conjugate().T, y)
        xAty = np.dot(x.conjugate().T, A.dot(y))
        diff = float(np.abs(xAy - xAty) / np.sqrt(np.abs(xAy*xAty)))

    else:
        # compute the difference, A - A.conj().T
        if sparse.isspmatrix(A):
            diff = np.ravel((A - A.conj().T).data)
        else:
            diff = np.ravel(A - A.conj().T)

        if np.max(diff.shape) == 0:
            diff = 0
        else:
            diff = np.max(np.abs(diff))

    if diff < tol:
        diff = 0
        return True

    if verbose:
        print(diff)

    return False


def pinv_array(a, tol=None):
    """Calculate the Moore-Penrose pseudo inverse of each block of the 3D array a.

    Parameters
    ----------
    a   : {dense array}
        Is of size (n, m, m)
    tol : {float}
        Used by gelss to filter numerically zeros singular values.
        If None, a suitable value is chosen for you.

    Returns
    -------
    Nothing, a is modified in place so that a[k] holds the pseudoinverse
    of that block.

    Notes
    -----
    By using lapack wrappers, this can be much faster for large n, than
    directly calling a pseudoinverse (SVD)

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.util.linalg import pinv_array
    >>> a = np.array([[[1.,2.],[1.,1.]], [[1.,1.],[3.,3.]]])
    >>> ac = a.copy()
    >>> # each block of a is inverted in-place
    >>> pinv_array(a)

    """
    n = a.shape[0]
    m = a.shape[1]

    if m == 1:
        # Pseudo-inverse of 1 x 1 matrices is trivial
        zero_entries = (a == 0.0).nonzero()[0]
        a[zero_entries] = 1.0
        a[:] = 1.0/a
        a[zero_entries] = 0.0
        del zero_entries

    else:
        # The block size is greater than 1

        # Create necessary arrays and function pointers for calculating pinv
        gelss, gelss_lwork = lapack.get_lapack_funcs(('gelss', 'gelss_lwork'),
                                                     (np.ones((1,), dtype=a.dtype)))
        RHS = np.eye(m, dtype=a.dtype)
        # pylint: disable=protected-access
        lwork = lapack._compute_lwork(gelss_lwork, m, m, m)
        # pylint: enable=protected-access

        # Choose tolerance for which singular values are zero in *gelss below
        if tol is None:
            tol = set_tol(a.dtype)

        # Invert each block of a
        for kk in range(n):
            gelssoutput = gelss(a[kk], RHS, cond=tol, lwork=lwork,
                                overwrite_a=True, overwrite_b=False)
            a[kk] = gelssoutput[1]
