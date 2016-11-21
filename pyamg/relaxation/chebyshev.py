"""Compute coefficients for polynomial smoothers
"""
from __future__ import print_function

import numpy as np


__all__ = ['chebyshev_polynomial_coefficients']


def chebyshev_polynomial_coefficients(a, b, degree):
    """Chebyshev polynomial coefficients for the interval [a,b]

    Parameters
    ----------
    a,b : float
        The left and right endpoints of the interval.
    degree : int
        Degree of desired Chebyshev polynomial

    Returns
    -------
    Coefficients of the Chebyshev polynomial C(t) with minimum
    magnitude on the interval [a,b] such that C(0) = 1.0.
    The coefficients are returned in descending order.

    Notes
    -----
    a,b typically represent the interval of the spectrum for some matrix
    that you wish to damp with a Chebyshev smoother.

    Examples
    --------
    >>> from pyamg.relaxation.chebyshev import \
            chebyshev_polynomial_coefficients
    >>> print chebyshev_polynomial_coefficients(1.0,2.0, 3)
    [-0.32323232  1.45454545 -2.12121212  1.        ]
    """

    if a >= b or a <= 0:
        raise ValueError('invalid interval [%s,%s]' % (a, b))

    # Chebyshev roots for the interval [-1,1]
    std_roots = np.cos(np.pi * (np.arange(degree) + 0.5) / degree)

    # Chebyshev roots for the interval [a,b]
    scaled_roots = 0.5 * (b-a) * (1 + std_roots) + a

    # Compute monic polynomial coefficients of polynomial with scaled roots
    scaled_poly = np.poly(scaled_roots)

    # Scale coefficients to enforce C(0) = 1.0
    scaled_poly /= np.polyval(scaled_poly, 0)

    return scaled_poly


def mls_polynomial_coefficients(rho, degree):
    """Determine the coefficients for a MLS polynomial smoother

    Parameters
    ----------
    rho : {float}
        Spectral radius of the matrix in question
    degree : {int}
        Degree of polynomial coefficients to generate

    Returns
    -------
    Tuple of arrays (coeffs,roots) containing the
    coefficients for the (symmetric) polynomial smoother and
    the roots of polynomial prolongation smoother.

    The coefficients of the polynomial are in descending order

    References
    ----------
    .. [1] Parallel multigrid smoothing: polynomial versus Gauss--Seidel
       M. F. Adams, M. Brezina, J. J. Hu, and R. S. Tuminaro
       J. Comp. Phys., 188 (2003), pp. 593--610

    Examples
    --------
    >>> from pyamg.relaxation.chebyshev import mls_polynomial_coefficients
    >>> mls = mls_polynomial_coefficients(2.0, 2)
    >>> print mls[0] # coefficients
    [   6.4  -48.   144.  -220.   180.   -75.8   14.5]
    >>> print mls[1] # roots
    [ 1.4472136  0.5527864]
    """

    # std_roots = np.cos(np.pi * (np.arange(degree) + 0.5)/ degree)
    # print std_roots

    roots = rho/2.0 * (1.0 - np.cos(2*np.pi*(np.arange(degree,
                                    dtype='float64') + 1)/(2.0*degree+1.0)))
    # print roots
    roots = 1.0/roots

    # S_coeffs = list(-np.poly(roots)[1:][::-1])

    S = np.poly(roots)[::-1]  # monomial coefficients of S error propagator

    SSA_max = rho/((2.0*degree+1.0)**2)  # upper bound spectral radius of S^2A
    S_hat = np.polymul(S, S)  # monomial coefficients of \hat{S} propagator
    S_hat = np.hstack(((-1.0/SSA_max)*S_hat, [1]))

    # coeff for combined error propagator \hat{S}S
    coeffs = np.polymul(S_hat, S)
    coeffs = -coeffs[:-1]             # coeff for smoother

    return (coeffs, roots)


if __name__ == '__main__':
    if True:
        # show Chebyshev polynomial
        a = 1.0/100.0
        b = 1.0
        degree = 4

        coeffs = chebyshev_polynomial_coefficients(a, b, degree)
        print("coeffs", coeffs)

        import matplotlib.pyplot as plt
        x = np.linspace(-0.1, 1.1, 100)
        plt.plot(x, np.polyval(coeffs, x))
        plt.vlines([a, b], -1, 1)
        y = np.polyval(coeffs, a)
        plt.hlines([-y, y], a, b)
        plt.ylim(-1.1, 1.1)
        plt.show()

    if False:
        # show MLS polynomial, currently broken?
        degree = 2
        rho = 1.0

        coeffs, roots = mls_polynomial_coefficients(rho, degree)

        x = np.linspace(0.0, 1.1, 100)
        plt.jplot(x, np.polyval(coeffs, x))
        plt.xlim(0, rho)
        plt.show()
