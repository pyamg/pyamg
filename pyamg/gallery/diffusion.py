"""Generate a diffusion stencil

Supports isotropic diffusion (FE,FD), anisotropic diffusion (FE, FD), and
rotated anisotropic diffusion (FD).

The stencils include redundancy to maintain readability for simple cases (e.g.
isotropic diffusion).

"""
from __future__ import print_function

import numpy as np


__all__ = ['diffusion_stencil_2d']


def diffusion_stencil_2d(epsilon=1.0, theta=0.0, type='FE'):
    """
    Rotated Anisotropic diffusion in 2d of the form:

    -div Q A Q^T grad u

    Q = [cos(theta) -sin(theta)]
        [sin(theta)  cos(theta)]

    A = [1          0        ]
        [0          eps      ]

    Parameters
    ----------
    epsilon  : float, optional
        Anisotropic diffusion coefficient: -div A grad u,
        where A = [1 0; 0 epsilon].  The default is isotropic, epsilon=1.0
    theta : float, optional
        Rotation angle `theta` in radians defines -div Q A Q^T grad,
        where Q = [cos(`theta`) -sin(`theta`); sin(`theta`) cos(`theta`)].
    type : {'FE','FD'}
        Specifies the discretization as Q1 finite element (FE) or 2nd order
        finite difference (FD)
        The default is `theta` = 0.0

    Returns
    -------
    stencil : numpy array
        A 3x3 diffusion stencil

    See Also
    --------
    stencil_grid, poisson

    Notes
    -----
    Not all combinations are supported.

    Examples
    --------
    >>> import scipy as sp
    >>> from pyamg.gallery.diffusion import diffusion_stencil_2d
    >>> sten = diffusion_stencil_2d(epsilon=0.0001,theta=sp.pi/6,type='FD')
    >>> print sten
    [[-0.2164847 -0.750025   0.2164847]
     [-0.250075   2.0002    -0.250075 ]
     [ 0.2164847 -0.750025  -0.2164847]]

    """
    eps = float(epsilon)  # for brevity
    theta = float(theta)

    C = np.cos(theta)
    S = np.sin(theta)
    CS = C*S
    CC = C**2
    SS = S**2

    if(type == 'FE'):
        """FE approximation to::

            - (eps c^2 +     s^2) u_xx +
            -2(eps - 1) c s       u_xy +
            - (    c^2 + eps s^2) u_yy

            [ -c^2*eps-s^2+3*c*s*(eps-1)-c^2-s^2*eps,
              2*c^2*eps+2*s^2-4*c^2-4*s^2*eps,
              -c^2*eps-s^2-3*c*s*(eps-1)-c^2-s^2*eps]

            [-4*c^2*eps-4*s^2+2*c^2+2*s^2*eps,
             8*c^2*eps+8*s^2+8*c^2+8*s^2*eps,
             -4*c^2*eps-4*s^2+2*c^2+2*s^2*eps]

            [-c^2*eps-s^2-3*c*s*(eps-1)-c^2-s^2*eps,
             2*c^2*eps+2*s^2-4*c^2-4*s^2*eps,
             -c^2*eps-s^2+3*c*s*(eps-1)-c^2-s^2*eps]

            c = cos(theta)
            s = sin(theta)
        """

        a = (-1*eps - 1)*CC + (-1*eps - 1)*SS + (3*eps - 3)*CS
        b = (2*eps - 4)*CC + (-4*eps + 2)*SS
        c = (-1*eps - 1)*CC + (-1*eps - 1)*SS + (-3*eps + 3)*CS
        d = (-4*eps + 2)*CC + (2*eps - 4)*SS
        e = (8*eps + 8)*CC + (8*eps + 8)*SS

        stencil = np.array([[a, b, c],
                            [d, e, d],
                            [c, b, a]]) / 6.0

    elif type == 'FD':
        """FD approximation to:

        - (eps c^2 +     s^2) u_xx +
        -2(eps - 1) c s       u_xy +
        - (    c^2 + eps s^2) u_yy

          c = cos(theta)
          s = sin(theta)

        A = [ 1/2(eps - 1) c s    -(c^2 + eps s^2)    -1/2(eps - 1) c s  ]
            [                                                            ]
            [ -(eps c^2 + s^2)       2 (eps + 1)    -(eps c^2 + s^2)     ]
            [                                                            ]
            [  -1/2(eps - 1) c s    -(c^2 + eps s^2)  1/2(eps - 1) c s   ]
        """

        a = 0.5*(eps - 1)*CS
        b = -(eps*SS + CC)
        c = -a
        d = -(eps*CC + SS)
        e = 2.0*(eps + 1)

        stencil = np.array([[a, b, c],
                            [d, e, d],
                            [c, b, a]])

    return stencil


def _symbolic_rotation_helper():
    """
    Simple SymPy script to generate the 3D rotation matrix and products for
    diffusion_stencil_3d.

    """

    from sympy import symbols, Matrix

    cpsi, spsi = symbols('cpsi, spsi')
    cth, sth = symbols('cth, sth')
    cphi, sphi = symbols('cphi, sphi')
    Rpsi = Matrix([[cpsi, spsi, 0], [-spsi, cpsi, 0], [0, 0, 1]])
    Rth = Matrix([[1, 0, 0], [0, cth, sth], [0, -sth, cth]])
    Rphi = Matrix([[cphi, sphi, 0], [-sphi, cphi, 0], [0, 0, 1]])

    Q = Rpsi * Rth * Rphi

    epsy, epsz = symbols('epsy, epsz')
    A = Matrix([[1, 0, 0], [0, epsy, 0], [0, 0, epsz]])

    D = Q * A * Q.T

    for i in range(3):
        for j in range(3):
            print('D[%d, %d] = %s' % (i, j, D[i, j]))


def _symbolic_product_helper():
    """
    Simple SymPy script to generate the 3D products for diffusion_stencil_3d.

    """

    from sympy import symbols, Matrix

    D11, D12, D13, D21, D22, D23, D31, D32, D33 =\
        symbols('D11, D12, D13, D21, D22, D23, D31, D32, D33')

    D = Matrix([[D11, D12, D13], [D21, D22, D23], [D31, D32, D33]])
    grad = Matrix([['dx', 'dy', 'dz']]).T
    div = grad.T

    a = div * D * grad

    print(a[0])


def diffusion_stencil_3d(epsilony=1.0, epsilonz=1.0, theta=0.0, phi=0.0,
                         psi=0.0, type='FD'):
    """
    Rotated Anisotropic diffusion in 3d of the form:

    -div Q A Q^T grad u

    Q = Rpsi Rtheta Rphi

    Rpsi = [ c   s   0 ]
           [-s   c   0 ]
           [ 0   0   1 ]
           c = cos(psi)
           s = sin(psi)

    Rtheta = [ 1   0   0 ]
             [ 0   c   s ]
             [ 0  -s   c ]
           c = cos(theta)
           s = sin(theta)

    Rphi = [ c   s   0 ]
           [-s   c   0 ]
           [ 0   0   1 ]
           c = cos(phi)
           s = sin(phi)

    Here Euler Angles are used:
    http://en.wikipedia.org/wiki/Euler_angles

    This results in

    Q = [   cphi*cpsi - cth*sphi*spsi, cpsi*sphi + cphi*cth*spsi, spsi*sth]
        [ - cphi*spsi - cpsi*cth*sphi, cphi*cpsi*cth - sphi*spsi, cpsi*sth]
        [                    sphi*sth,                 -cphi*sth,      cth]

    A = [1          0            ]
        [0          epsy         ]
        [0          0        epsz]

    D = Q A Q^T

    Parameters
    ----------
    epsilony  : float, optional
        Anisotropic diffusion coefficient in the y-direction
        where A = [1 0 0; 0 epsilon_y 0; 0 0 epsilon_z].  The default is
        isotropic, epsilon=1.0
    epsilonz  : float, optional
        Anisotropic diffusion coefficient in the z-direction
        where A = [1 0 0; 0 epsilon_y 0; 0 0 epsilon_z].  The default is
        isotropic, epsilon=1.0
    theta : float, optional
        Euler rotation angle `theta` in radians. The default is 0.0.
    phi : float, optional
        Euler rotation angle `phi` in radians. The default is 0.0.
    psi : float, optional
        Euler rotation angle `psi` in radians. The default is 0.0.
    type : {'FE','FD'}
        Specifies the discretization as Q1 finite element (FE) or 2nd order
        finite difference (FD)

    Returns
    -------
    stencil : numpy array
        A 3x3 diffusion stencil

    See Also
    --------
    stencil_grid, poisson, _symbolic_rotation_helper, _symbolic_product_helper

    Notes
    -----
    Not all combinations are supported.

    Examples
    --------
    >>> import scipy as sp
    >>> from pyamg.gallery.diffusion import diffusion_stencil_2d
    >>> sten = diffusion_stencil_2d(epsilon=0.0001,theta=sp.pi/6,type='FD')
    >>> print sten
    [[-0.2164847 -0.750025   0.2164847]
     [-0.250075   2.0002    -0.250075 ]
     [ 0.2164847 -0.750025  -0.2164847]]

    """

    epsy = float(epsilony)  # for brevity
    epsz = float(epsilonz)  # for brevity
    theta = float(theta)
    phi = float(phi)
    psi = float(psi)

    D = np.zeros((3, 3))
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    # from _symbolic_rotation_helper
    D[0, 0] = epsy*(cphi*cth*spsi + cpsi*sphi)**2 + epsz*spsi**2*sth**2 +\
        (cphi*cpsi - cth*sphi*spsi)**2
    D[0, 1] = cpsi*epsz*spsi*sth**2 +\
        epsy*(cphi*cpsi*cth - sphi*spsi)*(cphi*cth*spsi + cpsi*sphi) +\
        (cphi*cpsi - cth*sphi*spsi)*(-cphi*spsi - cpsi*cth*sphi)
    D[0, 2] = -cphi*epsy*sth*(cphi*cth*spsi + cpsi*sphi) +\
        cth*epsz*spsi*sth + sphi*sth*(cphi*cpsi - cth*sphi*spsi)
    D[1, 0] = cpsi*epsz*spsi*sth**2 +\
        epsy*(cphi*cpsi*cth - sphi*spsi)*(cphi*cth*spsi + cpsi*sphi) +\
        (cphi*cpsi - cth*sphi*spsi)*(-cphi*spsi - cpsi*cth*sphi)
    D[1, 1] = cpsi**2*epsz*sth**2 + epsy*(cphi*cpsi*cth - sphi*spsi)**2 +\
        (-cphi*spsi - cpsi*cth*sphi)**2
    D[1, 2] = -cphi*epsy*sth*(cphi*cpsi*cth - sphi*spsi) +\
        cpsi*cth*epsz*sth + sphi*sth*(-cphi*spsi - cpsi*cth*sphi)
    D[2, 0] = -cphi*epsy*sth*(cphi*cth*spsi + cpsi*sphi) + cth*epsz*spsi*sth +\
        sphi*sth*(cphi*cpsi - cth*sphi*spsi)
    D[2, 1] = -cphi*epsy*sth*(cphi*cpsi*cth - sphi*spsi) + cpsi*cth*epsz*sth +\
        sphi*sth*(-cphi*spsi - cpsi*cth*sphi)
    D[2, 2] = cphi**2*epsy*sth**2 + cth**2*epsz + sphi**2*sth**2

    stencil = np.zeros((3, 3, 3))

    if type == 'FD':
        # from _symbolic_product_helper
        # dx*(D11*dx + D21*dy + D31*dz) +
        # dy*(D12*dx + D22*dy + D32*dz) +
        # dz*(D13*dx + D23*dy + D33*dz)
        #
        # D00*dxx +
        # (D10+D01)*dxy +
        # (D20+D02)*dxz +
        # D11*dyy +
        # (D21+D12)*dyz +
        # D22*dzz

        i, j, k = (1, 1, 1)

        # dxx
        stencil[[i-1, i, i+1], j, k] += np.array([-1, 2, -1]) * D[0, 0]

        # dyy
        stencil[i, [j-1, j, j+1], k] += np.array([-1, 2, -1]) * D[1, 1]

        # dzz
        stencil[i, j, [k-1, k, k+1]] += np.array([-1, 2, -1]) * D[2, 2]

        L = np.array([-1, -1, 1, 1])
        M = np.array([-1, 1, -1, 1])
        # dxy
        stencil[i + L, j + M, k] \
            += 0.25 * np.array([1, -1, -1, 1]) * (D[1, 0] + D[0, 1])

        # dxz
        stencil[i + L, j, k + M] \
            += 0.25 * np.array([1, -1, -1, 1]) * (D[2, 0] + D[0, 2])

        # dyz
        stencil[i, j + L, k + M] \
            += 0.25 * np.array([1, -1, -1, 1]) * (D[2, 1] + D[1, 2])

        return stencil

    if type == 'FE':
        raise NotImplementedError('FE not implemented yet')
