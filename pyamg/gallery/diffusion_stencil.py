"""Generate a diffusion stencil

Supports isotropic diffusion (FE,FD), anisotropic diffusion (FE, FD), and
rotated anisotropic diffusion (FD).

The stencils include redundancy to maintain readability for simple cases (e.g.
isotropic diffusion).

-div Q A Q grad u

Q = [cos(beta) -sin(beta)]
    [sin(beta)  cos(beta)]

A = [1          0        ]
    [0          eps      ]

"""

from numpy import array, zeros
from scipy import cos, sin

__docformat__ = "restructuredtext en"

__all__ = ['diffusion_stencil']

def diffusion_stencil(type,dim=2,eps=1.0,epsz=1.0,beta=0.0,betaz=0.0):
    """
    Parameters
    ----------
    type : {'FE','FD'}
        Specifies the discretization as Q1 finite element (FE) or 2nd order
        finite difference (FD)
    dim : {1,2,3}
        The dimension of the problem.  2D or 3D
    eps  : float, optional
        Anisotropic diffusion coefficient: -div A grad u, 
        where A = [1 0; 0 `eps`].  The default is isotropic, `eps`=1.0
    epsz : float, optional
        Anisotropic diffusion coefficient for z in 3D.  epsy = `eps` for the z
        coordinate
    beta : float, optional
        Rotation angle `beta` in radians defines -div Q A Q^T grad,
        where Q = [cos(`beta`) -sin(`beta`); sin(`beta`) cos(`beta`)].  
        The default is `beta` = 0.0
    betaz : float, optional
        Rotation angle with the z-axis in the case of 3D.  betay = `beta` is
        the corresponding angle with the y-axis

    Return
    ------
    stencil : numpy array
        For 2D, a 3x3 connectivity stencil for a uniform mesh.
        For 3D, a 3x3x3 connectivity stencil for a uniform mesh.

    See Also
    --------
    stencil, poisson

    Notes
    -----
    Not all combinations are supported.

    Examples
    --------
    >>>> from diffusion_stencil import diffusion_stencil
    >>>> sten = diffusion_stencil('FD',dim=2,eps=0.0001,beta=pi/6)
    >>>> print sten

    """

    #####################################################
    # 2D cases
    if(dim==2):
        if(type=='FE'):
            if(eps==1.0 and beta==0.0):
                """
                FE approximation to
                u_xx + u_yy
                
                A = [ -1 -1  -1 ]
                    [ -1  8  -1 ]
                    [ -1 -1  -1 ]
                """
                stencil = array([[-1.0,-1.0,-1.0],
                                 [-1.0, 8.0,-1.0],
                                 [-1.0,-1.0,-1.0]])
            elif(beta==0.0):
                """
                FE approximation to
                u_xx + eps u_yy

                A = 1/3 * 
                 [- 1/2 - 1/2 eps    -2 eps + 1    - 1/2 - 1/2 eps]
                 [                                                ]
                 [   -2 + eps        4 eps + 4        -2 + eps    ]
                 [                                                ]
                 [- 1/2 - 1/2 eps    -2 eps + 1    - 1/2 - 1/2 eps]
                """
                a=-(1+eps)/2.0
                b=1-2.0*eps
                c=-2.0+eps
                d=4+4*eps
                stencil = (1.0/3.0) * array([[a,b,a],
                                             [c,d,c],
                                             [a,b,a]])
            else:
                """
                FE approximation to
                    - (eps c^2 +     s^2) u_xx + 
                    -2(eps - 1) c s       u_xy + 
                    - (    c^2 + eps s^2) u_yy

                [ -c^2*eps-s^2+3*c*s*(eps-1)-c^2-s^2*eps,        2*c^2*eps+2*s^2-4*c^2-4*s^2*eps, -c^2*eps-s^2-3*c*s*(eps-1)-c^2-s^2*eps]
                [       -4*c^2*eps-4*s^2+2*c^2+2*s^2*eps,        8*c^2*eps+8*s^2+8*c^2+8*s^2*eps,       -4*c^2*eps-4*s^2+2*c^2+2*s^2*eps]
                [ -c^2*eps-s^2-3*c*s*(eps-1)-c^2-s^2*eps,        2*c^2*eps+2*s^2-4*c^2-4*s^2*eps, -c^2*eps-s^2+3*c*s*(eps-1)-c^2-s^2*eps]

                c = cos(beta)
                s = sin(beta)
                """
                a = -cos(beta)**2*eps-sin(beta)**2\
                        +3*cos(beta)*sin(beta)*(eps-1)\
                        -cos(beta)**2-sin(beta)**2*eps
                b = 2*cos(beta)**2*eps+2*sin(beta)**2\
                        -4*cos(beta)**2-4*sin(beta)**2*eps
                c = -4*cos(beta)**2*eps-4*sin(beta)**2\
                        +2*cos(beta)**2+2*sin(beta)**2*eps
                d = 8*cos(beta)**2*eps+8*sin(beta)**2\
                        +8*cos(beta)**2+8*sin(beta)**2*eps
                e = -cos(beta)**2*eps-sin(beta)**2\
                        -3*cos(beta)*sin(beta)*(eps-1)\
                        -cos(beta)**2-sin(beta)**2*eps

                stencil = (1.0/6.0) * array([[a,b,e],
                                             [c,d,c],
                                             [e,b,a]])
        elif(type=='FD'):
            if(eps==1.0 and beta==0.0):
                """
                FD approximation to
                u_xx + u_yy
                
                A = [    -1     ]
                    [ -1  4  -1 ]
                    [    -1     ]
                """
                stencil = array([[0.0,-1.0, 0.0],
                                 [-1.0,4.0,-1.0],
                                 [0.0,-1.0,0.0]])
            elif(beta==0.0):
                """
                FD approximation to
                u_xx + eps u_yy 

                A = [        eps        ]
                    [                   ]
                    [ 1   -2*eps - 2   1]
                    [                   ]
                    [        eps        ]
                """
                stencil = array([[0,   eps,  0],
                                 [1,-2*eps-2,1],
                                 [0,   eps,  0]])
            else:
                """
                FD approximation to
                - (eps c^2 +     s^2) u_xx +
                -2(eps - 1) c s       u_xy +
                - (    c^2 + eps s^2) u_yy

                  c = cos(beta)
                  s = sin(beta)

                A = [ 1/2(eps - 1) c s    -(c^2 + eps s^2)    -1/2(eps - 1) c s  ]
                    [                                                            ]
                    [ -(eps c^2 + s^2)       2 (eps + 1)    -(eps c^2 + s^2)     ]
                    [                                                            ]
                    [  -1/2(eps - 1) c s    -(c^2 + eps s^2)  1/2(eps - 1) c s   ]
                """
                a =  (1/2)*(eps - 1) * cos(beta) * sin(beta)
                b =       -( (cos(beta))**2 + eps * (sin(beta))**2 )
                d = -( eps * (cos(beta))**2 +       (sin(beta))**2 )
                m =  2 * (eps + 1)

                stencil = array([[ a, b, -a],
                                 [ d, m,  d],
                                 [-a, b,  a]])
    #####################################################
    # 3D cases
    # TODO
    if(dim==3):
        if(type=='FE'):
            betay = beta
            epsy = eps
            stencil = zeros((3,3,3))

        else:
            stencil = array([[[ 0, 0, 0],
                    [ 0,-1, 0],
                    [ 0, 0, 0]],
                   [[ 0,-1, 0],
                    [-1, 6,-1],
                    [ 0,-1, 0]],
                   [[ 0, 0, 0],  
                    [ 0,-1, 0],
                    [ 0, 0, 0]]])

    return stencil
