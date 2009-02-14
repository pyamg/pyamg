import sys

from numpy import meshgrid, arange, concatenate, ones, zeros, intc, array, tile, \
                  repeat, c_, where, linspace, sort
from scipy.sparse import csr_matrix, spdiags, kron

from pyamg.gallery.laplacian import *
from pyamg.classical import CR, binormalize
from pyamg.vis import vis_splitting

def generate_from_stencil(sten,nx,ny):
#    """
#    Build a matrix from a 3x3 stencil (for quads only)
#    """
#
#    # middle
#    M = nx * ny - 1
#    #col[ct:ct+M] = arange
#
#         center      west/east        north/south     corners
    nnz = nx*ny + 2 * ny * (nx-1) + 2 * nx * (ny-1) + 4 * (nx-1) * (ny-1)
    
    N = nx * ny

    # corners
    Isw = [0]
    Ise = [(nx-1)]
    Inw = [(ny-1)*nx]
    Ine = [N-1]

    # sides (strict)
    Is = arange(1,nx-1) # south side
    In = Inw + Is # north side
    Iw = arange(nx,N-nx,nx)
    Ie = Ise + Iw
 
    # center (strict)
    Ic = repeat(Iw,nx-2) + tile(Is,ny-2)

    A = zeros((N,3,3))

    # nw
    A[ concatenate((Ic, Is, Ie, Ise)), 0, 0 ] = sten[0, 0]
    # n
    A[ concatenate((Ic, Is, Iw, Ie, Isw, Ise)), 0, 1 ] = sten[0, 1]
    # ne
    A[ concatenate((Ic, Is, Iw, Isw)), 0, 2 ] = sten[0, 2]

    # w
    A[ concatenate((Ic, In, Is, Ie, Ine, Ise)), 1, 0 ] = sten[1, 0]
    # c
    A[:,1,1] = sten[1,1]
    # e
    A[ concatenate((Ic, In, Is, Iw, Inw, Isw)), 1, 2 ] = sten[1, 2]
    
    # sw
    A[ concatenate((Ic, In, Ie, Ine)), 2, 0 ] = sten[2, 0]
    # s
    A[ concatenate((Ic, In, Iw, Ie, Inw, Ine)), 2, 1 ] = sten[2, 1]
    # se
    A[ concatenate((Ic, In, Iw, Inw)), 2, 2 ] = sten[2, 2]

    #print Isw
    #print Ise 
    #print Inw
    #print Ine 
    #print Is
    #print In 
    #print Iw 
    #print Ie 
    #print Ic 

    (i,j,k) = A.nonzero()
    row = i
    col = row + k-1 + nx*(abs(2-j)-1)   # adjust index
    B = csr_matrix( (A[i,j,k], (row,col)), shape=(N,N))

    # vertices
    x,y = meshgrid(linspace(0,1,nx),linspace(0,1,ny))
    x = x.flatten()
    y = y.flatten()
    Vert = concatenate([[x],[y]],axis=0).T

    # quad element list
    Nel = (nx - 1) * (ny - 1)
    E2V = zeros((Nel,4),dtype='intc')
    E2V[:,0] = sort(concatenate((Isw,Iw,Is,Ic)))  #sw
    E2V[:,1] = sort(concatenate((Ise,Ie,Is,Ic)))  #se
    E2V[:,2] = sort(concatenate((Ine,Ie,In,Ic)))  #ne
    E2V[:,3] = sort(concatenate((Inw,Iw,In,Ic)))  #nw

    return B, Vert, E2V

test=3

if test==3:
    sten=array([[ 0, -1,  0],
                [-1,  4, -1],
                [ 0, -1,  0]])
    sten=array([[ 0, -0.0002,  0],
                [-0.2498,  0.5, -0.2498],
                [ 0, -0.0002,  0]])

    #sten=array([[-1, -1, -1],
    #            [-1,  8, -1],
    #            [-1, -1, -1]])
    nx = 50
    ny = 50
    N = nx * ny
    A, Vert, E2V =generate_from_stencil(sten,nx,ny)

    splitting = CR(A)

    vis_splitting(Verts=Verts, splitting,output='matplotlib')

if test==1:
    n = 25
    N = n * n

    A=poisson((n,n)).tocsr()
    x,y = meshgrid(arange(0,n),arange(0,n))
    x = x.flatten()
    y = y.flatten()
    Vert = concatenate([[x],[y]],axis=0).T

    Nel = (n - 1) * (n - 1)
    E2V = zeros((Nel,4),dtype='intc')
    k=0
    for iy in arange(0,n-1):
        for ix in arange(0,n-1):
            E2V[k,0] = ix + iy*n
            E2V[k,1] = ix + iy*n + 1
            E2V[k,2] = ix + iy*n + n +1
            E2V[k,3] = ix + iy*n + n 
            k += 1

    splitting = CR(A)

    row = arange(0, N)
    col = splitting
    data = ones((N,1)).ravel()
    Agg = csr_matrix( (data, (row, col)), shape=(N, 2))

    vis_splitting(Verts=Verts, splitting,output='matplotlib')

if test==2:
    n = 25
    N = n * n

    A=poisson((n,n)).tocsr()
    C = binormalize(A)
