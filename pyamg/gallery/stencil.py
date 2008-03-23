from scipy import *
from scipy.sparse import dia_matrix

def stencil_grid(S, grid, format=None):
    S    = asarray(S)
    grid = tuple(grid)

    if not (asarray(S.shape) % 2 == 1).all():
        raise ValueError('all stencil dimensions must be odd')
    
    if len(grid) != rank(S):
        raise ValueError('stencil rank must equal number of grid dimensions')
    
    N_v = prod(grid)     # number of vertices in the mesh
    N_s = (S != 0).sum() # number of nonzero stencil entries

    # diagonal offsets 
    diags = zeros(N_s, dtype=int)  

    strides = cumprod( [1] + list(reversed(grid)) )[:-1]
    indices = S.nonzero()
    for i,s in zip(indices,S.shape):
        i -= s // 2
    for stride,coords in zip(strides, reversed(indices)):
        diags += stride * coords

    data = S[ S != 0 ].repeat(N_v).reshape(N_s,N_v)

    indices = vstack(indices).T

    for index,diag in zip(indices,data):
        diag = diag.reshape(grid)
        for n,i in enumerate(index):
            if i > 0:
                s = [ slice(None) ]*len(grid)
                s[n] = slice(0,i)
                diag[s] = 0
            elif i < 0:
                s = [ slice(None) ]*len(grid)
                s[n] = slice(i,None)
                diag[s] = 0

    return dia_matrix( (data,diags), shape=(N_v,N_v)).asformat(format)


#S = array([-1,2,-1])
#grid = (4,)
S = array([[ 0,-1, 0],
           [-1, 4,-1],
           [ 0,-1, 0]])
#S = array([[-1,-1,-1],
#           [-1, 8,-1],
#           [-1,-1,-1]])
grid = (4,4)

A = stencil_grid( S, grid )

print A.todense()



