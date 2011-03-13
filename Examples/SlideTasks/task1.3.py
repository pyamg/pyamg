from pyamg.gallery.diffusion import diffusion_stencil_2d
from numpy import set_printoptions
set_printoptions(precision=2)
sten = diffusion_stencil_2d(type='FD', \
       epsilon=0.001, theta=3.1416/3.0)
A = gallery.stencil_grid(sten, (100,100), format='csr')
# print the matrix stencil
#print(A[5050,:].data)
#print(sten)
