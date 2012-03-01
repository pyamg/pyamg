# task1.3
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery import stencil_grid
sten = diffusion_stencil_2d(type='FD', \
       epsilon=0.001, theta=3.1416/3.0)
A = stencil_grid(sten, (100,100), format='csr')

# task1.4
from pyamg import *
ml = smoothed_aggregation_solver(A)
print(ml)
print(ml.levels[0].A.shape)
# Use up-arrow to edit previous command
print(ml.levels[0].P.shape) 
print(ml.levels[0].R.shape)
