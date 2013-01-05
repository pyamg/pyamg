import numpy as np
import scipy as sp
from scipy.linalg import norm
from pyamg import *
from pyamg.gallery import stencil_grid
from pyamg.gallery.diffusion import diffusion_stencil_2d

n=1e2
stencil = diffusion_stencil_2d(type='FE',epsilon=0.001,theta=sp.pi/3)
A = stencil_grid(stencil, (n,n), format='csr')
b = sp.rand(A.shape[0])
x0 = 0 * b

runs = []
options = []
options.append(('symmetric', {'theta' : 0.0 }))
options.append(('symmetric', {'theta' : 0.25}))
options.append(('evolution', {'epsilon' : 4.0}))
options.append(('algebraic_distance', {'theta' : 1e-1, 'p' : np.inf, 'R' : 10, 'alpha' : 0.5, 'k' : 20}))
options.append(('algebraic_distance', {'theta' : 1e-2, 'p' : np.inf, 'R' : 10, 'alpha' : 0.5, 'k' : 20}))
options.append(('algebraic_distance', {'theta' : 1e-3, 'p' : np.inf, 'R' : 10, 'alpha' : 0.5, 'k' : 20}))
options.append(('algebraic_distance', {'theta' : 1e-4, 'p' : np.inf, 'R' : 10, 'alpha' : 0.5, 'k' : 20}))

for opt in options:
    optstr = opt[0]+'\n    '+',\n    '.join(['%s=%s'%(u,v) for (u,v) in opt[1].items()])
    print "running %s"%(optstr)
    
    ml = smoothed_aggregation_solver(A, strength = opt, max_levels=10, max_coarse=5, keep=False)
    res = []                                
    x = ml.solve(b, x0, tol=1e-12,residuals=res)
    runs.append((res,optstr))

import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
ax.hold(True)
for run in runs:
    ax.semilogy(run[0], label=run[1], linewidth=3) 
ax.set_xlabel('Iteration')
ax.set_ylabel('Relative Residual')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.setp(plt.gca().get_legend().get_texts(), fontsize='x-small')

plt.show()
