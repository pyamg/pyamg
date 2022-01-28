import pyamg
import numpy as np
import matplotlib.pyplot as plt
from common import set_figure

set_figure(fontsize=9, width=250)
fig, ax = plt.subplots()

np.random.seed(2022)
A = pyamg.gallery.poisson((1000,10000), format='csr')
#A = pyamg.gallery.poisson((10000,10000), format='csr')
ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10)
print(ml)

x0 = np.random.rand(A.shape[0])
b = np.zeros(A.shape[0])
res = []
x = ml.solve(b, x0, tol=1e-10, residuals=res)

res = np.array(res) # / res[0]
print(res[1:]/res[:-1])
ax.semilogy(res, marker='o', color='tab:gray', markerfacecolor='w', ms=3)
ax.set_xlabel('iterations')
ax.set_ylabel('residual')

xticks = [0,5,10,15,20,25]
ax.set_xticks(xticks)
ax.set_xticklabels([f'{x}' for x in xticks])

yticks = [0, -2, -4, -6, -8, -10]
ax.set_yticks([10**y for y in yticks])
ax.set_yticklabels([rf'10\textsuperscript{{{y}}}' for y in yticks])

ax.grid(True)

figname = 'example.pdf'
import sys
if len(sys.argv) > 1:
    if sys.argv[1] == '--savefig':
        plt.savefig(figname, bbox_inches='tight')
else:
    plt.show()
