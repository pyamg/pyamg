import numpy as np
import matplotlib.pyplot as plt
from common import set_figure

res = np.loadtxt('example.res.txt')

set_figure(fontsize=9, width=250)
fig, ax = plt.subplots()

ax.semilogy(res, marker='o', color='tab:gray',
                             markerfacecolor='w',
                             markeredgecolor='tab:blue',
                             markeredgewidth=1.5,
                             lw=1,
                             ms=3)
ax.set_xlabel('V-cycle iterations')
ax.set_ylabel('residual')

xticks = [0,5,10,15,20,25]
ax.set_xticks(xticks)
ax.set_xticklabels([f'{x}' for x in xticks])

yticks = [4, 2, 0, -2, -4, -6, -8, -10]
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
