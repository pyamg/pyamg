import sys
import numpy as np
from scipy.io import loadmat
import matplotlib as mplt
import matplotlib.pyplot as plt
import shapely.geometry as sg
from shapely.ops import cascaded_union

import pyamg
from pyamg.gallery.fem import mesh, gradgradform

def plotaggs(AggOp, V, E, G,
             ax, color='b', edgecolor='0.5', lw=1):
    """
    Parameters
    ----------
    AggOp : CSR sparse matrix
        n x nagg encoding of the aggregates AggOp[i,j] == 1 means node i is in aggregate j
    V : ndarray
        n x 2 coordinate array of the mesh
    E : ndarray
        nel x 3 index array of the mesh elements
    G : CSR sparse matrix
        n x n connectivity matrix for the vertices
    ax : axis
        matplotlib axis
    color : string
        color of the aggregates
    edgecolor : string
        color of the aggregate edges
    lw : float
        line width of the aggregate edges
    """

    for aggnum, agg in enumerate(AggOp.T):                                    # for each aggregate
        aggids = agg.indices                               # get the indices

        todraw = []                                        # collect things to draw
        if len(aggids) == 1:
            i = aggids[0]
            coords = (V[i, 0], V[i,1])
            newobj = sg.Point(coords)
            todraw.append(newobj)

        for i in aggids:                                   # for each point in the aggregate
            nbrs = G.getrow(i).indices                     # get the neighbors in the graph

            for j1 in nbrs:                                # for each neighbor
                found = False                              # mark if a triad ("triangle") is found
                for j2 in nbrs:
                    if (j1!=j2 and i!=j1 and i!=j2 and     # don't count i - j - j as a triangle
                        j1 in aggids and j2 in aggids and  # j1/j2 are in the aggregate
                        G[j1,j2]                           # j1/j2 are connected
                       ):
                        found = True                       # i - j1 - j2 are in the aggregate
                        coords = list(zip(V[[i,j1,j2], 0], V[[i,j1,j2],1]))
                        todraw.append(sg.Polygon(coords))  # add the triangle to the list
                if not found and i!=j1 and j1 in aggids:   # if we didn't find a triangle, then ...
                    coords = list(zip(V[[i,j1], 0], V[[i,j1],1]))
                    newobj = sg.LineString(coords)         # add a line object to the list
                    todraw.append(newobj)

        todraw = cascaded_union(todraw)                    # union all objects in the aggregate
        todraw = todraw.buffer(0.1)                        # expand to smooth
        todraw = todraw.buffer(-0.05)                      # then contract

        try:
            xs, ys = todraw.exterior.xy                    # get all of the exterior points
            ax.fill(xs, ys, color=color,
                    clip_on=False)                         # fill with a color
        except:
            print('uh oh')
            pass                                           # when does this happen

    # aggregate edges
    Edges = np.vstack((A.tocoo().row,A.tocoo().col)).T  # edges of the matrix graph
    inner_edges = AggOp.indices[Edges[:,0]] == AggOp.indices[Edges[:,1]]
    aggs = V[Edges[inner_edges].ravel(),:].reshape((-1, 2, 2))
    col = mplt.collections.LineCollection(aggs,
                                          color=edgecolor,
                                          linewidth=lw)
    ax.add_collection(col, autolim=True)

    ax.set_aspect('equal')

X = loadmat('pyamg.mat')
V = X['vertices']
E = X['elements']
mesh = mesh(V, E)
A, _ = gradgradform(mesh)
A = A.tocsr()

ml = pyamg.smoothed_aggregation_solver(A, keep=True)
AggOp = ml.levels[0].AggOp

fig, ax = plt.subplots(figsize=(16,8))
cmap = mplt.colors.ListedColormap("silver")
ax.tripcolor(V[:,0], V[:,1], E,
             facecolors=np.ones(E.shape[0]),
             edgecolors='darkgray', lw=0.5, cmap=cmap)
plotaggs(AggOp, V, E, A, ax, color='navy', edgecolor='tab:blue', lw=0.5)
ax.axis('equal')
ax.axis('off')

figname = 'pyamg_logo.pdf'
if len(sys.argv) > 2:
    if sys.argv[2] == '--withtext':
        figname = 'pyamg_logo_withtext.pdf'
        ax.text(170,85,
                'Algebraic Multigrid Solvers in Python',
               {'family': 'sans-serif',
                'color':  '0.4',
                'fontstyle': 'italic',
                'weight': 'bold',
                'size': 28,
                })

if len(sys.argv) > 1:
    if sys.argv[1] == '--savefig':
        ax.set_aspect('equal', 'box')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.subplots_adjust(0,0,1,1,0,0)
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(figname, bbox_inches='tight', pad_inches=0.25, transparent=True)
        plt.savefig(figname.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.25, transparent=True)
else:
    plt.show()
