import numpy as np
import matplotlib as mplt

__all__ = ['lineplot']

def lineplot(vertices, indices, linewidths=1):
    """Plot 2D line segments"""
    vertices = np.asarray(vertices)
    indices = np.asarray(indices)
    
    #3d tensor [segment index][vertex index][x/y value]
    lines = vertices[np.ravel(indices),:].reshape((indices.shape[0],2,2))
    
    col = mplt.collections.LineCollection(lines)
    col.set_color('k')
    col.set_linewidth(linewidths)

    sub = mplt.pylab.gca()
    sub.add_collection(col,autolim=True)
    sub.autoscale_view()

