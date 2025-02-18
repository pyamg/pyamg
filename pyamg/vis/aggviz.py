"""Plot aggregates."""

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt

    import shapely.geometry as sg
    from shapely.ops import unary_union
except ImportError as error:
    print(f'Packages matplotlib and shapely are required for aggviz.py: {error}')


def plotaggs(AggOp, V, G, ax,
             aggvals=None, vmin=None, vmax=None, cmapname='cool',
             buffer=None,
             **kwargs):
    """Plot aggregates.

    Parameters
    ----------
    AggOp : sparray
        Encoding, (n, nagg), of the aggregates ``AggOp[i,j] == 1``
        means node i is in aggregate j.
    V : ndarray
        Coordinate array of the mesh, (n, 2).
    G : sparray
        Connectivity matrix for the vertices, (n, n).
    ax : axis
        Matplotlib axis.
    aggvals : ndarray
        Values to use for plotting on the aggregates.
    vmin, vmax : float
        Min and max values to for ``cmapname``.
    cmapname : string
        Matplotlib cmap name.
    buffer : tuple, list
        - ``buffer[0]`` is the expansion buffer, to smooth.
        - ``buffer[1]`` is the contraction buffer, to make the aggregates smaller.
    **kwargs : dict
        Keyword arguments sent to ``matplotlib.pyplot.fill``.

    Returns
    -------
    mappable
        Mappable object for use with colorbar:
        ``matplotlib.pyplot.colorbar(mappable, ax=ax)``.
        None if aggval is None.

    """
    cmap = plt.get_cmap(cmapname)
    if aggvals is not None:
        if vmax is None or vmin is None:
            vmin = min(aggvals)
            vmax = max(aggvals)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        print(f'{vmin=} {vmax=}')
        aggcolor = [cmap(norm(v)) for v in aggvals]
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    else:
        mappable = None

    if buffer is None:
        buffer = (0.1, -0.05)

    for aggnum, agg in enumerate(AggOp.T):                 # for each aggregate
        aggids = agg.indices                               # get the indices

        todraw = []                                        # collect things to draw
        if len(aggids) == 1:
            i = aggids[0]
            coords = (V[i, 0], V[i, 1])
            newobj = sg.Point(coords)
            todraw.append(newobj)

        for i in aggids:                                   # for each point in the aggregate
            nbrs = G[i, :].indices                     # get the neighbors in the graph
            nbrs = np.array([k for k in nbrs if k != i])   # remove i from the neighbors

            for j1 in nbrs:                                # for each neighbor
                found = False                              # mark if a triad is found
                for j2 in nbrs:
                    if (j1 != j2                           # don't count i-j-j as a triangle
                       and j1 in aggids and j2 in aggids   # j1/j2 are in the aggregate
                       and G[j1, j2]):                     # j1/j2 are connected
                        found = True                       # i-j1-j2 are in the aggregate
                        coords = list(zip(V[[i, j1, j2], 0], V[[i, j1, j2], 1]))
                        todraw.append(sg.Polygon(coords))  # add the triangle to the list

                if not found and j1 in aggids:             # didn't find a triangle
                    coords = list(zip(V[[i, j1], 0], V[[i, j1], 1]))
                    newobj = sg.LineString(coords)         # add a line object to the list
                    todraw.append(newobj)

        todraw = unary_union(todraw)                       # union all in the aggregate
        todraw = todraw.buffer(buffer[0])                  # expand to smooth
        todraw = todraw.buffer(buffer[1])                  # then contract

        if not hasattr(todraw, 'geoms'):
            todraw = sg.MultiPolygon([todraw])
        try:
            # pylint: disable=no-member
            for poly in todraw.geoms:
                xs, ys = poly.exterior.xy                    # get all exterior points
                if aggvals is not None:
                    kwargs['color'] = aggcolor[aggnum]
                ax.fill(xs, ys,
                        clip_on=False,
                    **kwargs)                         # fill with a color
        except Exception:  # pylint: disable=broad-except
            print('Problem drawing exterior points')

    ax.set_aspect('equal')

    return mappable
