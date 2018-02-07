"""
Visualization tools for coarse grids, both C/F splittings and aggregation.

Output is either to file (VTK) or to the screen (matplotlib).

vis_splitting:        visualize C/F splittings through vertex elements
vis_aggregate_groups: visualize aggregation through groupins of edges, elements

"""
from __future__ import absolute_import


import warnings
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, triu
from .vtk_writer import write_basic_mesh, write_vtu

__all__ = ['vis_splitting', 'vis_aggregate_groups']


def vis_aggregate_groups(Verts, E2V, Agg, mesh_type, output='vtk',
                         fname='output.vtu'):
    """
    Coarse grid visualization of aggregate groups.  Create .vtu files for use
    in Paraview or display with Matplotlib

    Parameters
    ----------
    Verts : {array}
        coordinate array (N x D)
    E2V : {array}
        element index array (Nel x Nelnodes)
    Agg : {csr_matrix}
        sparse matrix for the aggregate-vertex relationship (N x Nagg)
    mesh_type : {string}
        type of elements: vertex, tri, quad, tet, hex (all 3d)
    fname : {string, file object}
        file to be written, e.g. 'output.vtu'
    output : {string}
        'vtk' or 'matplotlib'

    Returns
    -------
        - Writes data to .vtu file for use in paraview (xml 0.1 format) or
          displays to screen using matplotlib

    Notes
    -----
        - Works for both 2d and 3d elements.  Element groupings are colored
          with data equal to 2.0 and stringy edges in the aggregate are colored
          with 3.0

    Examples
    --------
    >>> from pyamg.aggregation import standard_aggregation
    >>> from pyamg.vis.vis_coarse import vis_aggregate_groups
    >>> from pyamg.gallery import load_example
    >>> data = load_example('unit_square')
    >>> A = data['A'].tocsr()
    >>> V = data['vertices']
    >>> E2V = data['elements']
    >>> Agg = standard_aggregation(A)[0]
    >>> vis_aggregate_groups(Verts=V, E2V=E2V, Agg=Agg, mesh_type='tri',
                             output='vtk', fname='output.vtu')

    >>> from pyamg.aggregation import standard_aggregation
    >>> from pyamg.vis.vis_coarse import vis_aggregate_groups
    >>> from pyamg.gallery import load_example
    >>> data = load_example('unit_cube')
    >>> A = data['A'].tocsr()
    >>> V = data['vertices']
    >>> E2V = data['elements']
    >>> Agg = standard_aggregation(A)[0]
    >>> vis_aggregate_groups(Verts=V, E2V=E2V, Agg=Agg, mesh_type='tet',
                             output='vtk', fname='output.vtu')

    """
    check_input(Verts=Verts, E2V=E2V, Agg=Agg, mesh_type=mesh_type)
    map_type_to_key = {'tri': 5, 'quad': 9, 'tet': 10, 'hex': 12}
    if mesh_type not in map_type_to_key:
        raise ValueError('unknown mesh_type=%s' % mesh_type)
    key = map_type_to_key[mesh_type]

    Agg = csr_matrix(Agg)

    # remove elements with dirichlet BCs
    if E2V.max() >= Agg.shape[0]:
        E2V = E2V[E2V.max(axis=1) < Agg.shape[0]]

    # 1 #
    # Find elements with all vertices in same aggregate

    # account for 0 rows.  Mark them as solitary aggregates
    # TODO: (Luke) full_aggs is not defined, I think its just a mask
    #       indicated with rows are not 0.
    if len(Agg.indices) != Agg.shape[0]:
        full_aggs = ((Agg.indptr[1:] - Agg.indptr[:-1]) == 0).nonzero()[0]
        new_aggs = np.array(Agg.sum(axis=1), dtype=int).ravel()
        new_aggs[full_aggs == 1] = Agg.indices    # keep existing aggregate IDs
        new_aggs[full_aggs == 0] = Agg.shape[1]   # fill in singletons maxID+1
        ElementAggs = new_aggs[E2V]
    else:
        ElementAggs = Agg.indices[E2V]

    # 2 #
    # find all aggregates encompassing full elements
    # mask[i] == True if all vertices in element i belong to the same aggregate
    mask = np.where(abs(np.diff(ElementAggs)).max(axis=1) == 0)[0]
    # mask = (ElementAggs[:,:] == ElementAggs[:,0]).all(axis=1)
    E2V_a = E2V[mask, :]   # elements where element is full
    Nel_a = E2V_a.shape[0]

    # 3 #
    # find edges of elements in the same aggregate (brute force)

    # construct vertex to vertex graph
    col = E2V.ravel()
    row = np.kron(np.arange(0, E2V.shape[0]),
                  np.ones((E2V.shape[1],), dtype=int))
    data = np.ones((len(col),))
    if len(row) != len(col):
        raise ValueError('Problem constructing vertex-to-vertex map')
    V2V = coo_matrix((data, (row, col)), shape=(E2V.shape[0], E2V.max()+1))
    V2V = V2V.T * V2V
    V2V = triu(V2V, 1).tocoo()

    # get all the edges
    edges = np.vstack((V2V.row, V2V.col)).T

    # all the edges in the same aggregate
    E2V_b = edges[Agg.indices[V2V.row] == Agg.indices[V2V.col]]
    Nel_b = E2V_b.shape[0]

    # 3.5 #
    # single node aggregates
    sums = np.array(Agg.sum(axis=0)).ravel()
    E2V_c = np.where(sums == 1)[0]
    Nel_c = len(E2V_c)

    # 4 #
    # now write out the elements and edges
    colors_a = 3*np.ones((Nel_a,))  # color triangles with threes
    colors_b = 2*np.ones((Nel_b,))  # color edges with twos
    colors_c = 1*np.ones((Nel_c,))  # color the vertices with ones

    Cells = {1: E2V_c, 3: E2V_b, key: E2V_a}
    cdata = {1: colors_c, 3: colors_b, key: colors_a}  # make sure it's a tuple
    write_vtu(Verts=Verts, Cells=Cells, fname=fname, cdata=cdata)


def vis_splitting(Verts, splitting, output='vtk', fname='output.vtu'):
    """
    Coarse grid visualization for C/F splittings.

    Parameters
    ----------
    Verts : {array}
        coordinate array (N x D)
    splitting : {array}
        coarse(1)/fine(0) flags
    fname : {string, file object}
        file to be written, e.g. 'output.vtu'
    output : {string}
        'vtk' or 'matplotlib'

    Returns
    -------
        - Displays in screen or writes data to .vtu file for use in paraview
          (xml 0.1 format)

    Notes
    -----
    D :
        dimension of coordinate space
    N :
        # of vertices in the mesh represented in Verts
    Ndof :
        # of dof (= ldof * N)

        - simply color different points with different colors.  This works
          best with classical AMG.

        - writes a file (or opens a window) for each dof

        - for Ndof>1, they are assumed orderd [...dof1..., ...dof2..., etc]

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.vis.vis_coarse import vis_splitting
    >>> Verts = np.array([[0.0,0.0],
    ...                   [1.0,0.0],
    ...                   [0.0,1.0],
    ...                   [1.0,1.0]])
    >>> splitting = np.array([0,1,0,1,1,0,1,0])    # two variables
    >>> vis_splitting(Verts,splitting,output='vtk',fname='output.vtu')

    >>> from pyamg.classical import RS
    >>> from pyamg.vis.vis_coarse import vis_splitting
    >>> from pyamg.gallery import load_example
    >>> data = load_example('unit_square')
    >>> A = data['A'].tocsr()
    >>> V = data['vertices']
    >>> E2V = data['elements']
    >>> splitting = RS(A)
    >>> vis_splitting(Verts=V,splitting=splitting,output='vtk',
                      fname='output.vtu')

    """

    check_input(Verts, splitting)

    N = Verts.shape[0]
    Ndof = int(len(splitting) / N)

    E2V = np.arange(0, N, dtype=int)

    # adjust name in case of multiple variables
    a = fname.split('.')
    if len(a) < 2:
        fname1 = a[0]
        fname2 = '.vtu'
    elif len(a) >= 2:
        fname1 = "".join(a[:-1])
        fname2 = a[-1]
    else:
        raise ValueError('problem with fname')

    new_fname = fname
    for d in range(0, Ndof):
        # for each variables, write a file or open a figure

        if Ndof > 1:
            new_fname = fname1 + '_%d.' % (d+1) + fname2

        cdata = splitting[(d*N):((d+1)*N)]

        if output == 'vtk':
            write_basic_mesh(Verts=Verts, E2V=E2V, mesh_type='vertex',
                             cdata=cdata, fname=new_fname)
        elif output == 'matplotlib':
            from pylab import figure, show, plot, xlabel, ylabel, title, axis
            cdataF = np.where(cdata == 0)[0]
            cdataC = np.where(cdata == 1)[0]
            xC = Verts[cdataC, 0]
            yC = Verts[cdataC, 1]
            xF = Verts[cdataF, 0]
            yF = Verts[cdataF, 1]
            figure()
            plot(xC, yC, 'r.', xF, yF, 'b.', clip_on=True)
            title('C/F splitting (red=coarse, blue=fine)')
            xlabel('x')
            ylabel('y')
            axis('off')
            show()
        else:
            raise ValueError('problem with outputtype')


def check_input(Verts=None, E2V=None, Agg=None, A=None, splitting=None,
                mesh_type=None):
    """Check input for local functions"""
    if Verts is not None:
        if not np.issubdtype(Verts.dtype, np.floating):
            raise ValueError('Verts should be of type float')

    if E2V is not None:
        if not np.issubdtype(E2V.dtype, np.integer):
            raise ValueError('E2V should be of type integer')
        if E2V.min() != 0:
            warnings.warn('element indices begin at %d' % E2V.min())

    if Agg is not None:
        if Agg.shape[1] > Agg.shape[0]:
            raise ValueError('Agg should be of size Npts x Nagg')

    if A is not None:
        if Agg is not None:
            if (A.shape[0] != A.shape[1]) or (A.shape[0] != Agg.shape[0]):
                raise ValueError('expected square matrix A\
                                  and compatible with Agg')
        else:
            raise ValueError('problem with check_input')

    if splitting is not None:
        splitting = splitting.ravel()
        if Verts is not None:
            if (len(splitting) % Verts.shape[0]) != 0:
                raise ValueError('splitting must be a multiple of N')
        else:
            raise ValueError('problem with check_input')

    if mesh_type is not None:
        valid_mesh_types = ('vertex', 'tri', 'quad', 'tet', 'hex')
        if mesh_type not in valid_mesh_types:
            raise ValueError('mesh_type should be %s' %
                             ' or '.join(valid_mesh_types))
