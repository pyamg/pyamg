"""
Visualization tools for coarse grids, both C/F splittings and aggregation.

Output is either to file (VTK) or to the screen (matplotlib).

vis_splitting:        visualize C/F splittings through vertex elements
vis_aggregate_points: visualize aggregation through vertex elements
vis_aggregate_groups: visualize aggregation through groupins of vertices,
                      edges, elements
"""

__docformat__ = "restructuredtext en"

import warnings

from numpy import array, ones, zeros, sqrt, asarray, empty, concatenate, \
        random, uint8, kron, arange, diff, c_, where, arange, issubdtype, \
        integer, mean, sum, prod, ravel, hstack, invert, repeat

from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from pyamg.graph import vertex_coloring
from vtk_writer import write_basic_mesh

__all__ = ['vis_splitting']
#__all__ = ['vis_splitting', 'vis_aggregate_points','vis_aggregate_groups']

def vis_splitting(Verts, splitting, fname='output.vtu', output='vtk'):
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
        - Displays in screen or writes data to .vtu file for use in paraview (xml 0.1 format)
    
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
    >>> from numpy import array, ones
    >>> from python.vis.vis_coarse import vis_splitting
    >>> fname = 'example_mesh.vtu'
    >>> Verts = array([[0.0,0.0],
    >>>               [1.0,0.0],
    >>>               [0.0,1.0],
    >>>               [1.0,1.0]])
    >>> splitting = array([0,1,0,1,1,0,1,0])    # two variables
    >>> vis_splitting(Verts,splitting,output='matplotlib',fname=fname)


    >>> from numpy import array, ones
    >>> from scipy.io import loadmat
    >>> from pyamg.classical import RS
    >>> from pyamg.vis.vis_coarse import vis_splitting
    >>> from pyamg.vis.vtk_writer import write_basic_mesh
    >>> 
    >>> data = loadmat('square.mat')
    >>> A = data['A'].tocsr()                        # matrix
    >>> V = data['vertices']
    >>> E2V = data['elements']
    >>> 
    >>> splitting = RS(A)
    >>> 
    >>> vis_splitting(Verts=V,splitting=splitting,output='vtk',fname='output.vtu')
    >>> write_basic_mesh(Verts=V, E2V=E2V, mesh_type='tri', fname='output_mesh.vtu')
    >>> 
    >>> vis_splitting(Verts=V,splitting=splitting,output='matplotlib')

    """

    check_input(Verts,splitting)

    N        = Verts.shape[0]
    Ndof     = len(splitting) / N

    E2V = arange(0,N,dtype=int)

    ## adjust name in case of multiple variables
    a = fname.split('.')
    if len(a)<2:
        fname1 = a[0]
        fname2 = '.vtu'
    elif len(a)>=2:
        fname1 = "".join(a[:-1])
        fname2 = a[-1]
    else:
        raise ValueError('problem with fname')

    new_fname = fname
    for d in range(0,Ndof):
        # for each variables, write a file or open a figure
        
        if Ndof>1:
            new_fname = fname1 + '_%d.'%(d+1) + fname2

        cdata = splitting[(d*N):((d+1)*N)]

        if output=='vtk':
            write_basic_mesh(Verts=Verts, E2V=E2V, mesh_type='vertex', \
                             cdata=cdata, fname=new_fname)
        elif output=='matplotlib':
            from pylab import figure, show, plot, xlabel, ylabel, title, legend, axis
            cdataF = where(cdata==0)[0]
            cdataC = where(cdata==1)[0]
            xC = Verts[cdataC,0]
            yC = Verts[cdataC,1]
            xF = Verts[cdataF,0]
            yF = Verts[cdataF,1]
            figure()
            plot(xC,yC,'r.',xF,yF,'b.')
            title('C/F splitting (red=coarse, blue=fine)')
            xlabel('x')
            ylabel('y')
            axis('off')
            show()
        else:
            raise ValueError('problem with outputtype')

####
def check_input(Verts=None,E2V=None,Agg=None,A=None,splitting=None,mesh_type=None):
    """Check input for local functions"""
    if Verts is not None:
        if not issubdtype(Verts.dtype,float):
            raise ValueError('Verts should be of type float')

    if E2V is not None:
        if not issubdtype(E2V.dtype,integer):
            raise ValueError('E2V should be of type integer')
        if E2V.min() != 0:
            warnings.warn('element indices begin at %d' % E2V.min() )

    if Agg is not None:
        if Agg.shape[1] > Agg.shape[0]:
            raise ValueError('Agg should be of size Npts x Nagg')

    if A is not None:
        if Agg is not None:
            if (A.shape[0] != A.shape[1]) or (A.shape[0] != Agg.shape[0]):
                raise ValueError('expected square matrix A and compatible with Agg')
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
        valid_mesh_types = ('vertex','tri','quad','tet','hex')
        if mesh_type not in valid_mesh_types:
            raise ValueError('mesh_type should be %s' % ' or '.join(valid_mesh_types))
