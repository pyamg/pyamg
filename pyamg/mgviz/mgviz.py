""" VTK output: meshes and multigrids views

This will use the XML VTK format for unstructured meshes, .vtu

See here for a guide:  http://www.vtk.org/pdf/file-formats.pdf
"""

__all__ = ['mgviz','write_vtu','write_mesh']
__docformat__ = "restructuredtext en"

import warnings

from numpy import array, ones, zeros, sqrt, asarray, empty, concatenate, random
from numpy import uint8, kron, arange

from scipy.sparse import csr_matrix, coo_matrix

from pyamg.graph import vertex_coloring


def mgviz(fid, Vert, E2V, Agg, mesh_type, A=None, plot_type='primal'):
    """Coarse grid visualization: create .vtu files for use in Paraview

    Usage
    -----
        - mgviz(file_name, Vert, E2V, Agg, mesh_type, [A], [plot_type])

    Parameters
    ----------
        fid : string or open file-like object
            file to be written, e.g. 'mymesh.vtu'
        Vert : array
            coordinate array (N x D)
        E2V : array
            element index array (Nel x Nelnodes)
        Agg : csr_matrix
            sparse matrix for the aggregate-vertex relationship (N x Nagg)  
        mesh_type : string
            type of elements: tri, quad, tet, hex (all 3d)
        plot_type : string
            primal or dual or points

    Return
    ------
        - Writes data to .vtu file for use in paraview (xml 0.1 format)
    
    Notation
    --------
        D : 
            dimension of coordinate space
        N : 
            # of vertices in the mesh represented in A
        Ndof : 
            # of dof
        Nel : 
            # of elements in the mesh
        Nelnodes : 
            # of nodes per element (e.g. 3 for triangle)
        Nagg  : 
            # of aggregates

    Notes
    -----
        There are three views of the aggregates:

        1. primal 
            nodes are collected and lines and triangles are grouped.
            This has the benefit of clear separation between colored entities
            (aggregates) and blank space

        2. dual 
            aggregates are viewed through the dual mesh.  This has the 
            benefit of filling the whole domain and aggregation through 
            rounder (good) or long (bad) aggregates.

        3. points 
            just color different point different colors.  This works 
            also with classical AMG

        And in different settings:

        - non-conforming
            shrink triangles toward barycenter

        - high-order 
            view aggregates individually 

    Examples
    --------

    TODO
    ----
    - add error checks
    - add support for vector problems: A = dN x dN

     """

    N        = Vert.shape[0]
    Ndof     = N
    Nel      = E2V.shape[0]
    Nelnodes = E2V.shape[1]
    Nagg     = Agg.shape[0]
    Ncolors  = 12
        
    if E2V.min() != 0:
        warnings.warn('element indices begin at %d' % E2V.min() )

    # ------------------
    # points: basic
    #         works for aggregation and classical AMG
    if plot_type=='points':

        if A is not None:
            # color aggregates with vertex coloring
            G = Agg.T * abs(A) * Agg
            colors = vertex_coloring(G, method='LDF')
            pdata = Agg * colors  # extend aggregate colors to vertices
            pdata = pdata.reshape((Ndof,1))
        else:
            # color aggregates in sequence
            Agg   = coo_matrix(Agg)
            pdata = zeros((Ndof,1))
            pdata[Agg.row,0] = Agg.col % Ncolors

        write_mesh(fid, Vert, E2V, mesh_type=mesh_type, pdata=pdata)

    if plot_type == 'primal':
        Agg = csr_matrix(Agg)

        # mask[i] == True if all vertices in element i belong to the same aggregate
        ElementAggs = Agg.indices[E2V]
        mask = (ElementAggs[:,:-1] == ElementAggs[:,1:]).all(axis=1)

        E2V3 = E2V[mask,:]
        Nel3 = E2V3.shape[0]

        colors = ones((Nel3,1))
        write_mesh(fid, Vert, E2V3, mesh_type=mesh_type, cdata=colors)

def write_vtu( fid, Verts, Cells, pdata=None, cdata=None):
    """
    Write a .vtu file in xml format

    Parameters
    ----------
    Verts: 
        Ndof x 3 (if 2, then expanded by 0)
        list of (x,y,z) point coordinates
    Cells: 
        Dictionary of with the keys

    Return
    ------
        - writes a .vtu file for use in Paraview

    Notes
    -----
        - Non-Poly data is stored in Numpy array: Ncell x vtk_cell_info
    
        - Poly data stored in Nx1 numpy array
    
        - [Ncell I1 d1 d2 d3 ... dI1 I2 d1 d2 d3 ... dI2 I3 ... ... dINcell]

        - Each I1 must be >=3
    
        - pdata = Ndof x Nfields
    
        - cdata = list of dictionaries in the form of Cells

    =====  =================== ============= ===
    keys   type                n points      dim
    =====  =================== ============= ===
       1   VTK_VERTEX:         1 point        2d
       2   VTK_POLY_VERTEX:    n points       2d
       3   VTK_LINE:           2 points       2d
       4   VTK_POLY_LINE:      n+1 points     2d
       5   VTK_TRIANGLE:       3 points       2d
       6   VTK_TRIANGLE_STRIP: n+2 points     2d
       7   VTK_POLYGON:        n points       2d
       8   VTK_PIXEL:          4 points       2d
       9   VTK_QUAD:           4 points       2d
       10  VTK_TETRA:          4 points       3d
       11  VTK_VOXEL:          8 points       3d
       12  VTK_HEXAHEDRON:     8 points       3d
       13  VTK_WEDGE:          6 points       3d
       14  VTK_PYRAMID:        5 points       3d
    =====  =================== ============= ===
       
    Examples
    --------
    Cells = {'1':None,'2':None,'3':None,'4':None,'5':E2V,.....}
    
    TODO
    ----
    TODO : I/O error checking
    TODO : add checks for array sizes
    TODO : add poly data structures (2,4,6,7 below)
    """

    # number of indices per cell for each cell type
    vtk_cell_info = [-1, 1, None, 2, None, 3, None, None, 4, 4, 4, 8, 8, 6, 5]

    Ndof,dim = Verts.shape
    if dim==2:
        # always use 3d coordinates (x,y) -> (x,y,0)
        Verts = concatenate((Verts,zeros((Ndof,1))),1) 


    Ncells = 0
    idx_min = 1
    for key in range(1,15):
        if Cells.has_key(key):
            if (vtk_cell_info[key] == None) and (Cells[key] != None):
                # Poly data
                Ncells += Cells[key][0,0]
                raise NotImplementedError('Poly Data not implemented yet')
            elif (vtk_cell_info[key] != None) and (Cells[key] != None):
                # non-Poly data
                Ncells += Cells[key].shape[0]

    if type(fid) is type(''):
        fid = open(fid,'w')

    fid.writelines('<?xml version=\"1.0\"?>\n')
    fid.writelines('<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n')
    fid.writelines('  <UnstructuredGrid>\n')
    fid.writelines('    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n' % (Ndof,Ncells))
    
    #------------------------------------------------------------------
    fid.writelines('      <Points>\n')
    fid.writelines('        <DataArray type=\"Float32\" Name=\"vertices\" NumberOfComponents=\"3\" format=\"ascii\">\n')
    Verts.tofile(fid, sep=' ') # prints Verts row-wise
    fid.writelines('\n')
    fid.writelines('        </DataArray>\n')
    fid.writelines('      </Points>\n')
    #------------------------------------------------------------------
    
    #------------------------------------------------------------------
    fid.writelines('      <Cells>\n')
    
    fid.writelines('        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n')
    cell_offset = zeros((Ncells,1),dtype=uint8) # offsets are zero indexed
    cell_type   = zeros((Ncells,1),dtype=uint8)
    k=0
    for key in range(1,15):
        if key in Cells:
            if (vtk_cell_info[key] == None) and (Cells[key] != None):
                # Poly data
                raise NotImplementedError('Poly Data not implemented yet')
            elif (vtk_cell_info[key] != None) and (Cells[key] != None):
                # non-Poly data
                cell_array = Cells[key]
                offset     = cell_array.shape[1]
                
                cell_type  [k: k + cell_array.shape[0]] = key
                cell_offset[k: k + cell_array.shape[0]] = offset
                k += cell_array.shape[0]

                cell_array.tofile(fid, sep=' ')  # array of cell connectivity data
                fid.writelines('\n');
    fid.writelines('        </DataArray>\n')
    
    fid.writelines('        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n')
    cell_offset=cell_offset.cumsum()
    cell_offset.tofile(fid, sep=' ') # array of cell offsets (index of the end of each cell)
    fid.writelines('\n');
    fid.writelines('        </DataArray>\n')
    
    fid.writelines('        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n')
    cell_type.tofile(fid, sep=' ')   # array of cell types (e.g. '3 3' for two VTK lines) 
    fid.writelines('\n')
    fid.writelines('        </DataArray>\n')
    
    fid.writelines('      </Cells>\n')
    #------------------------------------------------------------------
    
    #------------------------------------------------------------------
    fid.writelines('      <PointData>\n')
    if pdata!=None:
        if pdata.shape[0]!=Ndof:
            raise ValueError('dimension of pdata must be of length = # of vertices')
        Nfields = pdata.shape[1]
        for j in range(0,Nfields):
            fid.writelines('        <DataArray type=\"Float32\" Name=\"pfield%d\" format=\"ascii\">\n' % (j+1))
            pdata[:,j].tofile(fid, sep=' ') # per vertex data
            fid.writelines('\n')
            fid.writelines('        </DataArray>\n')
    fid.writelines('      </PointData>\n')
    #------------------------------------------------------------------

    #------------------------------------------------------------------
    fid.writelines('      <CellData>\n')
    if cdata != None:
        for k in range(0, len(cdata)):
            for key in range(1,15):
                if key in Cells:
                    if key not in cdata[k]:
                        raise ValueError('cdata needs to have the same dictionary form as Cells')
                    if cdata[k][key].shape[0] != Cells[key].shape[0]:
                        raise ValueError('cdata needs to have the same dictionary number of as Cells')
                    if (vtk_cell_info[key] == None) and (cdata[k][key] != None):
                        # Poly data
                        raise NotImplementedError,'Poly Data not implemented yet'
                    elif (vtk_cell_info[key] != None) and (cdata[k][key] != None):
                        # non-Poly data
                        fid.writelines('        <DataArray type=\"Float32\" Name=\"cfield%d\" format=\"ascii\">\n' % (k+1))
                        cdata[k][key].tofile(fid,' ')
                        fid.writelines('\n')
                        fid.writelines('        </DataArray>\n')
    fid.writelines('      </CellData>\n')
    #------------------------------------------------------------------

    fid.writelines('    </Piece>\n')
    fid.writelines('  </UnstructuredGrid>\n')
    fid.writelines('</VTKFile>\n')
    
    #fid.close()  #don't close file
    
def write_mesh(fid, Vert, E2V, mesh_type='tri', pdata=None, cdata=None):
    """
    Write mesh file for basic types of elements

    Parameters
    ----------
        fid : string or open file-like object
            file to be written, e.g. 'mymesh.vtu'
        Vert : array
            coordinate array (N x D)
        E2V : array
            element index array (Nel x Nelnodes)
        mesh_type : string
            type of elements: tri, quad, tet, hex (all 3d)
        pdata : array
            data on vertices (N x Nfields)
        cdata : array
            data on elements (Nel x Nfields)

    Return
    ------
        - writes a .vtu file for use in Paraview
    """

    if mesh_type == 'tri':
        Cells = {  5: E2V }
    elif mesh_type == 'quad':
        Cells = {  9: E2V }
    elif mesh_type == 'tet':
        Cells = { 10: E2V }
    elif mesh_type == 'hex':
        Cells = { 12 : E2V }
    else:
        raise ValueError('unknown mesh_type=%s' % mesh_type)

    if cdata is not None:
        cdata = ( { Cells.keys()[0] : cdata }, )

    write_vtu( fid, Verts=Vert, Cells=Cells, pdata=pdata, cdata=cdata)
