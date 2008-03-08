""" VTK output: meshes and multigrids views

This will use the XML VTK format for unstructured meshes, .vtu

See here for a guide:  http://www.vtk.org/pdf/file-formats.pdf
"""

__all__ = ['mgviz','write_vtu']

import warnings

from numpy import array, ones, zeros, sqrt, asarray, empty, concatenate, random
from numpy import uint8, kron, arange

from scipy.sparse import csr_matrix, coo_matrix

from pyamg.graph import vertex_coloring


def mgviz(file_name, Vert, E2V, Agg, mesh_type, A=None, plot_type='primal'):
    """Coarse grid visualization: create .vtu files for use in Paraview

    Usage
    =====
        - mgviz(file_name, Vert, E2V, Agg, mesh_type, [A], [plot_type])

    Input
    =====
        file_name  : .vtu filename
        Vert       : coordinate array (N x D)
        E2V        : element index array (Nel x Nelnodes)
        Agg        : sparse matrix for the aggregate-vertex relationship (N x Nagg)  
        mesh_type  : type of elements: tri, quad, tet, hex (all 3d)
        plot_type  : primal or dual or points

    Output
    ======
        Writes data to .vtu file for use in paraview (xml 0.1 format)
    
    Notation
    ========
        D         : dimension of coordinate space
        N         : # of vertices in the mesh represented in A
        Ndof      : # of dof
        Nel       : # of elements in the mesh
        Nelnodes  : # of nodes per element (e.g. 3 for triangle)
        Nagg      : # of aggregates

    Notes
    =====
        There are three views of the aggregates:
        1. primal:  nodes are collected and lines and triangles are grouped.  This
                    has the benefit of clear separation between colored entities (aggregates)
                    and blank space
        2. dual:    aggregates are viewed through the dual mesh.  This has the benefit
                    of filling the whole domain and aggregation through rounder (good) or long
                    (bad) aggregates.
        3. points:  just color different point different colors.  This works also
                    with classical AMG

        And in different settings:
        1. non-conforming:  shrink triangles toward barycenter
        2. high-order:      view aggregates individually 

    Examples
    ========

    TODO
    ====
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

        if mesh_type == 'tri':
            Cells = { '5': E2V }
        elif mesh_type == 'quad':
            Cells = { '9': E2V }
        elif mesh_type == 'tet':
            Cells = {'10': E2V }
        elif mesh_type == 'hex':
            Cells = {'12': E2V }
        else:
            raise ValueError('unknown mesh_type=%s' % mesh_type)

        write_vtu( Verts=Vert, Cells=Cells, file_name=file_name, pdata=pdata)

    if plot_type == 'primal':
        Npts     = Vert.shape[0]
        Nel      = E2V.shape[0]
        Nelnodes = E2V.shape[1]

        # mask[i] == True if all vertices in element i belong to the same aggregate
        ElementAggs = Agg.indices[E2V]
        mask = (ElementAggs[:,:-1] == ElementAggs[:,1:]).all(axis=1)

        E2V3 = E2V[mask,:]
        Nel3 = E2V3.shape[0]

        colors = ones((Nel3,1))
        Cells  =  {'5':E2V3}
        cdata  = ({'5':colors},) # make sure it's a tuple
        write_vtu(Verts=Vert,Cells=Cells,file_name=file_name,pdata=None,cdata=cdata)

def write_vtu( Verts, Cells, file_name='tmp.vtu', pdata=None, cdata=None):
    """
    TODO : I/O error checking
    TODO : add checks for array sizes
    TODO : add poly data structures (2,4,6,7 below)
    
    Verts: Ndof x 3 (if 2, then expanded by 0)
           list of (x,y,z) point coordinates
    Cells: Dictionary of with the keys
      keys:  info:
      -----  -------------------------------------
          1  VTK_VERTEX:         1 point        2d
          2  VTK_POLY_VERTEX:    n points       2d
          3  VTK_LINE:           2 points       2d
          4  VTK_POLY_LINE:      n+1 points     2d
          5  VTK_TRIANGLE:       3 points       2d
          6  VTK_TRIANGLE_STRIP: n+2 points     2d
          7  VTK_POLYGON:        n points       2d
          8  VTK_PIXEL:          4 points       2d
          9  VTK_QUAD:           4 points       2d
          10 VTK_TETRA:          4 points       3d
          11 VTK_VOXEL:          8 points       3d
          12 VTK_HEXAHEDRON:     8 points       3d
          13 VTK_WEDGE:          6 points       3d
          14 VTK_PYRAMID:        5 points       3d
       
       e.g. Cells = {'1':None,'2':None,'3':None,'4':None,'5':E2V,.....}
    
       Non-Poly data is stored in Numpy array: Ncell x vtk_cell_info
    
       Poly data stored in Nx1 numpy array
       [Ncell I1 d1 d2 d3 ... dI1 I2 d1 d2 d3 ... dI2 I3 ... ... dINcell]
       Each I1 must be >=3
    
       pdata = Ndof x Nfields
    
       cdata = list of dictionaries in the form of Cells
    
                   1  2     3  4     5  6     7     8  9 10 11 12 13 14
    """

    # number of indices per cell for each cell type
    vtk_cell_info = [1, None, 2, None, 3, None, None, 4, 4, 4, 8, 8, 6, 5]

    Ndof,dim = Verts.shape
    if dim==2:
        # always use 3d coordinates (x,y) -> (x,y,0)
        Verts = concatenate((Verts,zeros((Ndof,1))),1) 


    Ncells = 0
    idx_min = 1
    for j in range(1,15):
        key = '%d' % j
        if Cells.has_key(key):
            if (vtk_cell_info[j-1] == None) and (Cells[key] != None):
                # Poly data
                Ncells += Cells[key][0,0]
                raise NotImplementedError('Poly Data not implemented yet')
            elif (vtk_cell_info[j-1] != None) and (Cells[key] != None):
                # non-Poly data
                Ncells += Cells[key].shape[0]

                
    FID = open(file_name,'w')
    FID.writelines('<?xml version=\"1.0\"?>\n')
    FID.writelines('<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n')
    FID.writelines('  <UnstructuredGrid>\n')
    FID.writelines('    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n' % (Ndof,Ncells))
    
    #------------------------------------------------------------------
    FID.writelines('      <Points>\n')
    FID.writelines('        <DataArray type=\"Float32\" Name=\"vertices\" NumberOfComponents=\"3\" format=\"ascii\">\n')
    Verts.tofile(FID, sep=' ') # prints Verts row-wise
    FID.writelines('\n')
    FID.writelines('        </DataArray>\n')
    FID.writelines('      </Points>\n')
    #------------------------------------------------------------------
    
    #------------------------------------------------------------------
    FID.writelines('      <Cells>\n')
    
    FID.writelines('        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n')
    cell_offset = zeros((Ncells,1),dtype=uint8) # offsets are zero indexed
    cell_type   = zeros((Ncells,1),dtype=uint8)
    k=0
    for j in range(1,15):
        key = '%d' % j
        if Cells.has_key(key):
            if (vtk_cell_info[j-1] == None) and (Cells[key] != None):
                # Poly data
                raise NotImplementedError('Poly Data not implemented yet')
            elif (vtk_cell_info[j-1] != None) and (Cells[key] != None):
                # non-Poly data
                cell_array = Cells[key]
                offset     = cell_array.shape[1]
                
                cell_type  [k: k + cell_array.shape[0]] = j
                cell_offset[k: k + cell_array.shape[0]] = offset
                k += cell_array.shape[0]

                cell_array.tofile(FID, sep=' ')  # array of cell connectivity data
                FID.writelines('\n');
    FID.writelines('        </DataArray>\n')
    
    FID.writelines('        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n')
    cell_offset=cell_offset.cumsum()
    cell_offset.tofile(FID, sep=' ') # array of cell offsets (index of the end of each cell)
    FID.writelines('\n');
    FID.writelines('        </DataArray>\n')
    
    FID.writelines('        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n')
    cell_type.tofile(FID, sep=' ')   # array of cell types (e.g. '3 3' for two VTK lines) 
    FID.writelines('\n')
    FID.writelines('        </DataArray>\n')
    
    FID.writelines('      </Cells>\n')
    #------------------------------------------------------------------
    
    #------------------------------------------------------------------
    FID.writelines('      <PointData>\n')
    if pdata!=None:
        if pdata.shape[0]!=Ndof:
            raise ValueError('dimension of pdata must be of length = # of vertices')
        Nfields = pdata.shape[1]
        for j in range(0,Nfields):
            FID.writelines('        <DataArray type=\"Float32\" Name=\"pfield%d\" format=\"ascii\">\n' % (j+1))
            pdata[:,j].tofile(FID, sep=' ') # per vertex data
            FID.writelines('\n')
            FID.writelines('        </DataArray>\n')
    FID.writelines('      </PointData>\n')
    #------------------------------------------------------------------

    #------------------------------------------------------------------
    FID.writelines('      <CellData>\n')
    if cdata!=None:
        for k in range(0,len(cdata)):
            for j in range(1,15):
                key= '%d' % j
                if Cells.has_key(key):
                    if not cdata[k].has_key(key):
                        raise ValueError('cdata needs to have the same dictionary form as Cells')
                    if cdata[k][key].shape[0] != Cells[key].shape[0]:
                        raise ValueError('cdata needs to have the same dictionary number of as Cells')
                    if (vtk_cell_info[j-1] == None) and (cdata[k][key] != None):
                        # Poly data
                        raise NotImplementedError,'Poly Data not implemented yet'
                    elif (vtk_cell_info[j-1] != None) and (cdata[k][key] != None):
                        # non-Poly data
                        FID.writelines('        <DataArray type=\"Float32\" Name=\"cfield%d\" format=\"ascii\">\n' % (k+1))
                        cdata[k][key].tofile(FID,' ')
                        FID.writelines('\n')
                        FID.writelines('        </DataArray>\n')
    FID.writelines('      </CellData>\n')
    #------------------------------------------------------------------

    FID.writelines('    </Piece>\n')
    FID.writelines('  </UnstructuredGrid>\n')
    FID.writelines('</VTKFile>\n')
    FID.close()
    
def write_mesh( Vert, E2V, file_name='tmp.vtu', mesh_type='tri',pdata=None, cdata=None):
    """
    Write mesh file for basic types of elements

    Input
    ====
        file_name  : .vtu filename
        Vert       : coordinate array (N x D)
        E2V        : element index array (Nel x Nelnodes)
        mesh_type  : type of elements: tri, quad, tet, hex (all 3d)
        pdata      : data on vertices (N x Nfields)
    """

    if mesh_type == 'tri':
        Cells = { '5': E2V }
    elif mesh_type == 'quad':
        Cells = { '9': E2V }
    elif mesh_type == 'tet':
        Cells = {'10': E2V }
    elif mesh_type == 'hex':
        Cells = {'12': E2V }
    else:
        raise ValueError('unknown mesh_type=%s' % mesh_type)

    write_vtu( Verts=Vert, Cells=Cells, file_name=file_name, pdata=pdata)
