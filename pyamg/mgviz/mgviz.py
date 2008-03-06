""" VTK output: meshes and multigrids views

This will use the XML VTK format for unstructured meshes, .vtu

See here for a guide:  http://www.vtk.org/pdf/file-formats.pdf
"""
__all__ = ['mgviz','write_vtu']

from numpy import array, ones, zeros, sqrt, asarray, empty, concatenate, random
from numpy import uint8, kron, arange

from scipy.sparse import csr_matrix, isspmatrix_csr

from pyamg.graph import vertex_coloring

import warnings

"""Coarse grid visualization

There are three views of the aggregates:
    1. primal: nodes are collected and lines and triangles are grouped.  This
    has the benefit of clear separation between colored entities (aggregates)
    and blank space
    2. dual: aggregates are viewed through the dual mesh.  This has the benefit
    of filling the whole domain and aggregation through rounder (good) or long
    (bad) aggregates.
    3. points: just color different point different colors.  This works also
    with classical AMG

Plus...
    1. non-conforming aggregate view: shrink triangles toward baricenter
    2. high-order aggregate view: ?

"""

def mgviz(A=None,Vert=None, E2V=None, Agg=None, plot_type='primal',
          vtk_name='tmp_agg_plot.vtu', mesh_type='tri'):
    """Create .vtu files for use in Paraview

    Assumptions: 
           A = d*N x d*N sparse matrix (not needed yet)
        Vert = N+M x 2 coordinate list
         E2V = Nel x 3 triangular Element-Vertex list
         Agg = N x Nagg sparse matrix
   plot_type = primal or dual or points
    vtk_name = prefix for the .vtu file
   mesh_type = type of elements: tri, quad, tet, hex (all 3d)
           d = # of variables 
           N = # of vertices in the mesh represented in A
           M = additional (Dirichelet) nodes removed from A
         Nel = # of elements in the mesh
        Nagg = # of aggregates
     """

    # ------------------
    # points: basic
    #         works for aggregation and classical AMG
    if plot_type=='points':
        colors = None
        if A!=None:
            G=Agg * A * Agg.transpose()
            #colors = vertex_coloring(G, method='MIS')
            colors = vertex_coloring(G, method='LDF')
            colors = Agg.transpose() * colors
        Ncolors = 12
        Npts = Vert.shape[0]
        Nagg = Agg.shape[0]
        pdata = zeros((Npts,1))
        Agg = Agg.tocoo()

        if colors==None:
            for j in range(0,len(Agg.row)):
                pdata[Agg.col[j]] = Agg.row[j] % Ncolors
        else:
            pdata=colors.reshape((Npts,1))

        if mesh_type=='tri':
            Cells = {'5':E2V}
        elif mesh_type=='quad':
            Cells = {'9':E2V}
        elif mesh_type=='tet':
            Cells = {'10':E2V}
        elif mesh_type=='hex':
            Cells = {'12':E2V}

        write_vtu(Verts=Vert,Cells=Cells,vtk_name=vtk_name,index_base=None,pdata=pdata)

    if plot_type=='primal':
        Npts     = Vert.shape[0]
        Nel      = E2V.shape[0]
        Nelnodes = E2V.shape[1]
        data = ones((Nel*Nelnodes,1),dtype=uint8).ravel()
        col  = E2V.ravel()-1
        row  = kron( arange(0,Nel),ones((1,Nelnodes),dtype=uint8) ).ravel()
        spE  = csr_matrix( (data,(row,col)), shape=(Nel,Npts) )
        
        C3 = Agg * spE.transpose()
        C3 = C3.tocoo()
        mask = C3.data==3
        E3 = C3.col[mask]


        E2V3=E2V[E3,:]
        Nel3=E2V3.shape[0]
        colors = ones((Nel3,1))
        Cells = {'5':E2V3}
        cdata = ({'5':colors}), # make sure it's a tuple
        write_vtu(Verts=Vert,Cells=Cells,vtk_name=vtk_name,index_base=None,pdata=None,cdata=cdata)

def write_vtu(Verts,Cells,vtk_name='tmp.vtu',index_base=None,pdata=None,cdata=None):
    # TODO : I/O error checking
    # TODO : add checks for array sizes
    # TODO : add poly data structures (2,4,6,7 below)
    #
    # Verts: Npts x 3 (if 2, then expanded by 0)
    #        list of (x,y,z) point coordinates
    # Cells: Dictionary of with the keys
    #   keys:  info:
    #   -----  -------------------------------------
    #       1  VTK_VERTEX:         1 point        2d
    #       2  VTK_POLY_VERTEX:    n points       2d
    #       3  VTK_LINE:           2 points       2d
    #       4  VTK_POLY_LINE:      n+1 points     2d
    #       5  VTK_TRIANGLE:       3 points       2d
    #       6  VTK_TRIANGLE_STRIP: n+2 points     2d
    #       7  VTK_POLYGON:        n points       2d
    #       8  VTK_PIXEL:          4 points       2d
    #       9  VTK_QUAD:           4 points       2d
    #       10 VTK_TETRA:          4 points       3d
    #       11 VTK_VOXEL:          8 points       3d
    #       12 VTK_HEXAHEDRON:     8 points       3d
    #       13 VTK_WEDGE:          6 points       3d
    #       14 VTK_PYRAMID:        5 points       3d
    #    
    #    e.g. Cells = ['1':None,'2':None,'3':None,'4':None,'5':E2V,.....]
    #
    #    Non-Poly data is stored in Numpy array: Ncell x vtk_cell_info
    #
    #    Poly data stored in Nx1 numpy array
    #    [Ncell I1 d1 d2 d3 ... dI1 I2 d1 d2 d3 ... dI2 I3 ... ... dINcell]
    #    Each I1 must be >=3
    #
    #    index_base can override, what is found in check.  put 0 or 1 usually
    #
    #    pdata = Npts x Nfields
    #
    #    cdata = list of dictionaries in the form of Cells
    #
    #                1  2     3  4     5  6     7     8  9 10 11 12 13 14
    vtk_cell_info = [1, None, 2, None, 3, None, None, 4, 4, 4, 8, 8, 6, 5]

    Npts,dim   = Verts.shape
    if dim==2:
        Verts = concatenate((Verts,zeros((Npts,1))),1)

    Ncells = 0
    idx_min = 1
    for j in range(1,15):
        key='%d'%j
        if Cells.has_key(key):
            if (vtk_cell_info[j-1] == None) and (Cells[key] != None):
                # Poly data
                Ncells += Cells[key][0,0]
                idx_min=min(idx_min,Cells[key].min())
                raise NotImplementedError,'Poly Data not implemented yet'
            elif (vtk_cell_info[j-1] != None) and (Cells[key] != None):
                # non-Poly data
                Ncells += Cells[key].shape[0]
                idx_min=min(idx_min,Cells[key].min())

    if index_base==None:
        if abs(idx_min)>0:
            warnings.warn('Found data with 1-based index.  Attempting to adjust.  Override with index_base')
        index_base=idx_min
                
    FID = open(vtk_name,'w')
    FID.writelines('<?xml version=\"1.0\"?>\n')
    FID.writelines('<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n')
    FID.writelines('  <UnstructuredGrid>\n')
    FID.writelines('    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n' % (Npts,Ncells))
    #------------------------------------------------------------------
    FID.writelines('      <Points>\n')
    FID.writelines('        <DataArray type=\"Float32\" Name=\"vertices\" NumberOfComponents=\"3\" format=\"ascii\">\n')
    Verts.tofile(FID,' ') # prints Verts row-wise
    #for j in range(0,Npts):
    #    xyz = (Verts[j,0],Verts[j,1],Verts[j,2])
    #    FID.writelines('%15.15f %15.15f %15.15f\n' % xyz)
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
        key='%d'%j
        if Cells.has_key(key):
            if (vtk_cell_info[j-1] == None) and (Cells[key] != None):
                # Poly data
                raise NotImplementedError,'Poly Data not implemented yet'
            elif (vtk_cell_info[j-1] != None) and (Cells[key] != None):
                # non-Poly data
                offset=Cells[key].shape[1]
                for i1 in range(0,Cells[key].shape[0]):
                    cell_type[k]=j
                    cell_offset[k]=offset
                    k+=1
                #    for i2 in range(0,offset):
                #        FID.writelines('%d ' % (Cells[key][i1,i2]-index_base))
                (Cells[key]-1).tofile(FID, ' ')
                FID.writelines('\n');
    FID.writelines('        </DataArray>\n')
    FID.writelines('        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n')
    cell_offset=cell_offset.cumsum()
    cell_offset.tofile(FID,' ') # prints ints to file
    FID.writelines('\n');
    #total_offset=0
    #for k in range(0,Ncells):
    #    total_offset+=cell_offset[k]
    #    FID.writelines('%d ' % total_offset)
    FID.writelines('        </DataArray>\n')
    FID.writelines('        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n')
    cell_type.tofile(FID,' ') # prints ints to file
    #for k in range(0,Ncells):
    #    FID.writelines('%d ' % cell_type[k])
    FID.writelines('\n')
    FID.writelines('        </DataArray>\n')
    FID.writelines('      </Cells>\n')
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    FID.writelines('      <PointData>\n')
    if pdata!=None:
        if pdata.shape[0]!=Npts:
            raise ValueError, 'dimension of pdata must be of length = # of vertices'
        Nfields = pdata.shape[1]
        for j in range(0,Nfields):
            FID.writelines('        <DataArray type=\"Float32\" Name=\"pfield%d\" format=\"ascii\">\n' % (j+1))
            pdata[:,j].tofile(FID,' ') # print floats to file
            #for i in range(0,Npts):
            #    FID.writelines('%15.15f ' % pdata[i,j])
            FID.writelines('\n')
            FID.writelines('        </DataArray>\n')
    FID.writelines('      </PointData>\n')
    FID.writelines('      <CellData>\n')
    if cdata!=None:
        for k in range(0,len(cdata)):
            for j in range(1,15):
                key='%d'%j
                if Cells.has_key(key):
                    if not cdata[k].has_key(key):
                        raise ValueError, 'cdata needs to have the same dictionary form as Cells'
                    if cdata[k][key].shape[0] != Cells[key].shape[0]:
                        raise ValueError, 'cdata needs to have the same dictionary number of as Cells'
                    if (vtk_cell_info[j-1] == None) and (cdata[k][key] != None):
                        # Poly data
                        raise NotImplementedError,'Poly Data not implemented yet'
                    elif (vtk_cell_info[j-1] != None) and (cdata[k][key] != None):
                        # non-Poly data
                        FID.writelines('        <DataArray type=\"Float32\" Name=\"cfield%d\" format=\"ascii\">\n' % (k+1))
                        cdata[k][key].tofile(FID,' ')
                        #for i1 in range(0,cdata[k][key].shape[0]):
                        #    FID.writelines('%15.15f ' % (cdata[k][key][i1,0]))
                        FID.writelines('\n')
                        FID.writelines('        </DataArray>\n')
    FID.writelines('      </CellData>\n')
    FID.writelines('    </Piece>\n')
    FID.writelines('  </UnstructuredGrid>\n')
    FID.writelines('</VTKFile>\n')
    FID.close()
