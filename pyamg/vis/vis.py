"""VTK output functions.

Create coarse grid views and write meshes/primitives to .vtu files.  Use the
XML VTK format for unstructured meshes (.vtu)

This will use the XML VTK format for unstructured meshes, .vtu

See here for a guide:  http://www.vtk.org/pdf/file-formats.pdf
"""

__all__ = ['coarse_grid_vis','write_vtu','write_mesh']
__docformat__ = "restructuredtext en"

import warnings

from numpy import array, ones, zeros, sqrt, asarray, empty, concatenate, \
        random, uint8, kron, arange, diff, c_, where, arange, issubdtype, \
        integer, mean, sum

from scipy.sparse import csr_matrix, coo_matrix

from pyamg.graph import vertex_coloring


def coarse_grid_vis(fid, Vert, E2V, Agg, mesh_type, A=None, plot_type='primal'):
    """Coarse grid visualization: create .vtu files for use in Paraview

    Parameters
    ----------
        fid : {string, file object}
            file to be written, e.g. 'mymesh.vtu'
        Vert : {array}
            coordinate array (N x D)
        E2V : {array}
            element index array (Nel x Nelnodes)
        Agg : {csr_matrix}
            sparse matrix for the aggregate-vertex relationship (N x Nagg)  
        mesh_type : {string}
            type of elements: vertex, tri, quad, tet, hex (all 3d)
        plot_type : {string}
            primal or dual or points

    Returns
    -------
        - Writes data to .vtu file for use in paraview (xml 0.1 format)
    
    Notes
    -----
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
            simply color different points with different colors.  This works 
            best with classical AMG

        4. non-conforming
            shrink triangles toward barycenter

    Examples
    --------
    >>> file_name     = 'example_mesh.vtu'
    >>> agg_file_name = 'example_agg.vtu'
    >>> Vert = array([[0.0,0.0],
                      [1.0,0.0],
                      [2.0,0.0],
                      [0.0,1.0],
                      [1.0,1.0],
                      [2.0,1.0],
                      [0.0,2.0],
                      [1.0,2.0],
                      [2.0,2.0],
                      [0.0,3.0],
                      [1.0,3.0],
                      [2.0,3.0]])
    >>> E2V = array([[0,4,3],
                     [0,1,4],
                     [1,5,4],
                     [1,2,5],
                     [3,7,6],
                     [3,4,7],
                     [4,8,7],
                     [4,5,8],
                     [6,10,9],
                     [6,7,10],
                     [7,11,10],
                     [7,8,11]],dtype=int)
    >>> row = array([0,1,2,3,4,5,6,7,8,9,10,11])
    >>> col = array([1,0,1,1,0,1,0,1,0,1,0, 1])
    >>> data = ones((1,12),dtype=uint32).ravel()
    >>> Agg = csr_matrix((data,(row,col)),shape=(12,2))
    >>> coarse_grid_vis(agg_file_name, Vert=Vert, E2V=E2V, Agg=Agg, mesh_type='tri', A=None, plot_type='points')
    >>> write_mesh(file_name, Vert, E2V, mesh_type='tri')

    TODO
    ----
    - add the dual mesh
    - add support for vector problems: A = dN x dN
    - add support for primal 3D

     """

    #----------------------
    if not issubdtype(Vert.dtype,float):
        raise ValueError('Vert should be of type float')

    if E2V!=None:
        if not issubdtype(E2V.dtype,integer):
            raise ValueError('E2V should be of type integer')

    if Agg.shape[1] > Agg.shape[0]:
        raise ValueError('Agg should be of size Npts x Nagg')

    valid_mesh_types = ('vertex','tri','quad','tet','hex')
    if mesh_type not in valid_mesh_types:
        raise ValueError('mesh_type should be %s' % ' or '.join(valid_mesh_types))

    if A!=None:
        if (A.shape[0] != A.shape[1]) or (A.shape[0] != Agg.shape[0]):
            raise ValueError('expected square matrix A and compatible with Agg')

    valid_plot_types = ('points','primal','dual','dg')
    if plot_type not in valid_plot_types:
        raise ValueError('plot_type should be %s' % ' or '.join(valid_plot_types))
    #----------------------

    N        = Vert.shape[0]
    Ndof     = N
    if E2V!=None:
        Nel      = E2V.shape[0]
        Nelnodes = E2V.shape[1]
        if E2V.min() != 0:
            warnings.warn('element indices begin at %d' % E2V.min() )
    Nagg     = Agg.shape[0]

    Ncolors  = 8 # number of colors to use in the coloring algorithm


    # ------------------
    # points: basic (best for classical AMG)
    #
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

    # ------------------
    # primal: shows elements and edges in the aggregation (best for SA AMG)
    #
    if plot_type == 'primal':
        Agg = csr_matrix(Agg)

        # mask[i] == True if all vertices in element i belong to the same aggregate
        if len(Agg.indices)!=Agg.shape[0]:
            # account for 0 rows.  mark them as solitary aggregates
            full_aggs = array(Agg.sum(axis=1),dtype=int).ravel()
            full_aggs[full_aggs==1] = Agg.indices
            full_aggs[full_aggs==0] = Agg.shape[1] + arange(0,Agg.shape[0]-Agg.nnz,dtype=int).ravel()
            ElementAggs = full_aggs[E2V]
        else:
            ElementAggs = Agg.indices[E2V]

        mask = (ElementAggs[:,:-1] == ElementAggs[:,1:]).all(axis=1)

        E2V3 = E2V[mask,:]
        Nel3 = E2V3.shape[0]

        # 3 edges = 4 nodes.  find where the difference is 0 (bdy edge)
        markedges = diff(c_[ElementAggs,ElementAggs[:,0]])
        markedges[mask,:]=1
        markedelements, markededges = where(markedges==0)

        # now concatenate the edges (ie. first and next one (mod 3 index)
        E2V2 = c_[[E2V[markedelements,markededges], 
                   E2V[markedelements,(markededges+1)%3]]].T 
        Nel2 = E2V2.shape[0]

        colors2 = 2*ones((Nel2,1))  # color edges with twos
        colors3 = 3*ones((Nel3,1))  # color triangles with threes

        Cells  =  {3: E2V2, 5: E2V3}
        cdata  = ({3: colors2, 5: colors3},) # make sure it's a tuple

        write_vtu( fid, Verts=Vert, Cells=Cells, pdata=None, cdata=cdata)

    # ------------------
    # dual: shows dual elemental aggregation
    #
    if plot_type == 'dual':
        raise NotImplementedError('dual visualization not yet supported')
    
    # ------------------
    # dg: shows elements and edges in the aggregation for non-conforming meshes
    #
    if plot_type == 'dg':
        #Shrink each element in the mesh for nice plotting
        E2V, Vert = shrink_elmts(E2V, Vert)

        # plot_type = 'points' output to .vtu
        coarse_grid_vis(fid, Vert, E2V, Agg, mesh_type, A, plot_type='points')

def shrink_elmts(E2V, Vert, shrink=0.75):
    """
    Shrink the elements in the mesh by factor "shrink" towards the barycenter
    Only works for simplicial meshes

    Parameters
    ----------
        Vert   : {array}
               coordinate array (N x D)
        E2V    : {array}
               element index array (Nel x Nelnodes)
        shrink : {scalar}
               factor by which to move each element's points to each element's barycenter

    Returns
    -------
        - Vert and E2V with Vert appropriately scaled
    """
    E2V = array(E2V)
    Vert = array(Vert)
    Nelnodes = E2V.shape[1]
    Nel = E2V.shape[0]

    if(Vert.shape[1] == 2):
        Dimen = 2
        #Determine if polynomial order is greater than 1
        if(Nelnodes > 3):
            nonlin = True
            num_non_verts = Nelnodes - 3
        else:
            nonlin = False
    elif(Vert[:,2].nonzero()[0].shape[0] == 0):   #Assume 2D if last column of Vert is all zero
        Dimen = 2
        #Determine if polynomial order is greater than 1
        if(Nelnodes > 3):
            nonlin = True
            num_non_verts = Nelnodes - 3
        else:
            nonlin = False
    else:
        Dimen = 3
        #Determine if polynomial order of basis functions is greater than 1
        if(Nelnodes > 4):
            nonlin = True
            num_non_verts = Nelnodes - 4
        else:
            nonlin = False

    # Account for shared faces, for case that this is used to shrink a cont Gal mesh
    #Vert = Vert[E2V.flatten(),:]
    #Agg = Agg[E2V.flatten(),:]
    #E2V = array(range(Vert.shape[0])).reshape(Vert.shape[0]/Nelnodes, Nelnodes)
    #Nel = E2V.shape[0]
    
    #Store Barycenter for each element
    Bcenter = zeros((Nel, Vert.shape[1]))

    for i in range(Nel):
        #Assumes first Dimen+1 nodes are verts for the simplex
        verts_K = Vert[E2V[i,0:(Dimen+1)], :]
        
        #Calculate Barycenter of element i
        Bcenter[i,:] = mean(verts_K, 0)
        
        #Shift vertices to barycenter
        Vert[E2V[i,0:Dimen+1],:] = shrink*verts_K + (1-shrink)*kron(Bcenter[i,:], ones((Dimen+1,1)) )

        if(nonlin):
            # Move non-vertices to barycenter with the same formula, namely
            #    shrink*point_barycoords + (1-shrink)*barycenter.
            Vert[ E2V[i, (Dimen+1):], :] = shrink*(Vert[ E2V[i, (Dimen+1):], :]) + \
                                       (1-shrink)*kron(Bcenter[i,:], ones((num_non_verts,1)) )
    
    return E2V, Vert



def write_vtu( fid, Verts, Cells, pdata=None, cdata=None):
    """
    Write a .vtu file in xml format

    Parameters
    ----------
        fid : {string, file object}
            file to be written, e.g. 'mymesh.vtu'
        Verts : {array}
            Ndof x 3 (if 2, then expanded by 0)
            list of (x,y,z) point coordinates
        Cells : {dictionary}
            Dictionary of with the keys
        pdata : {array}
            Nfields of values for the vertices
        cdata : {dictionary}
            data for cell normals

    Returns
    -------
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
       
    TODO
    ----
        - I/O error checking
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
        try:
            fid = open(fid,'w')
        except IOError, (errno, strerror):
            print ".vtu error (%s): %s" % (errno, strerror)

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
            fid.writelines('        <DataArray type=\"Float32\" Name=\"cfield%d\" format=\"ascii\">\n' % (k+1))
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
        fid : {string, file object}
            file to be written, e.g. 'mymesh.vtu'
        Vert : {array}
            coordinate array (N x D)
        E2V : {array}
            element index array (Nel x Nelnodes)
        mesh_type : {string}
            type of elements: tri, quad, tet, hex (all 3d)
        pdata : {array}
            data on vertices (N x Nfields)
        cdata : {array}
            data on elements (Nel x Nfields)

    Returns
    -------
        - writes a .vtu file for use in Paraview
    """
    if E2V==None:
        mesh_type='vertex'

    if mesh_type == 'tri':
        Cells = {  5: E2V }
    elif mesh_type == 'quad':
        Cells = {  9: E2V }
    elif mesh_type == 'tet':
        Cells = { 10: E2V }
    elif mesh_type == 'hex':
        Cells = { 12 : E2V }
    elif mesh_type=='vertex':
        Cells = { 1: arange(0,Vert.shape[0]).reshape((Vert.shape[0],1))}
    else:
        raise ValueError('unknown mesh_type=%s' % mesh_type)

    if cdata is not None:
        cdata = ( { Cells.keys()[0] : cdata }, )

    write_vtu( fid, Verts=Vert, Cells=Cells, pdata=pdata, cdata=cdata)
