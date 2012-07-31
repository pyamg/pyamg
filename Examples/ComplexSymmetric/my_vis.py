"""VTK output functions.

Create coarse grid views and write meshes/primitives to .vtu files.  Use the
XML VTK format for unstructured meshes (.vtu)

This will use the XML VTK format for unstructured meshes, .vtu

See here for a guide:  http://www.vtk.org/pdf/file-formats.pdf
"""

__docformat__ = "restructuredtext en"


import warnings

from numpy import array, ones, zeros, sqrt, asarray, empty, concatenate, \
        random, uint8, kron, arange, diff, c_, where, issubdtype, \
        integer, mean, sum, prod, ravel, hstack, invert, repeat, floor

from scipy import array, zeros, mean, kron, ones, sparse, rand
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from os import system
# pyamg
from pyamg.vis import write_basic_mesh, write_vtu
from pyamg.util.utils import scale_rows, scale_columns

# have to manually install Delaunay package from scikits
try:
    from scikits import delaunay
except:
    try:
        import delaunay
    except:
        raise ValueError("Install delaunay package from SciKits for this example")

__all__ = ['my_vis', 'shrink_elmts', 'dg_vis']

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
    Vert and E2V with Vert appropriately scaled

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



def dg_vis(fname, Vert, E2V, Agg, mesh_type, A=None):
    """Coarse grid visualization for 2-D discontinuous Galerkin Problems, for use with Paraview

    Parameters
    ----------
    fname : {string}
        file to be written, e.g. 'mymesh.vtu'
    Vert : {array}
        coordinate array (N x D)
    E2V : {array}
        element index array (Nel x Nelnodes)
    Agg : {csr_matrix}
        sparse matrix for the aggregate-vertex relationship (N x Nagg)  
    mesh_type : {string}
        type of elements: tri
    A : {sparse amtrix}
        optional, used for better coloring

    Returns
    -------
        - Writes data to two .vtk files for use in Paraview (xml 0.1 format)
    
    Notes
    -----


    Examples
    --------

     """

    #----------------------
    if not issubdtype(Vert.dtype,float):
        raise ValueError('Vert should be of type float')

    if E2V is not None:
        if not issubdtype(E2V.dtype,integer):
            raise ValueError('E2V should be of type integer')

    if Agg.shape[1] > Agg.shape[0]:
        raise ValueError('Agg should be of size Npts x Nagg')

    valid_mesh_types = ('tri')
    if mesh_type not in valid_mesh_types:
        raise ValueError('mesh_type should be %s' % ' or '.join(valid_mesh_types))

    if A is not None:
        if (A.shape[0] != A.shape[1]) or (A.shape[0] != Agg.shape[0]):
            raise ValueError('expected square matrix A and compatible with Agg')

    #----------------------

    N        = Vert.shape[0]
    Ndof     = N
    if E2V is not None:
        Nel      = E2V.shape[0]
        Nelnodes = E2V.shape[1]
        if E2V.min() != 0:
            warnings.warn('element indices begin at %d' % E2V.min() )
    Nagg = Agg.shape[0]

    Ncolors  = 16 # number of colors to use in the coloring algorithm


    # ------------------
    
    #Shrink each element in the mesh for nice plotting
    #E2V, Vert = shrink_elmts(E2V, Vert)

    # plot_type = 'vertex' output to .vtu --- throw point list down on mesh, so E2V becomes Nx1 array
    filename = fname + "_point-aggs.vtu"
    if False:#A is not None:
        # color aggregates with vertex coloring
        G = Agg.T * abs(A) * Agg
        colors = vertex_coloring(G, method='LDF')
        pdata = Agg * colors  # extend aggregate colors to vertices
    else:
        # color aggregates in sequence
        Agg   = coo_matrix(Agg)
        pdata = zeros(Ndof)
        colors = array(range(Agg.shape[1])) % Ncolors
        pdata[Agg.row] = Agg.col % Ncolors

    write_basic_mesh(Vert, E2V=array(range(N)).reshape(N,1), mesh_type='vertex', pdata=pdata, fname=filename)

    # plot_type = 'primal', using a global Delaunay triangulation of the shrunken mesh,
    #   we visualize the aggregates as if the global Delaunay triangulation defined a continuous Galerkin
    #   mesh upon which our aggregates are defined.
    #circum_cent, edges, tri_pts, tri_nbs = delaunay.delaunay(Vert[:,0], Vert[:,1])
    #coarse_grid_vis(filename, Vert, tri_pts, Agg, A=A, plot_type='primal', mesh_type='tri')
    filename = fname + "_aggs.vtu"

    # Do a local Delaunay triangulation for each aggregate, and throw that down as an element in a new mesh
    Agg = csc_matrix(Agg)
    E2Vnew = zeros((0,),dtype=int)
    colors_new = zeros((0,),dtype=int)
    for i in range(Agg.shape[1]):
        rowstart = Agg.indptr[i]
        rowend = Agg.indptr[i+1]
        #The nonzeros in column i of Agg define the dofs in Agg i
        members = Agg.indices[rowstart:rowend]
        if max(members.shape) > 2:
            circum_cent, edges, tri_pts, tri_nbs = delaunay.delaunay(Vert[members,0], Vert[members,1])
            #if i == 0:
            #    E2Vnew = ravel(members[ravel(tri_pts)])
            #    colors_new = ravel(repeat(colors[i], tri_pts.shape[0]))
            #else:    
            E2Vnew = hstack( (E2Vnew, ravel(members[ravel(tri_pts)]) ) )
            colors_new = hstack( (colors_new, ravel(repeat(colors[i], tri_pts.shape[0]))) ) 

        if max(members.shape) == 2:
            #create a dummy element, so that only a line is drawn between these two points in paraview
            #if i == 0:
            #    E2Vnew = array([0,members[0],members[1]])
            #    colors_new = array([colors[i]])
            #else:
            E2Vnew = hstack( (E2Vnew, array([0,members[0],members[1]]) ) )
            colors_new = hstack( (colors_new, colors[i]) )

    #Begin Primal plotting
    E2V = E2Vnew.reshape(-1,3)
    Agg = csr_matrix(Agg)

    if E2V.max() >= Agg.shape[0]:
        # remove elements with Dirichlet BCs
        E2V = E2V[E2V.max(axis=1) < Agg.shape[0]]

    # Find elements with all vertices in same aggregate
    if len(Agg.indices) != Agg.shape[0]:
        # account for 0 rows.  Mark them as solitary aggregates
        full_aggs = array(Agg.sum(axis=1),dtype=int).ravel()
        full_aggs[full_aggs==1] = Agg.indices
        full_aggs[full_aggs==0] = Agg.shape[1] + arange(0,Agg.shape[0]-Agg.nnz,dtype=int).ravel()
        ElementAggs = full_aggs[E2V]
    else:
        ElementAggs = Agg.indices[E2V]
    
    # mask[i] == True if all vertices in element i belong to the same aggregate
    mask = (ElementAggs[:,:-1] == ElementAggs[:,1:]).all(axis=1)
    E2V3 = E2V[mask,:]
    Nel3 = E2V3.shape[0]

    # 3 edges = 4 nodes.  Find where the difference is 0 (bdy edge)
    markedges = diff(c_[ElementAggs,ElementAggs[:,0]])
    markedges[mask,:]=1
    markedelements, markededges = where(markedges==0)

    # now concatenate the edges (i.e. first and next one (mod 3 index)
    E2V2 = c_[[E2V[markedelements,markededges], 
               E2V[markedelements,(markededges+1)%3]]].T 
    Nel2 = E2V2.shape[0]

    colors2 = colors_new[markedelements]  #2*ones((1,Nel2))  # color edges with twos
    colors3 = colors_new[mask]  #3*ones((1,Nel3))  # color triangles with threes

    Cells  =  {3: E2V2, 5: E2V3}
    cdata  =  {3: colors2, 5: colors3}

    write_vtu(Verts=Vert, Cells=Cells, pdata=None, cdata=cdata, pvdata=None, fname=filename)


def my_vis(ml, V, error=None, fname="", E2V=None, Pcols=None):
    """Coarse grid visualization for 2-D problems, for use with Paraview
       For all levels, outputs meshes, aggregates, near nullspace modes B, and selected
       prolongator basis functions.  Coarse level meshes are constructed by doing a
       Delaunay triangulation of interpolated fine grid vertices.

    Parameters
    ----------
    ml : {multilevel hierarchy}
        defines the multilevel hierarchy to visualize
    V : {array}
        coordinate array (N x D)
    Error : {array}
        Fine grid error to plot (N x D)
    fname : {string}
        string to be appended to all output files, e.g. 'diffusion1'
    E2V : {array}
        Element index array (Nel x Nelnodes) for the finest level.  If None,
        then a Delaunay triangulation is done for the finest level.  All coarse
        levels use an internally calculated Delaunay triangulation
    P_cols : {list of tuples}
        Optional input list of tuples of the form [(lvl, [ints]), ...]
        where lvl is an integer defining the level on which to output
        the list of columns in [ints].

    Returns
    -------
        - Writes data to .vtk files for use in Paraview (xml 0.1 format)
    
    Notes
    -----


    Examples
    --------

     """
    system('rm -f *.vtu')

    ##
    # For the purposes of clearer plotting, perturb vertices slightly
    V += rand(V.shape[0], V.shape[1])*1e-6

    ## 
    # Create a list of vertices and meshes for all levels
    levels = ml.levels
    Vlist = [V]
    if E2V is None:
        [circ_cent,edges,E2V,tri_nbs]=delaunay.delaunay(V[:,0], V[:,1])
    E2Vlist = [E2V]

    mesh_type_list = []
    mesh_num_list = []
    if E2V.shape[1] == 1:
        mesh_type_list.append('vertex')
        mesh_num_list.append(1)
    if E2V.shape[1] == 3:
        mesh_type_list.append('tri')
        mesh_num_list.append(5)
    if E2V.shape[1] == 4:
        if vertices.shape[1] == 2:
            mesh_type_list.append('quad')
            mesh_num_list.append(9)
    
    if sparse.isspmatrix_bsr(levels[0].A):
        nPDEs = levels[0].A.blocksize[0]
    else:
        nPDEs = 1
    
    Agglist = []
    Agg = sparse.eye(levels[0].A.shape[0]/nPDEs, levels[0].A.shape[1]/nPDEs, format='csr') 
    for i in range(1,len(levels)):
        ##
        # Interpolate the vertices to the next level by taking each
        # aggregate's center of gravity (i.e. average x and y value).
        Agg = Agg.tocsr()*levels[i-1].AggOp.tocsr()
        Agg.data[:] = 1.0
        Agglist.append(Agg)
            
        AggX = scale_rows(Agg, Vlist[0][:,0], copy=True) 
        AggY = scale_rows(Agg, Vlist[0][:,1], copy=True) 
        AggX = ones((1, AggX.shape[0]))*AggX
        AggY = ones((1, AggY.shape[0]))*AggY
        Agg = Agg.tocsc()
        count = Agg.indptr[1:]-Agg.indptr[:-1]
        AggX = (ravel(AggX)/count).reshape(-1,1)
        AggY = (ravel(AggY)/count).reshape(-1,1)
        Vlist.append(hstack((AggX, AggY)))

        [circ_cent,edges,E2Vnew,tri_nbs]=delaunay.delaunay(Vlist[i][:,0], Vlist[i][:,1])
        E2Vlist.append(E2Vnew)
        mesh_type_list.append('tri')
        mesh_num_list.append(5)

        
    ##
    # On each level, output aggregates, B, the mesh
    for i in range(len(levels)):
        mesh_num = mesh_num_list[i]
        mesh_type = mesh_type_list[i]
        vertices = Vlist[i]
        elements = E2Vlist[i]
        # Print mesh
        write_basic_mesh(vertices, elements, mesh_type=mesh_type, \
                             fname=fname+"mesh_lvl"+str(i)+".vtu")    
        # Visualize the aggregates
        if i != (len(levels)-1):
            dg_vis(fname+"aggs_lvl"+str(i), Vlist[0], \
                    E2Vlist[0], Agglist[i], mesh_type)
        # Visualize B
        if sparse.isspmatrix_bsr(levels[i].A):
            nPDEs = levels[i].A.blocksize[0]
        else:
            nPDEs = 1
        cell_stuff = {mesh_num : elements}
        for j in range(nPDEs):
            indys = arange(j,levels[i].A.shape[0],nPDEs)
            write_vtu(Verts=vertices, Cells=cell_stuff, pdata=levels[i].B[indys,:], \
                          fname=fname+"B_variable"+str(j)+"_lvl"+str(i)+".vtu")


    ##
    # Output requested prolongator basis functions
    if Pcols is not None:
        for (lvl,cols) in Pcols:
            P = levels[lvl].P.tocsc()
            cell_stuff = {mesh_num_list[lvl] : E2Vlist[lvl]}
            for i in cols:           
                Pcol = array(P[:,i].todense())
                write_vtu(Verts=Vlist[lvl], Cells=cell_stuff, pdata=Pcol, 
                          fname=fname+"P_lvl"+str(lvl)+"col"+str(i)+".vtu")
    
    ##
    # Output the error on the finest level
    if error is not None:
        error = error.reshape(-1,1)
        cell_stuff = {mesh_num_list[0] : E2Vlist[0]}
        if sparse.isspmatrix_bsr(levels[0].A):
            nPDEs = levels[0].A.blocksize[0]
        else:
            nPDEs = 1
        for j in range(nPDEs):
            indys = arange(j, levels[0].A.shape[0], nPDEs)
            write_vtu(Verts=Vlist[0], Cells=cell_stuff, pdata=error[indys,:], \
                      fname=fname+"error_variable"+str(j)+".vtu")



      
