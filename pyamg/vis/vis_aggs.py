"""
vis_aggs(...) when given a .mat file name 
with matrix, near nullspace, and mesh information, outputs .vtu 
files to help visualize aggregates on the first coarse level.
"""
__docformat__ = "restructuredtext en"

import warnings

from numpy import array, ones, zeros
from pyamg.aggregation import standard_aggregation
from pyamg.strength import symmetric_strength_of_connection, ode_strength_of_connection, energy_based_strength_of_connection
from scipy.sparse import csr_matrix
from scipy.io import loadmat
from vis import coarse_grid_vis, write_mesh, shrink_elmts

__all__ = ['vis_aggs']

def vis_aggs(mat_file, prob_type='dg', str_type='ode', ODE_epsilon=2.0, k=2, proj_type="l2", SA_theta=0.1):
    """Coarse grid visualization: create .vtu files for use in Paraview
       Only implemented for simplices

    Parameters
    ----------
    mat_file : {string}
        path for a mat file storing a dictionary that contains
        'A', 'B', 'elements' and 'vertices.  These are the matrix, near nullspace modes,
        element to node connectivity and a vertex list, respectively.
    prob_type : {string}
        'cg' denotes a continuous Galerkin mesh
        'dg' denotes a discontinous Galerkin mesh  
    str_type : {string}
        'ode' uses the ode_strength_of_connection routine to determine strength of connections
        'classic' uses the classic strength measure
        'energy_based' uses the energy_based strength measure
    ODE_epsilon : {scalar > 1.0}
        drop tolerance for the ode strength measure
    k : {integer}
        number of time steps for the ode strength measure
    proj_type : {string}
        'l2' uses l2 norm to solve minimization problem in ode strength measure
        'D_A' uses DA norm to solve minimization problem in ode strength measure
    SA_theta : {scalar in [0,1]}
        drop tolerance for the scalar strength measure
    
    Returns
    -------
        - Writes aggregates and mesh to separate .vtu files for use in paraview (xml 0.1 format)
    
    Examples
    --------
    #TODO add diffusion to gallery 
    #>>> vis_aggs(matfile='diffusion.mat', prob_type='cg', k=4)
    """
    
    #parse mat_file to determine a suitable start to our output files
    out_file_stem= mat_file[ (mat_file.rfind("/")+1):(mat_file.rfind(".mat")) ]
    
    #Load problem data
    ex = loadmat(mat_file)
    A = ex['A']
    E2V  = ex['elements']
    Vert = ex['vertices']
    B = ex['B']

    #Determine if we are in 2 or 3 D
    if(Vert.shape[1] == 2):
        Dimen = 2
    elif(Vert[:,2].nonzero()[0].shape[0] == 0):   #Assume 2D if last column of Vert is all zero
        Dimen = 2
    else:
        Dimen = 3

    # Calculate strength of connection and aggregation 
    # Note that the new pyamg hierarchy now stores C and Agg, so you can call viz based only on a hierarchy.
    if(str_type == 'ode'):
        C = ode_strength_of_connection(csr_matrix(A), B, epsilon=ODE_epsilon, k=k, proj_type=proj_type)
    elif(str_type == 'energy_based'):
        C = energy_based_strength_of_connection(A.tocsr(), theta=SA_theta, k=k)
    elif(str_type == 'classic'):
        C = symmetric_strength_of_connection(A.tocsr(), theta=SA_theta)
    else:
        raise ValueError('vis_aggs() only works for strength measures, str_type = [\'ode\' | \'classic\']')

    Agg  = standard_aggregation(C.tocsr())


    if(prob_type == 'dg'):  
        # visualize the aggregates 
        coarse_grid_vis( (out_file_stem + "_points_aggs.vtu"), Vert, E2V, Agg, A=None, plot_type='dg', mesh_type='tri')
    
        # visualize the mesh
        fid = open( (out_file_stem + "_mesh.vtu"), 'w') #test with open file object
        E2V, Vert = shrink_elmts(E2V, Vert)
        write_mesh(fid, Vert, E2V[:,0:(Dimen+1)], mesh_type='tri')
    
    elif(prob_type == 'cg'):
        # visualize the aggregates two different ways
        coarse_grid_vis( (out_file_stem + "_points_aggs.vtu"), Vert, E2V, Agg, A=A, plot_type='points', mesh_type='tri')
        coarse_grid_vis( (out_file_stem + "_primal_aggs.vtu"), Vert, E2V, Agg, A=A, plot_type='primal', mesh_type='tri')
        write_mesh( (out_file_stem + "_mesh.vtu"), Vert, E2V, mesh_type='tri')

    else:
        raise ValueError('vis_aggs() only works for prob_type = [\'dg\' | \'cg\']')
    
