from numpy import array, ones, zeros, sqrt, asarray, empty, concatenate, random

from scipy.sparse import csr_matrix, isspmatrix_csr

import warnings

from mgviz import mgviz

""" Using the examples in the examples directory"""

import os
from scipy import rand, array
from scipy.linalg import norm
from scipy.splinalg import cg
from scipy.io import loadmat
from pyamg import smoothed_aggregation_solver
from numpy import array, ones, zeros, sqrt, asarray, empty, concatenate, random

from mgviz import write_vtu
from mgviz import mgviz

from pyamg.gallery import load_example

test=1

if test==1:
    #ex = load_example('airfoil')
    ex = loadmat('Airfoil_p1_ref1.mat')
    aggex = loadmat('Airfoil_p1_ref1_aggs.mat')

    E2V  = ex['elements']
    Vert = ex['vertices']
    Agg = aggex['aggregates']

    # visualize the aggregates two different ways
    mgviz(A=ex['A'],Vert=Vert,E2V=E2V,Agg=Agg,plot_type='points',vtk_name='Airfoil_aggs_points.vtu',mesh_type='tri')
    mgviz(A=ex['A'],Vert=Vert,E2V=E2V,Agg=Agg,plot_type='primal',vtk_name='Airfoil_aggs_primal.vtu',mesh_type='tri')

    # visualize the mesh easily
    Cells = {'5':E2V}
    write_vtu(Vert,Cells,'Airfoil.vtu')

if test==2:
    E2V=array([[1,2,5],
               [2,6,5],
               [3,6,2],
               [3,7,6],
               [3,4,7],
               [7,4,8],
               [4,9,8]])

    Vert=array([[0,0],
                [1,0],
                [2,0],
                [3,0],
                [0,1],
                [1,1],
                [2,1],
                [3,1],
                [4,1]])

    row = array([0,0,0,0,0,1,1,1,2])
    col = array([0,1,4,5,6,2,3,7,8])
    data= array([1,1,1,1,1,1,1,1,1])
    Agg=csr_matrix((data,(row,col)),shape=(3,9))

    # visualize the aggregates two different ways
    mgviz(A=None,Vert=Vert,E2V=E2V,Agg=Agg,plot_type='points',vtk_name='test_agg_points.vtu',mesh_type='tri')
    mgviz(A=None,Vert=Vert,E2V=E2V,Agg=Agg,plot_type='primal',vtk_name='test_agg_priaml.vtu',mesh_type='tri')

    # visualize the mesh easily
    Cells = {'5':E2V}
    write_vtu(Vert,Cells,'test.vtu')
