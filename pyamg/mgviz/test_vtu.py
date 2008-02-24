""" Using the examples in the examples directory"""

import os
from scipy import rand, array
from scipy.linalg import norm
from scipy.splinalg import cg
from scipy.io import loadmat
from pyamg import smoothed_aggregation_solver
from numpy import array, ones, zeros, sqrt, asarray, empty, concatenate, random
from mgviz import write_vtu

from pyamg.gallery import load_example

#d = loadmat('Airfoil_p1_ref1.mat')
#agg = loadmat('Airfoil_p1_ref1_aggs.mat')

ex = load_example('airfoil')

E2V  = ex['elements']
Vert = ex['vertices']
Cells = {'5':E2V}
cdata = ({'5':random.random((E2V.shape[0],1))}, {'5':2*random.random((E2V.shape[0],1))})
pdata=concatenate((random.random((Vert.shape[0],1)),2*random.random((Vert.shape[0],1))),1)

write_vtu(Vert,Cells,'airfoil.vtu',None,pdata,cdata)
