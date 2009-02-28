import numpy

from pyamg.gallery import poisson
from pyamg.classical import CR
from pyamg.vis import vis_splitting

n = 50
A=poisson((n,n)).tocsr()

xx = numpy.arange(0,n,dtype=float)
x,y = numpy.meshgrid(xx,xx)
Verts = numpy.concatenate([[x.ravel()],[y.ravel()]],axis=0).T

splitting = CR(A)
vis_splitting(Verts=Verts, splitting=splitting,output='matplotlib')
