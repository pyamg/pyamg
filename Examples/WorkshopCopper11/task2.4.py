from scipy.io import loadmat

data = loadmat('../../pyamg/gallery/example_data/recirc_flow.mat')

A = data['A']

from pyamg import *               
import numpy
b = numpy.ones((A.shape[0],))
#ml = smoothed_aggregation_solver(A, symmetry='symmetric',max_coarse=5)
ml = smoothed_aggregation_solver(A, symmetry='nonsymmetric',max_coarse=5)

res=[]
x = ml.solve(b,residuals=res)

from pylab import *
semilogy(res)
show()
