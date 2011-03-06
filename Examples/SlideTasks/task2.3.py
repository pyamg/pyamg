# 2D example of viewing aggregates from SA using VTK
from pyamg.aggregation import standard_aggregation
from pyamg.vis import vis_coarse, vtk_writer
from pyamg.gallery import load_example
from pyamg import *
from scipy import *

# retrieve the problem
data = load_example('unit_square')
A = data['A'].tocsr()
V = data['vertices']
E2V = data['elements']

# perform smoothed aggregation
ml = smoothed_aggregation_solver(A,keep=True,max_coarse=10)
b = sin(pi*V[:,0])*sin(pi*V[:,1])
x = ml.solve(b)

# create the vtk file of aggregates
vis_coarse.vis_aggregate_groups(Verts=V, E2V=E2V, 
        Agg=ml.levels[0].AggOp, mesh_type='tri', 
        output='vtk', fname='output_aggs.vtu')

# create the vtk file for mesh and solution 
vtk_writer.write_basic_mesh(Verts=V, E2V=E2V, 
                            pdata = x,
                            mesh_type='tri', 
                            fname='output_mesh.vtu')

# to use Paraview:
# start Paraview: Paraview --data=output_mesh.vtu
# apply
# under display in the object inspector: 
#           select wireframe representation
#           select a better solid color
# open file: output_aggs.vtu
# under display in the object inspector: 
#           select surface with edges representation
#           select a better solid color
#           increase line width and point size to see these aggs (if present)
