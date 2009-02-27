# 3d example of viewing aggregates from SA using VTK
from pyamg.aggregation import standard_aggregation
from pyamg.vis import vis_coarse, vtk_writer
from pyamg.gallery import load_example

# retrieve the problem
data = load_example('unit_cube')
A = data['A'].tocsr()
V = data['vertices']
E2V = data['elements']

# perform smoothed aggregation
Agg = standard_aggregation(A)

# create the vtk file of aggregates
vis_coarse.vis_aggregate_groups(Verts=V, E2V=E2V, Agg=Agg, \
                                mesh_type='tet', output='vtk', \
                                fname='output_aggs.vtu')

# create the vtk file for a mesh 
vtk_writer.write_basic_mesh(Verts=V, E2V=E2V, \
                            mesh_type='tet', \
                            fname='output_mesh.vtu')

# to use paraview:
# start paraview: paraview --data=output_mesh.vtu
# apply
# under display in the object inspector: 
#           select wireframe representation
#           select a better solid color
#           selecting surface with edges and low opacity also helps
# open file: output_aggs.vtu
# under display in the object inspector: 
#           select surface with edges representation
#           select a better solid color
#           increase line width and point size to see these aggs (if present)
#           reduce the opacity, sometimes helps
