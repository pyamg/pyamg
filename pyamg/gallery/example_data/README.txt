This folder stores *small* examples in MATLAB format using the following convention

from scipy.io import loadmat

data = loadmat(example_name)  

    data['A']        - sparse matrix
    data['B']        - near nullspace modes (for Smoothed Aggregation)
    data['vertices'] - vertex coordinates 
                        data['vertices'][i,:] are the coords of the i-th vertex
    data['elements'] - element indices
                        data['elements'][i,:] are the indices of the i-th element

Dirichlet boundary conditions have been removed from the matrix.  Vertices 
corresponding to Dirichlet boundary nodes come after those that correspond to
free DoF.


