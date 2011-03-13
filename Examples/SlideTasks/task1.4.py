from pyamg import *
ml = smoothed_aggregation_solver(A)
print(ml)
print(ml.levels[0].A.shape)
# Use up-arrow to edit previous command
print(ml.levels[0].P.shape) 
print(ml.levels[0].R.shape)
