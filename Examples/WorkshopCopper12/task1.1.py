from pyamg import gallery, smoothed_aggregation_solver
A = gallery.poisson( (50,50), format='csr')
ml = smoothed_aggregation_solver(A)
# experiment with documentation 
#ml.<tab>
#ml.solve?
#gallery.poisson?
