ml = smoothed_aggregation_solver(A, \
     strength='evolution',          \
     smooth=('energy', {'degree':4}) )
res = []
x = ml.solve(b, tol=1e-8, residuals=res)
semilogy(res[1:])
xlabel('iteration')
ylabel('residual norm')
title('Residual History')
show()
