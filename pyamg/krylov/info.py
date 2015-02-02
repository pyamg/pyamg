"""Krylov Solvers

This module contains several Krylov subspace methods, in addition to two simple
iterations, to solve linear systems iteratively.  These methods often use
multigrid as a preconditioner to accelerate convergence to the solution.

Functions
---------
    - gmres
    - fgmres
    - cgne
    - cgnr
    - cg
    - bicgstab
    - steepest descent, (simple iteration)
    - minimial residual (MR), (simple iteration)


References
----------
.. [1] Yousef Saad, "Iterative Methods for Sparse Linear Systems,
   Second Edition", SIAM, pp. 231-234, 2003
   http://www-users.cs.umn.edu/~saad/books.html

.. [2] Richard Barrett et al.  "Templates for the Solution of Linear Systems:
   Building Blocks for Iterative Methods, 2nd Edition", SIAM
   http://www.netlib.org/linalg/html_templates/Templates.html
   http://www.netlib.org/templates/

"""

postpone_import = 1
