"""Krylov solvers.

This module contains several Krylov subspace methods, in addition to two simple
iterations, to solve linear systems iteratively.  These methods often use
multigrid as a preconditioner to accelerate convergence to the solution.  See [1]_ and [2]_.

Functions
---------
    - gmres
    - fgmres
    - cgne
    - cgnr
    - cg
    - bicgstab
    - steepest descent, (simple iteration)
    - minimal residual (MR), (simple iteration)


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

from ._gmres import gmres
from ._gmres_householder import gmres_householder
from ._gmres_mgs import gmres_mgs
from ._fgmres import fgmres
from ._cg import cg
from ._cr import cr
from ._cgnr import cgnr
from ._cgne import cgne
from ._bicgstab import bicgstab
from ._steepest_descent import steepest_descent
from ._minimal_residual import minimal_residual
from ._ekcg import ekcg
from ._srecg import srecg
from ._srecg_bcgs import srecg_bcgs
from ._srecg_orthodir import srecg_orthodir
from ._srecg_orthodir_new import srecg_orthodir_new

__all__ = [
    'bicgstab',
    'cg',
    'cgne',
    'cgnr',
    'cr',
    'ekcg',
    'fgmres',
    'gmres',
    'gmres_householder',
    'gmres_mgs',
    'minimal_residual',
    'srecg',
    'srecg_bcgs',
    'srecg_orthodir',
    'srecg_orthodir_new',
    'steepest_descent',
]
