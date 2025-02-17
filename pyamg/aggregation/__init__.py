"""Aggregation-based AMG."""

from .adaptive import adaptive_sa_solver
from .aggregate import (standard_aggregation, naive_aggregation,
                        lloyd_aggregation, balanced_lloyd_aggregation,
                        metis_aggregation)
from .aggregation import smoothed_aggregation_solver
from .tentative import fit_candidates
from .smooth import (jacobi_prolongation_smoother, richardson_prolongation_smoother,
                     energy_prolongation_smoother)
from .rootnode import rootnode_solver
from .pairwise import pairwise_solver

__all__ = [
    'adaptive_sa_solver',
    'balanced_lloyd_aggregation',
    'energy_prolongation_smoother',
    'fit_candidates',
    'jacobi_prolongation_smoother',
    'lloyd_aggregation',
    'metis_aggregation',
    'naive_aggregation',
    'pairwise_solver',
    'richardson_prolongation_smoother',
    'rootnode_solver',
    'smoothed_aggregation_solver',
    'standard_aggregation',
]
