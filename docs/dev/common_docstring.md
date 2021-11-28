    Parameters
    ----------
    A : csr_matrix or bsr_matrix
        Square, sparse matrix in CSR or BSR format
    B : array_like
        Right near-nullspace candidates stored in the columns of an NxK array.
        The default value B=None is equivalent to B=ones((N,1))
    BH : array_like
        Left near-nullspace candidates stored in the columns of an NxK array.
        BH is only used if symmetry is 'nonsymmetric'.
        The default value B=None is equivalent to BH=B.copy()
    symmetry : {'symmetric', 'hermitian', 'nonsymmetric'}
        Symmetry decisions that impact interpolation.
        'symmetric' refers to both real and complex symmetric
        'hermitian' refers to both complex Hermitian and real Hermitian
        'nonsymmetric' i.e. nonsymmetric in a hermitian sense
        For the real case, symmetric and hermitian are the same.
    strength : {'symmetric', 'classical', 'distance', 'evolution',
                ('predefined', {'C': csr_matrix}), None}
        Method used to determine the strength of connection in the graph of A.
        Method-specific parameters are passed using a
        tuple, e.g. strength=('symmetric',{'theta': 0.25 }). If strength=None,
        all nonzero entries of the matrix are considered strong.
    aggregate : {'standard', 'naive', 'lloyd',
                 ('predefined', {'AggOp': csr_matrix})}
        Method used to aggregate nodes.
    smooth : {'jacobi', 'richardson', 'energy', None}
        Method used to improve the tentative prolongator.  Method-specific
        parameters are passed using a tuple, e.g.  
        smooth= ('jacobi',{'filter': True }).
    presmoother : string or tuple or list : default ('block_gauss_seidel',
                  {'sweep':'symmetric'})
        Method used as the presmoother for the multilevel cycling. See
        smoothing.change_smoothers for a list of available methods.
    postsmoother : tuple or string or list
        Method used as the postsmoother.  See presmoother.
    Bimprove : tuple or string or list : default
                        [('block_gauss_seidel', {'sweep':'symmetric'}), None]
        Method used to improve candidate vectors in B.  A value of none indicates
        no improvement.  The list elements are of the form used for presmoother
        and postsmoother.
    max_levels : integer : default 10
        Maximum number of levels to be used in the multilevel solver.
    max_coarse : integer : default 500
        Maximum number of variables permitted on the coarse grid.
    diagonal_dominance : bool or tuple : default False
        If True (or the first tuple entry is True), then
        diagonally dominant rows are not coarsened.  The second tuple entry is
        {'theta': float} and is used for a threshold.
    keep : {bool} : default False
        Flag to indicate keeping extra operators in the hierarchy for
        diagnostics.  If True, then strength of connection (C),
        tentative prolongation (T), and aggregation (AggOp) are kept.
