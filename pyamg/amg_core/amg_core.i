/* -*- C -*-  (not really, but good for syntax highlighting) */
%module amg_core

 /* why does SWIG complain about int arrays? a typecheck is provided */
#pragma SWIG nowarn=467

%{
#define SWIG_FILE_WITH_INIT
#include "numpy/arrayobject.h"

#include "complex_ops.h"
#include "ruge_stuben.h"
#include "smoothed_aggregation.h"
#include "relaxation.h"
#include "graph.h"
#include "evolution_strength.h"
#include "krylov.h"
#include "linalg.h"
%}

%feature("autodoc", "1");

%include "numpy.i"

%init %{
    import_array();
%}





 /*
  * IN types
  */
%define I_IN_ARRAY1( ctype )
%apply ctype * IN_ARRAY1 {
    const ctype Ap [ ],
    const ctype Ai [ ],
    const ctype Aj [ ],
    const ctype Bp [ ],
    const ctype Bi [ ],
    const ctype Bj [ ],
    const ctype Sp [ ],
    const ctype Si [ ],
    const ctype Sj [ ],
    const ctype Tp [ ],
    const ctype Ti [ ],
    const ctype Tj [ ],
    const ctype Id [ ]
};
%enddef

%define T_IN_ARRAY1( ctype )
%apply ctype * IN_ARRAY1 {
          ctype* Ax,
          ctype* Bx,
          ctype* Sx,
          ctype* x,
    const ctype Ax [ ],
    const ctype Bx [ ],
    const ctype Sx [ ],
    const ctype Tx [ ],
    const ctype Xx [ ],
    const ctype Yx [ ],
    const ctype  x [ ],
    const ctype  y [ ],
    const ctype  z [ ],
    const ctype  b [ ],
    const ctype  B [ ],
    const ctype omega[ ]
};
%enddef



 /*
  * OUT types
  */
%define I_ARRAY_ARGOUT( ctype )
%apply std::vector<ctype>* array_argout {
    std::vector<ctype>* Ap,
    std::vector<ctype>* Ai,
    std::vector<ctype>* Aj,
    std::vector<ctype>* Bp,
    std::vector<ctype>* Bi,
    std::vector<ctype>* Bj,
    std::vector<ctype>* Cp,
    std::vector<ctype>* Ci,
    std::vector<ctype>* Cj,
    std::vector<ctype>* Sp,
    std::vector<ctype>* Si,
    std::vector<ctype>* Sj,
    std::vector<ctype>* Tp,
    std::vector<ctype>* Ti,
    std::vector<ctype>* Tj
};
%enddef

%define T_ARRAY_ARGOUT( ctype )
%apply std::vector<ctype>* array_argout {
    std::vector<ctype>* Ax, 
    std::vector<ctype>* Bx,
    std::vector<ctype>* Cx, 
    std::vector<ctype>* Sx,
    std::vector<ctype>* Tx, 
    std::vector<ctype>* Xx,
    std::vector<ctype>* Yx 
};
%enddef



 /*
  * INPLACE types
  */
%define I_INPLACE_ARRAY1( ctype )
%apply ctype * INPLACE_ARRAY {
  ctype Ap [],
  ctype Aj [],
  ctype Bp [],
  ctype Bj [],
  ctype Sp [],
  ctype Sj [],
  ctype Tp [],
  ctype Tj [],
  ctype  x [],
  ctype  y [],
  ctype  z [],
  ctype splitting [],
  ctype order [],
  ctype level [],
  ctype components [],
  ctype Id []
};
%enddef

%define T_INPLACE_ARRAY1( ctype )
%apply ctype * INPLACE_ARRAY {
  ctype   Ax [ ],
  ctype   Bx [ ],
  ctype   Sx [ ],
  ctype   Tx [ ],
  ctype    R [ ],
  ctype    x [ ],
  ctype    y [ ],
  ctype    z [ ],
  ctype temp [ ]
};
%enddef

/*%apply char * INPLACE_ARRAY {
    char splitting []
};*/


/*
 * Macros to instantiate index types and data types
 */
%define DECLARE_INDEX_TYPE( ctype )
I_IN_ARRAY1( ctype )
I_ARRAY_ARGOUT( ctype )
I_INPLACE_ARRAY1( ctype )
%enddef

%define DECLARE_DATA_TYPE( ctype )
T_IN_ARRAY1( ctype )
T_ARRAY_ARGOUT( ctype )
T_INPLACE_ARRAY1( ctype )
%enddef

/*
 * Create all desired index and data types here
 */
DECLARE_INDEX_TYPE( int )

DECLARE_DATA_TYPE( int    )
DECLARE_DATA_TYPE( float  )
DECLARE_DATA_TYPE( double )
DECLARE_DATA_TYPE( npy_cfloat_wrapper  )
DECLARE_DATA_TYPE( npy_cdouble_wrapper )


%include "ruge_stuben.h"
%include "smoothed_aggregation.h"
%include "relaxation.h"
%include "graph.h"
%include "evolution_strength.h"
%include "krylov.h"
%include "linalg.h"
 /*
  * Order may be important here, list float before double
  */
 
%define INSTANTIATE_INDEX( f_name )
%template(f_name)   f_name<int>;
%enddef

%define INSTANTIATE_DATA( f_name )
%template(f_name)   f_name<float>;
%template(f_name)   f_name<double>;
%enddef

%define INSTANTIATE_BOTH( f_name )
%template(f_name)   f_name<int,float>;
%template(f_name)   f_name<int,double>;
%enddef

%define INSTANTIATE_COMPLEX( f_name )
%template(f_name)   f_name<int,float,float>;
%template(f_name)   f_name<int,double,double>;
%template(f_name)   f_name<int,npy_cfloat_wrapper,float>;
%template(f_name)   f_name<int,npy_cdouble_wrapper,double>;
%enddef

%define INSTANTIATE_ALL( f_name )
%template(f_name)   f_name<int,int>;
%template(f_name)   f_name<int,float>;
%template(f_name)   f_name<int,double>;
%enddef
 
INSTANTIATE_INDEX(cljp_naive_splitting)
INSTANTIATE_INDEX(naive_aggregation)
INSTANTIATE_INDEX(standard_aggregation)
INSTANTIATE_INDEX(rs_cf_splitting)
INSTANTIATE_INDEX(rs_direct_interpolation_pass1)
INSTANTIATE_BOTH(rs_direct_interpolation_pass2)

INSTANTIATE_COMPLEX(satisfy_constraints_helper)
INSTANTIATE_COMPLEX(calc_BtB)
INSTANTIATE_COMPLEX(incomplete_mat_mult_bsr)

INSTANTIATE_COMPLEX(pinv_array)
INSTANTIATE_COMPLEX(classical_strength_of_connection)
INSTANTIATE_COMPLEX(symmetric_strength_of_connection)
INSTANTIATE_COMPLEX(evolution_strength_helper)
INSTANTIATE_COMPLEX(incomplete_mat_mult_csr)
INSTANTIATE_BOTH(apply_distance_filter)
INSTANTIATE_BOTH(apply_absolute_distance_filter)
INSTANTIATE_BOTH(min_blocks)

INSTANTIATE_COMPLEX(bsr_gauss_seidel)
INSTANTIATE_COMPLEX(bsr_jacobi)
INSTANTIATE_COMPLEX(gauss_seidel)
INSTANTIATE_COMPLEX(jacobi)
INSTANTIATE_COMPLEX(block_jacobi)
INSTANTIATE_COMPLEX(block_gauss_seidel)
INSTANTIATE_COMPLEX(gauss_seidel_indexed)
INSTANTIATE_COMPLEX(jacobi_ne)
INSTANTIATE_COMPLEX(gauss_seidel_ne)
INSTANTIATE_COMPLEX(gauss_seidel_nr)
INSTANTIATE_COMPLEX(overlapping_schwarz_csr)
INSTANTIATE_COMPLEX(extract_subblocks)

INSTANTIATE_COMPLEX(apply_householders)
INSTANTIATE_COMPLEX(householder_hornerscheme)
INSTANTIATE_COMPLEX(apply_givens)

%template(maximal_independent_set_serial)     maximal_independent_set_serial<int,int>;
%template(maximal_independent_set_parallel)   maximal_independent_set_parallel<int,int,double>;
%template(maximal_independent_set_k_parallel) maximal_independent_set_k_parallel<int,int,double>;
%template(vertex_coloring_mis)                vertex_coloring_mis<int,int>;
%template(vertex_coloring_jones_plassmann)    vertex_coloring_jones_plassmann<int,int,double>;
%template(vertex_coloring_LDF)                vertex_coloring_LDF<int,int,double>;

INSTANTIATE_INDEX(breadth_first_search)
INSTANTIATE_INDEX(connected_components)

INSTANTIATE_ALL(bellman_ford)
INSTANTIATE_ALL(lloyd_cluster)


%template(fit_candidates)   fit_candidates_real<int,float>;
%template(fit_candidates)   fit_candidates_real<int,double>;
%template(fit_candidates)   fit_candidates_complex<int,float,npy_cfloat_wrapper>;
%template(fit_candidates)   fit_candidates_complex<int,double,npy_cdouble_wrapper>;
/*INSTANTIATE_BOTH()*/

