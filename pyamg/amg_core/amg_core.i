/* -*- C -*-  (not really, but good for syntax highlighting) */
%module amg_core
%ignorewarn("509:") operator=;

%{
#define SWIG_FILE_WITH_INIT

#include "linalg.h"
#include "graph.h"
#include "krylov.h"
#include "relaxation.h"
#include "smoothed_aggregation.h"
#include "ruge_stuben.h"
#include "evolution_strength.h"
%}

%feature("autodoc", "1");

%include "numpy.i"

%init %{
    import_array();
%}

/*
 * INPLACE types
 */
%define I_INPLACE_ARRAY1( ctype )
%apply (ctype* INPLACE_ARRAY1, int DIM1) {
    (const ctype Ap [], const int Ap_size),
    (const ctype Bp [], const int Bp_size),
    (      ctype Bp [], const int Bp_size),
    (const ctype Tp [], const int Tp_size),
    (const ctype Sp [], const int Sp_size),
    (      ctype Sp [], const int Sp_size),
    (const ctype Ai [], const int Ai_size),
    (const ctype Aj [], const int Aj_size),
    (const ctype Bj [], const int Bj_size),
    (      ctype Bj [], const int Bj_size),
    (const ctype Sj [], const int Sj_size),
    (      ctype Sj [], const int Sj_size),
    (const ctype Tj [], const int Tj_size),
    (      ctype order [], const int order_size),
    (      ctype level [], const int level_size),
    (      ctype components [], const int components_size),
    (const ctype Id [], const int Id_size),
    (const ctype splitting [], const int splitting_size),
    (      ctype splitting [], const int splitting_size),
    (      ctype indices [], const int indices_size),
    (const ctype A_rowptr [], const int A_rowptr_size),
    (const ctype A_colinds [], const int A_colinds_size)

};
%enddef

%define T_INPLACE_ARRAY1( ctype )
%apply (ctype* INPLACE_ARRAY1, int DIM1) {
    (const ctype  B [], const int  B_size),
    (const ctype  b [], const int  b_size),
    (      ctype  e [], const int  e_size),
    (      ctype  w [], const int  w_size),
    (const ctype  x [], const int  x_size),
    (      ctype  x [], const int  x_size),
    (const ctype  y [], const int  y_size),
    (      ctype  y [], const int  y_size),
    (const ctype  z [], const int  z_size),
    (      ctype  z [], const int  z_size),
    (const ctype Sx [], const int Sx_size),
    (      ctype Sx [], const int Sx_size),
    (const ctype Ax [], const int Ax_size),
    (      ctype Ax [], const int Ax_size),
    (const ctype Bx [], const int Bx_size),
    (      ctype Bx [], const int Bx_size),
    (const ctype Tx [], const int Tx_size),
    (      ctype Tx [], const int Tx_size),
    (      ctype AA [], const int AA_size),
    (      ctype  R [], const int  R_size),
    (      ctype temp [], const int temp_size),
    (      ctype gamma [], const int gamma_size),
    (const ctype omega [], const int omega_size)
};
%enddef

/*
 * Macros to instantiate index types and data types
 */
%define DECLARE_INDEX_TYPE( ctype )
I_INPLACE_ARRAY1( ctype )
%enddef

%define DECLARE_DATA_TYPE( ctype )
T_INPLACE_ARRAY1( ctype )
%enddef

/*
 * Create all desired index and data types here
 */
DECLARE_INDEX_TYPE( int )

DECLARE_DATA_TYPE( int    )
DECLARE_DATA_TYPE( float  )
DECLARE_DATA_TYPE( double )
DECLARE_DATA_TYPE( std::complex<float>  )
DECLARE_DATA_TYPE( std::complex<double> )

/*
* Order may be important here, list float before double
*/
%define INSTANTIATE_INDEX_ONLY( f_name )
%template(f_name)   f_name<int>;
%enddef

%define INSTANTIATE_DATA_ONLY( f_name )
%template(f_name)   f_name<float>;
%template(f_name)   f_name<double>;
%enddef

%define INSTANTIATE_INDEXDATA( f_name )
%template(f_name)   f_name<int,float>;
%template(f_name)   f_name<int,double>;
%enddef

%define INSTANTIATE_INDEXDATA_INT( f_name )
%template(f_name)   f_name<int,int>;
%template(f_name)   f_name<int,float>;
%template(f_name)   f_name<int,double>;
%enddef

%define INSTANTIATE_INDEXDATA_COMPLEX( f_name )
%template(f_name)   f_name<int,float,float>;
%template(f_name)   f_name<int,double,double>;
%template(f_name)   f_name<int,std::complex<float>,float>;
%template(f_name)   f_name<int,std::complex<double>,double>;
%enddef

/*----------------------------------------------------------------------------
  linalg.h
  ---------------------------------------------------------------------------*/
%include "linalg.h"
INSTANTIATE_INDEXDATA_COMPLEX(pinv_array)

/*----------------------------------------------------------------------------
  graph.h
  ---------------------------------------------------------------------------*/
%include "graph.h"

%template(maximal_independent_set_serial)     maximal_independent_set_serial<int,int>;
%template(maximal_independent_set_parallel)   maximal_independent_set_parallel<int,int,double>;
%template(maximal_independent_set_k_parallel) maximal_independent_set_k_parallel<int,int,double>;

%template(vertex_coloring_mis)                vertex_coloring_mis<int,int>;
%template(vertex_coloring_jones_plassmann)    vertex_coloring_jones_plassmann<int,int,double>;
%template(vertex_coloring_LDF)                vertex_coloring_LDF<int,int,double>;

INSTANTIATE_INDEXDATA_INT(bellman_ford)
INSTANTIATE_INDEXDATA_INT(lloyd_cluster)
INSTANTIATE_INDEX_ONLY(breadth_first_search)
INSTANTIATE_INDEX_ONLY(connected_components)

/*----------------------------------------------------------------------------
  krylov.h
  ---------------------------------------------------------------------------*/
%include "krylov.h"

INSTANTIATE_INDEXDATA_COMPLEX(apply_householders)
INSTANTIATE_INDEXDATA_COMPLEX(householder_hornerscheme)
INSTANTIATE_INDEXDATA_COMPLEX(apply_givens)

/*----------------------------------------------------------------------------
  relaxation.h
  ---------------------------------------------------------------------------*/
%include "relaxation.h"

INSTANTIATE_INDEXDATA_COMPLEX(gauss_seidel)
INSTANTIATE_INDEXDATA_COMPLEX(bsr_gauss_seidel)
INSTANTIATE_INDEXDATA_COMPLEX(jacobi)
INSTANTIATE_INDEXDATA_COMPLEX(bsr_jacobi)
INSTANTIATE_INDEXDATA_COMPLEX(gauss_seidel_indexed)
INSTANTIATE_INDEXDATA_COMPLEX(jacobi_ne)
INSTANTIATE_INDEXDATA_COMPLEX(gauss_seidel_nr)
INSTANTIATE_INDEXDATA_COMPLEX(gauss_seidel_ne)
INSTANTIATE_INDEXDATA_COMPLEX(block_jacobi)
INSTANTIATE_INDEXDATA_COMPLEX(block_gauss_seidel)
INSTANTIATE_INDEXDATA_COMPLEX(extract_subblocks)
INSTANTIATE_INDEXDATA_COMPLEX(overlapping_schwarz_csr)

/*----------------------------------------------------------------------------
  smoothed_aggregation.h
  ---------------------------------------------------------------------------*/
%include "smoothed_aggregation.h"

INSTANTIATE_INDEXDATA_COMPLEX(symmetric_strength_of_connection)
INSTANTIATE_INDEX_ONLY(naive_aggregation)
INSTANTIATE_INDEX_ONLY(standard_aggregation)

%template(fit_candidates)   fit_candidates_real<int,float>;
%template(fit_candidates)   fit_candidates_real<int,double>;
%template(fit_candidates)   fit_candidates_complex<int,float,std::complex<float>>;
%template(fit_candidates)   fit_candidates_complex<int,double,std::complex<double>>;

INSTANTIATE_INDEXDATA_COMPLEX(satisfy_constraints_helper)
INSTANTIATE_INDEXDATA_COMPLEX(calc_BtB)
INSTANTIATE_INDEXDATA_COMPLEX(incomplete_mat_mult_bsr)
INSTANTIATE_INDEXDATA_COMPLEX(truncate_rows_csr)

/*----------------------------------------------------------------------------
  ruge_stuben.h
  ---------------------------------------------------------------------------*/
%include "ruge_stuben.h"

INSTANTIATE_INDEXDATA_COMPLEX(classical_strength_of_connection)

INSTANTIATE_INDEXDATA_COMPLEX(maximum_row_value)

INSTANTIATE_INDEX_ONLY(rs_cf_splitting)
INSTANTIATE_INDEX_ONLY(cljp_naive_splitting)
INSTANTIATE_INDEX_ONLY(rs_direct_interpolation_pass1)
INSTANTIATE_INDEXDATA(rs_direct_interpolation_pass2)
INSTANTIATE_INDEXDATA(cr_helper)

/*----------------------------------------------------------------------------
  evolution_strength.h
  ---------------------------------------------------------------------------*/
%include "evolution_strength.h"

INSTANTIATE_INDEXDATA(apply_distance_filter)
INSTANTIATE_INDEXDATA(apply_absolute_distance_filter)
INSTANTIATE_INDEXDATA(min_blocks)

INSTANTIATE_INDEXDATA_COMPLEX(evolution_strength_helper)
INSTANTIATE_INDEXDATA_COMPLEX(incomplete_mat_mult_csr)
