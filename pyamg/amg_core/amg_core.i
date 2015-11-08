/* -*- C -*-  (not really, but good for syntax highlighting) */
%module amg_core
%ignorewarn("509:") operator=;

%{
#define SWIG_FILE_WITH_INIT

#include "relaxation.h"
#include "krylov.h"
#include "linalg.h"
#include "graph.h"
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
    (const ctype Tp [], const int Tp_size),
    (const ctype Sp [], const int Sp_size),
    (const ctype Aj [], const int Aj_size),
    (const ctype Sj [], const int Sj_size),
    (      ctype order [], const int order_size),
    (      ctype level [], const int level_size),
    (      ctype components [], const int components_size),
    (const ctype Id [], const int Id_size)
};
%enddef

%define T_INPLACE_ARRAY1( ctype )
%apply (ctype* INPLACE_ARRAY1, int DIM1) {
    (const ctype  B [], const int  B_size),
    (const ctype  b [], const int  b_size),
    (      ctype  w [], const int  w_size),
    (      ctype  x [], const int  x_size),
    (const ctype  y [], const int  y_size),
    (      ctype  z [], const int  z_size),
    (const ctype Ax [], const int Ax_size),
    (const ctype Tx [], const int Tx_size),
    (      ctype Tx [], const int Tx_size),
    (      ctype AA [], const int AA_size),
    (      ctype temp [], const int temp_size),
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
