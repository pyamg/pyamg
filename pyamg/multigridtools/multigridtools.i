/* -*- C -*-  (not really, but good for syntax highlighting) */
%module multigridtools

 /* why does SWIG complain about int arrays? a typecheck is provided */
#pragma SWIG nowarn=467

%{
#define SWIG_FILE_WITH_INIT
#include "numpy/arrayobject.h"

#include "ruge_stuben.h"
#include "smoothed_aggregation.h"
#include "relaxation.h"
#include "graph.h"
#include "ode_strength.h"
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
    const ctype  B [ ]    
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


%include "ruge_stuben.h"
%include "smoothed_aggregation.h"
%include "relaxation.h"
%include "graph.h"
%include "ode_strength.h"
 /*
  * Order may be important here, list float before double
  */

%define INSTANTIATE_BOTH( f_name )
%template(f_name)   f_name<int,float>;
%template(f_name)   f_name<int,double>;
/* 64-bit indices would go here */
%enddef
 
%define INSTANTIATE_INDEX( f_name )
%template(f_name)   f_name<int>;
%enddef

%define INSTANTIATE_DATA( f_name )
%template(f_name)   f_name<float>;
%template(f_name)   f_name<double>;
%enddef

%define INSTANTIATE_BOTH2( f_name )
%template(f_name)   f_name<int,int>;
%template(f_name)   f_name<int,float>;
%template(f_name)   f_name<int,double>;
%enddef
 
 
INSTANTIATE_INDEX(standard_aggregation)
INSTANTIATE_INDEX(rs_cf_splitting)
INSTANTIATE_INDEX(rs_direct_interpolation_pass1)
INSTANTIATE_BOTH(rs_direct_interpolation_pass2)

INSTANTIATE_BOTH(fit_candidates)

INSTANTIATE_BOTH(satisfy_constraints_helper)
INSTANTIATE_BOTH(invert_BtB)
INSTANTIATE_BOTH(min_blocks)

INSTANTIATE_BOTH(classical_strength_of_connection)
INSTANTIATE_BOTH(symmetric_strength_of_connection)
INSTANTIATE_BOTH(apply_distance_filter)
INSTANTIATE_BOTH(ode_strength_helper)

INSTANTIATE_BOTH(block_gauss_seidel)
INSTANTIATE_BOTH(gauss_seidel)
INSTANTIATE_BOTH(jacobi)
INSTANTIATE_BOTH(gauss_seidel_indexed)
INSTANTIATE_BOTH(kaczmarz_jacobi)
INSTANTIATE_BOTH(kaczmarz_gauss_seidel)

%template(maximal_independent_set_serial)     maximal_independent_set_serial<int,int>;
%template(maximal_independent_set_parallel)   maximal_independent_set_parallel<int,int,double>;
%template(maximal_independent_set_k_parallel) maximal_independent_set_k_parallel<int,int,double>;
%template(vertex_coloring_mis)                vertex_coloring_mis<int,int>;
%template(vertex_coloring_jones_plassmann)    vertex_coloring_jones_plassmann<int,int,double>;
%template(vertex_coloring_LDF)                vertex_coloring_LDF<int,int,double>;

INSTANTIATE_INDEX(breadth_first_search)
INSTANTIATE_INDEX(connected_components)

INSTANTIATE_BOTH2(bellman_ford)
INSTANTIATE_BOTH2(lloyd_cluster)

