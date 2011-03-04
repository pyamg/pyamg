/* -*- C -*-  (not really, but good for syntax highlighting) */
%module splinalg

 /* why does SWIG complain about int arrays? a typecheck is provided */
#pragma SWIG nowarn=467

%{
#define SWIG_FILE_WITH_INIT
#include "numpy/arrayobject.h"

#include "complex_ops.h"
#include "splinalg.h"
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
    const ctype Aj [ ]
};
%enddef

%define T_IN_ARRAY1( ctype )
%apply ctype * IN_ARRAY1 {
    const ctype Ax [ ],
    const ctype  b [ ]
};
%enddef

 /*
  * INPLACE types
  */
%define T_INPLACE_ARRAY1( ctype )
%apply ctype * INPLACE_ARRAY {
  ctype    x [ ]
};
%enddef

/*
 * Macros to instantiate index types and data types
 */
%define DECLARE_INDEX_TYPE( ctype )
I_IN_ARRAY1( ctype )
%enddef

%define DECLARE_DATA_TYPE( ctype )
T_IN_ARRAY1( ctype )
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

%include "splinalg.h"

%define INSTANTIATE( f_name )
%template( f_name )  f_name<int,float>;
%template( f_name )  f_name<int,double>;
%template( f_name )  f_name<int,npy_cfloat_wrapper>;
%template( f_name )  f_name<int,npy_cdouble_wrapper>;
%enddef

INSTANTIATE(backsolve)
INSTANTIATE(forwardsolve)
