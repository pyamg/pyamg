/* -*- C -*-  (not really, but good for syntax highlighting) */
%module amg_core

%{
#define SWIG_FILE_WITH_INIT
#include "complex_ops.h"

#include "linalg.h"
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
    (ctype J [], const int J_size)
};
%enddef

%define T_INPLACE_ARRAY1( ctype )
%apply (ctype* INPLACE_ARRAY1, int DIM1) {
    (ctype Ax [], const int Ax_size)
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
