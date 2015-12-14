Before regenerating amg_core_wrap.cxx with SWIG, ensure that you
are using SWIG Version 3.0 or newer (check with: 'swig -version')

The wrappers are generated with:
   swig -c++ -python amg_core.i
or
   swig -c++ -python -w509 -py3 amg_core.i


Try to keep numpy.i and pyfragments.swg:
https://github.com/numpy/numpy/tree/master/tools/swig

Use pep8 to fix safely
autopep8 --select=E302 --in-place amg_core.py
autopep8 --select=E303 --in-place amg_core.py
autopep8 --select=W391 --in-place amg_core.py

For 1.6 compatibility add:

#ifndef NPY_ARRAY_F_CONTIGUOUS
#define NPY_ARRAY_F_CONTIGUOUS NPY_F_CONTIGUOUS
#endif

right after the
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

in amg_core_wrap.cxx
