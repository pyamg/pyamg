Before regenerating amg_core_wrap.cxx with SWIG, ensure that you
are using SWIG Version 3.0 or newer (check with: 'swig -version')

The wrappers are generated with:
   swig -c++ -python amg_core.i
or
   swig -c++ -python -w509 -py3 amg_core.i


Try to keep numpy.i and pyfragments.swg:
https://github.com/numpy/numpy/tree/master/tools/swig
