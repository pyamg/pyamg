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
