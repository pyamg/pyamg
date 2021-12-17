#!/bin/bash

#--show-leak-kinds=definite,indirect \
#--errors-for-leak-kinds=definite,indirect \
#--error-exitcode=1 \
#--read-var-info=yes \
# --suppressions="valgrind-numpy-scipy.supp" \
#--gen-suppressions=all \ 
wget https://raw.githubusercontent.com/python/cpython/main/Misc/valgrind-python.supp
wget https://raw.githubusercontent.com/pybind/pybind11/master/tests/valgrind-numpy-scipy.supp
export PYTHONMALLOC=malloc; valgrind \
    --track-origins=yes \
    --suppressions="valgrind-numpy-scipy.supp" \
    --suppressions="valgrind-python.supp" \
    --log-file="output.txt" \
    python $1
