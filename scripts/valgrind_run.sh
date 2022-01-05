#!/bin/bash

# usage:
# checkvalgrind.sh somescript.py

#--show-leak-kinds=definite,indirect \
#--errors-for-leak-kinds=definite,indirect \
#--error-exitcode=1 \
#--read-var-info=yes \
#--gen-suppressions=all \ 
#   --log-file="output.txt" \
export PYTHONMALLOC=malloc; valgrind \
    --track-origins=yes \
    --show-leak-kinds=definite,indirect \
    --leak-check=full \
    --suppressions="valgrind-numpy-scipy.supp" \
    --suppressions="valgrind-python.supp" \
    --suppressions="valgrind-extras.supp" \
    python -m pytest $1
