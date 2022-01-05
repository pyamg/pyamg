#!/bin/bash

# usage:
# valgrind_setup.sh

echo "getting pybind11 suppressions"
curl -s https://raw.githubusercontent.com/python/cpython/main/Misc/valgrind-python.supp > valgrind-python.supp
curl -s https://raw.githubusercontent.com/pybind/pybind11/master/tests/valgrind-numpy-scipy.supp > valgrind-numpy-scipy.supp

echo "removing ###"
sed -i 's/^###//' valgrind-python.supp
sed -i 's/\(^ All the suppress.*\)//' valgrind-python.supp
sed -i 's/\(^ that Python uses.*\)//' valgrind-python.supp
sed -i 's/\(^ use of the libra.*\)//' valgrind-python.supp
sed -i 's/\(^ These occur from.*\)//' valgrind-python.supp
sed -i 's/\(^  test_socket_sll.*\)//' valgrind-python.supp

echo "generating numpy suppressions"
export PYTHONMALLOC=malloc; valgrind \
    --leak-check=full \
    --log-file="valgrind-numpy-extras.txt" \
    --show-leak-kinds=definite,indirect \
    --gen-suppressions=all \
    --suppressions="valgrind-numpy-scipy.supp" \
    python -c "import numpy"

echo "seding numpy suppressions to a .supp file"
# https://stackoverflow.com/questions/5972908/print-text-between-sed
sed -n '/{/,/}/{:a; $!N; /}/!{$!ba}; s/.*\({[^}]*}\).*/\1\n/p}' valgrind-numpy-extras.txt > valgrind-extras.supp

echo "generating scipy suppressions"
export PYTHONMALLOC=malloc; valgrind \
    --leak-check=full \
    --log-file="valgrind-scipy-extras.txt" \
    --show-leak-kinds=definite,indirect \
    --gen-suppressions=all \
    --suppressions="valgrind-numpy-scipy.supp" \
    --suppressions="valgrind-extras.supp" \
    python -c "from scipy import sparse; from scipy import linalg; from scipy import special; from scipy import io"

echo "seding scipy suppressions to a .supp file"
# https://stackoverflow.com/questions/5972908/print-text-between-sed
sed -n '/{/,/}/{:a; $!N; /}/!{$!ba}; s/.*\({[^}]*}\).*/\1\n/p}' valgrind-scipy-extras.txt >> valgrind-extras.supp

echo "generating pytest suppressions"
export PYTHONMALLOC=malloc; valgrind \
    --leak-check=full \
    --log-file="valgrind-pytest-extras.txt" \
    --show-leak-kinds=definite,indirect \
    --gen-suppressions=all \
    --suppressions="valgrind-numpy-scipy.supp" \
    --suppressions="valgrind-extras.supp" \
    python -m pytest test_lloyd_blank.py

echo "seding pytest suppressions to a .supp file"
# https://stackoverflow.com/questions/5972908/print-text-between-sed
sed -n '/{/,/}/{:a; $!N; /}/!{$!ba}; s/.*\({[^}]*}\).*/\1\n/p}' valgrind-pytest-extras.txt >> valgrind-extras.supp
