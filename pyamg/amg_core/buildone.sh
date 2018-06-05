#! /bin/bash

f=$@
echo "[Building ${f%.*}_bind.cpp]"
/usr/local/bin/g++-8 -O3 -shared -std=c++11 -I /usr/local/Cellar/pybind11/2.1.3/include `python3-config --cflags --ldflags` ${f%.*}_bind.cpp -o ${f%.*}.so
