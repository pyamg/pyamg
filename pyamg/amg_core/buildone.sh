#! /bin/bash

f=$@
echo "[Building ${f%.*}_bind.cpp]"
g++ -O3 -shared -std=c++11 -I /usr/local/Cellar/pybind11/2.1.1/include `python3-config --cflags --ldflags` ${f%.*}_bind.cpp -o ${f%.*}.so
