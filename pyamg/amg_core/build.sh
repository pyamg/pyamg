#! /bin/bash

flist=(
evolution_strength.h
graph.h
krylov.h
linalg.h
relaxation.h
ruge_stuben.h
smoothed_aggregation.h
sparse.h);

for f in "${flist[@]}"; do
    echo "[Building ${f%.*}_bind.cpp]"
    g++ -O3 -shared -std=c++11 -I /usr/local/Cellar/pybind11/2.1.1/include `python3-config --cflags --ldflags` ${f%.*}_bind.cpp -o ${f%.*}.so
done

