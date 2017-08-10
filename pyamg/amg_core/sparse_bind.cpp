// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "sparse.h"

namespace py = pybind11;

PYBIND11_PLUGIN(sparse) {
    py::module m("sparse", R"pbdoc(
    pybind11 bindings for sparse.h

    Methods
    -------
    )pbdoc");

    return m.ptr();
}

