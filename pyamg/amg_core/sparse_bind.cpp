// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "sparse.h"

namespace py = pybind11;

template <class I, class T>
void _csr_matvec(
     I n_row,
     I n_col,
     py::array_t<I> & Ap,
     py::array_t<I> & Aj,
     py::array_t<T> & Ax,
     py::array_t<T> & Xx,
     py::array_t<T> & Yx)
{
auto py_Ap = Ap.unchecked();
auto py_Aj = Aj.unchecked();
auto py_Ax = Ax.unchecked();
auto py_Xx = Xx.unchecked();
auto py_Yx = Yx.mutable_unchecked();

const I *_Ap = py_Ap.data();
const I *_Aj = py_Aj.data();
const T *_Ax = py_Ax.data();
const T *_Xx = py_Xx.data();
      T *_Yx = py_Yx.mutable_data();

 return csr_matvec <I, T>(
     n_row,
     n_col,
     _Ap, Ap.size(),
     _Aj, Aj.size(),
     _Ax, Ax.size(),
     _Xx, Xx.size(),
     _Yx, Yx.size());
}




PYBIND11_PLUGIN(sparse) {
    py::module m("sparse", R"pbdoc(
    pybind11 bindings for sparse.h

    Methods
    -------
    csr_matvec
    )pbdoc");

    m.def("csr_matvec", &_csr_matvec<int, float>,
        py::arg("n_row"), py::arg("n_col"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Xx").noconvert(), py::arg("Yx").noconvert());
    m.def("csr_matvec", &_csr_matvec<int, double>,
        py::arg("n_row"), py::arg("n_col"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Xx").noconvert(), py::arg("Yx").noconvert());
    m.def("csr_matvec", &_csr_matvec<int, std::complex<float>>,
        py::arg("n_row"), py::arg("n_col"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Xx").noconvert(), py::arg("Yx").noconvert());
    m.def("csr_matvec", &_csr_matvec<int, std::complex<double>>,
        py::arg("n_row"), py::arg("n_col"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Xx").noconvert(), py::arg("Yx").noconvert(),
R"pbdoc(

Compute Y += A*X for CSR matrix A and dense vectors X,Y


Input Arguments:
I  n_row         - number of rows in A
I  n_col         - number of columns in A
I  Ap[n_row+1]   - row pointer
I  Aj[nnz(A)]    - column indices
T  Ax[nnz(A)]    - nonzeros
T  Xx[n_col]     - input vector

Output Arguments:
T  Yx[n_row]     - output vector

Note:
Output array Yx must be preallocated

Complexity: Linear.  Specifically O(nnz(A) + n_row)


)pbdoc");

    return m.ptr();
}

