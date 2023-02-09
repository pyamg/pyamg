// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "linalg.h"

namespace py = pybind11;

template<class I, class T, class F>
void _pinv_array(
      py::array_t<T> & AA,
                const I m,
                const I n,
        const char TransA
                 )
{
    auto py_AA = AA.mutable_unchecked();
    T *_AA = py_AA.mutable_data();

    return pinv_array<I, T, F>(
                      _AA, AA.shape(0),
                        m,
                        n,
                   TransA
                               );
}

template <class I, class T>
void _csc_scale_columns(
            const I n_row,
            const I n_col,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<T> & Xx
                        )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.mutable_unchecked();
    auto py_Xx = Xx.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    T *_Ax = py_Ax.mutable_data();
    const T *_Xx = py_Xx.data();

    return csc_scale_columns <I, T>(
                    n_row,
                    n_col,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Xx, Xx.shape(0)
                                    );
}

template <class I, class T>
void _csc_scale_rows(
            const I n_row,
            const I n_col,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<T> & Xx
                     )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.mutable_unchecked();
    auto py_Xx = Xx.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    T *_Ax = py_Ax.mutable_data();
    const T *_Xx = py_Xx.data();

    return csc_scale_rows <I, T>(
                    n_row,
                    n_col,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Xx, Xx.shape(0)
                                 );
}

template<class I, class T, class F>
void _filter_matrix_rows(
            const I n_row,
            const F theta,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
          const bool lump
                         )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    T *_Ax = py_Ax.mutable_data();

    return filter_matrix_rows<I, T, F>(
                    n_row,
                    theta,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                     lump
                                       );
}

PYBIND11_MODULE(linalg, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for linalg.h

    Methods
    -------
    signof
    signof
    signof
    conjugate
    conjugate
    conjugate
    conjugate
    real
    real
    real
    real
    imag
    imag
    imag
    imag
    mynorm
    mynorm
    mynorm
    mynorm
    mynormsq
    mynormsq
    mynormsq
    mynormsq
    zero_real
    zero_real
    zero_real
    zero_real
    zero_imag
    zero_imag
    zero_imag
    zero_imag
    pinv_array
    csc_scale_columns
    csc_scale_rows
    filter_matrix_rows
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("pinv_array", &_pinv_array<int, float, float>,
        py::arg("AA").noconvert(), py::arg("m"), py::arg("n"), py::arg("TransA"));
    m.def("pinv_array", &_pinv_array<int, double, double>,
        py::arg("AA").noconvert(), py::arg("m"), py::arg("n"), py::arg("TransA"));
    m.def("pinv_array", &_pinv_array<int, std::complex<float>, float>,
        py::arg("AA").noconvert(), py::arg("m"), py::arg("n"), py::arg("TransA"));
    m.def("pinv_array", &_pinv_array<int, std::complex<double>, double>,
        py::arg("AA").noconvert(), py::arg("m"), py::arg("n"), py::arg("TransA"),
R"pbdoc(
Replace each block of A with a Moore-Penrose pseudoinverse of that block.
Routine is designed to invert many small matrices at once.

Parameters
----------
AA : array
    (m, n, n) array, assumed to be "raveled" and in row major form
m,n : int
    dimensions of AA
TransA : char
    'T' or 'F'.  Decides whether to transpose each nxn block
    of A before inverting.  If using Python array, should be 'T'.

Returns
-------
AA : array
    AA is modified in place with the pseduoinverse replacing each
    block of AA.  AA is returned in row-major form for Python

Notes
-----
This routine is designed to be called once for a large m.
Calling this routine repeatably would not be efficient.

This function offers substantial speedup over native Python
code for many small matrices, e.g. 5x5 and 10x10.  Tests have
indicated that matrices larger than 27x27 are faster if done
in native Python.

Examples
--------
>>> from pyamg.amg_core import pinv_array
>>> from scipy import arange, ones, array, dot
>>> A = array([arange(1,5, dtype=float).reshape(2,2), ones((2,2),dtype=float)])
>>> Ac = A.copy()
>>> pinv_array(A, 2, 2, 'T')
>>> print "Multiplication By Inverse\n" + str(dot(A[0], Ac[0]))
>>> print "Multiplication by PseudoInverse\n" + str(dot(Ac[1], dot(A[1], Ac[1])))
>>>
>>> A = Ac.copy()
>>> pinv_array(A,2,2,'F')
>>> print "Changing flag to \'F\' results in different Inverse\n" + str(dot(A[0], Ac[0]))
>>> print "A holds the inverse of the transpose\n" + str(dot(A[0], Ac[0].T)))pbdoc");

    m.def("csc_scale_columns", &_csc_scale_columns<int, int>,
        py::arg("n_row"), py::arg("n_col"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Xx").noconvert());
    m.def("csc_scale_columns", &_csc_scale_columns<int, float>,
        py::arg("n_row"), py::arg("n_col"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Xx").noconvert());
    m.def("csc_scale_columns", &_csc_scale_columns<int, double>,
        py::arg("n_row"), py::arg("n_col"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Xx").noconvert(),
R"pbdoc(
Scale the columns of a CSC matrix *in place*

..
  A[:,i] *= X[i]

References
----------
https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h)pbdoc");

    m.def("csc_scale_rows", &_csc_scale_rows<int, int>,
        py::arg("n_row"), py::arg("n_col"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Xx").noconvert());
    m.def("csc_scale_rows", &_csc_scale_rows<int, float>,
        py::arg("n_row"), py::arg("n_col"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Xx").noconvert());
    m.def("csc_scale_rows", &_csc_scale_rows<int, double>,
        py::arg("n_row"), py::arg("n_col"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Xx").noconvert(),
R"pbdoc(
Scale the rows of a CSC matrix *in place*

..
  A[i,:] *= X[i]

References
----------
https://github.com/scipy/scipy/blob/master/scipy/sparse/sparsetools/csr.h)pbdoc");

    m.def("filter_matrix_rows", &_filter_matrix_rows<int, float, float>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("lump"));
    m.def("filter_matrix_rows", &_filter_matrix_rows<int, double, double>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("lump"));
    m.def("filter_matrix_rows", &_filter_matrix_rows<int, std::complex<float>, float>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("lump"));
    m.def("filter_matrix_rows", &_filter_matrix_rows<int, std::complex<double>, double>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("lump"),
R"pbdoc(
Filter matrix rows by diagonal entry, that is set A_ij = 0 if::

   |A_ij| < theta * |A_ii|

Parameters
----------
num_rows : int
    number of rows in A
theta : float
    stength of connection tolerance
Ap : array
    CSR row pointer
Aj : array
    CSR index array
Ax : array
    CSR data array

Returns
-------
Nothing, Ax is modified in place)pbdoc");

}

