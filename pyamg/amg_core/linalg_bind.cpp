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
                      _AA, AA.size(),
                        m,
                        n,
                   TransA
                               );
}

PYBIND11_MODULE(linalg, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for linalg.h

    Methods
    -------
    pinv_array
    )pbdoc";

    m.def("pinv_array", &_pinv_array<int, float, float>,
        py::arg("AA").noconvert(), py::arg("m"), py::arg("n"), py::arg("TransA"));
    m.def("pinv_array", &_pinv_array<int, double, double>,
        py::arg("AA").noconvert(), py::arg("m"), py::arg("n"), py::arg("TransA"));
    m.def("pinv_array", &_pinv_array<int, std::complex<float>, float>,
        py::arg("AA").noconvert(), py::arg("m"), py::arg("n"), py::arg("TransA"));
    m.def("pinv_array", &_pinv_array<int, std::complex<double>, double>,
        py::arg("AA").noconvert(), py::arg("m"), py::arg("n"), py::arg("TransA"),
R"pbdoc(
/* Replace each block of A with a Moore-Penrose pseudoinverse of that block.
 * Routine is designed to invert many small matrices at once.
 * Parameters
 * ----------
 * AA : {float|complex array}
 *      (m, n, n) array, assumed to be "raveled" and in row major form
 * m,n : int
 *      dimensions of AA
 * TransA : char
 *      'T' or 'F'.  Decides whether to transpose each nxn block
 *      of A before inverting.  If using Python array, should be 'T'.
 *
 * Return
 * ------
 * AA : {array}
 *      AA is modified in place with the pseduoinverse replacing each
 *      block of AA.  AA is returned in row-major form for Python
 *
 * Notes
 * -----
 * This routine is designed to be called once for a large m.
 * Calling this routine repeatably would not be efficient.
 *
 * This function offers substantial speedup over native Python
 * code for many small matrices, e.g. 5x5 and 10x10.  Tests have
 * indicated that matrices larger than 27x27 are faster if done
 * in native Python.
 *
 * Examples
 * --------
 * >>> from pyamg.amg_core import pinv_array
 * >>> from scipy import arange, ones, array, dot
 * >>> A = array([arange(1,5, dtype=float).reshape(2,2), ones((2,2),dtype=float)])
 * >>> Ac = A.copy()
 * >>> pinv_array(A, 2, 2, 'T')
 * >>> print "Multiplication By Inverse\n" + str(dot(A[0], Ac[0]))
 * >>> print "Multiplication by PseudoInverse\n" + str(dot(Ac[1], dot(A[1], Ac[1])))
 * >>>
 * >>> A = Ac.copy()
 * >>> pinv_array(A,2,2,'F')
 * >>> print "Changing flag to \'F\' results in different Inverse\n" + str(dot(A[0], Ac[0]))
 * >>> print "A holds the inverse of the transpose\n" + str(dot(A[0], Ac[0].T))
 *
 */
)pbdoc");

}

