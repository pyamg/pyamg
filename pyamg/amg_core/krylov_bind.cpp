// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "krylov.h"

namespace py = pybind11;

template<class I, class T, class F>
void _apply_householders(
       py::array_t<T> & z,
       py::array_t<T> & B,
                const I n,
            const I start,
             const I stop,
             const I step
                         )
{
    auto py_z = z.mutable_unchecked();
    auto py_B = B.unchecked();
    T *_z = py_z.mutable_data();
    const T *_B = py_B.data();

    return apply_householders<I, T, F>(
                       _z, z.shape(0),
                       _B, B.shape(0),
                        n,
                    start,
                     stop,
                     step
                                       );
}

template<class I, class T, class F>
void _householder_hornerscheme(
       py::array_t<T> & z,
       py::array_t<T> & B,
       py::array_t<T> & y,
                const I n,
            const I start,
             const I stop,
             const I step
                               )
{
    auto py_z = z.mutable_unchecked();
    auto py_B = B.unchecked();
    auto py_y = y.unchecked();
    T *_z = py_z.mutable_data();
    const T *_B = py_B.data();
    const T *_y = py_y.data();

    return householder_hornerscheme<I, T, F>(
                       _z, z.shape(0),
                       _B, B.shape(0),
                       _y, y.shape(0),
                        n,
                    start,
                     stop,
                     step
                                             );
}

template<class I, class T, class F>
void _apply_givens(
       py::array_t<T> & B,
       py::array_t<T> & x,
                const I n,
             const I nrot
                   )
{
    auto py_B = B.unchecked();
    auto py_x = x.mutable_unchecked();
    const T *_B = py_B.data();
    T *_x = py_x.mutable_data();

    return apply_givens<I, T, F>(
                       _B, B.shape(0),
                       _x, x.shape(0),
                        n,
                     nrot
                                 );
}

PYBIND11_MODULE(krylov, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for krylov.h

    Methods
    -------
    apply_householders
    householder_hornerscheme
    apply_givens
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("apply_householders", &_apply_householders<int, float, float>,
        py::arg("z").noconvert(), py::arg("B").noconvert(), py::arg("n"), py::arg("start"), py::arg("stop"), py::arg("step"));
    m.def("apply_householders", &_apply_householders<int, double, double>,
        py::arg("z").noconvert(), py::arg("B").noconvert(), py::arg("n"), py::arg("start"), py::arg("stop"), py::arg("step"));
    m.def("apply_householders", &_apply_householders<int, std::complex<float>, float>,
        py::arg("z").noconvert(), py::arg("B").noconvert(), py::arg("n"), py::arg("start"), py::arg("stop"), py::arg("step"));
    m.def("apply_householders", &_apply_householders<int, std::complex<double>, double>,
        py::arg("z").noconvert(), py::arg("B").noconvert(), py::arg("n"), py::arg("start"), py::arg("stop"), py::arg("step"),
R"pbdoc(
Apply Householder reflectors in B to z

Implements the below python

.. code-block:: python

    for j in range(start,stop,step):
      z = z - 2.0*dot(conjugate(B[j,:]), v)*B[j,:]

Parameters
----------
z : array
    length n vector to be operated on
B : array
    n x m matrix of householder reflectors
    must be in row major form
n : int
    dimensionality of z
start, stop, step : int
    control the choice of vectors in B to use

Returns
-------
z is modified in place to reflect the application of
the Householder reflectors, B[:,range(start,stop,step)]

Notes
-----
Principle calling routine is gmres(...) and fgmres(...) in krylov.py)pbdoc");

    m.def("householder_hornerscheme", &_householder_hornerscheme<int, float, float>,
        py::arg("z").noconvert(), py::arg("B").noconvert(), py::arg("y").noconvert(), py::arg("n"), py::arg("start"), py::arg("stop"), py::arg("step"));
    m.def("householder_hornerscheme", &_householder_hornerscheme<int, double, double>,
        py::arg("z").noconvert(), py::arg("B").noconvert(), py::arg("y").noconvert(), py::arg("n"), py::arg("start"), py::arg("stop"), py::arg("step"));
    m.def("householder_hornerscheme", &_householder_hornerscheme<int, std::complex<float>, float>,
        py::arg("z").noconvert(), py::arg("B").noconvert(), py::arg("y").noconvert(), py::arg("n"), py::arg("start"), py::arg("stop"), py::arg("step"));
    m.def("householder_hornerscheme", &_householder_hornerscheme<int, std::complex<double>, double>,
        py::arg("z").noconvert(), py::arg("B").noconvert(), py::arg("y").noconvert(), py::arg("n"), py::arg("start"), py::arg("stop"), py::arg("step"),
R"pbdoc(
For use after gmres is finished iterating and the least squares
solution has been found.  This routine maps the solution back to
the original space via the Householder reflectors.

Apply Householder reflectors in B to z
while also adding in the appropriate value from y, so
that we follow the Horner-like scheme to map our least squares
solution in y back to the original space

Implements the below python

.. code-block:: python

    for j in range(inner,-1,-1):
      z[j] += y[j]
      # Apply j-th reflector, (I - 2.0*w_j*w_j.T)*update
      z = z - 2.0*dot(conjugate(B[j,:]), update)*B[j,:]

Parameters
----------
z : array
    length n vector to be operated on
B : array
    n x m matrix of householder reflectors
    must be in row major form
y : array
    solution to the reduced system at the end of GMRES
n : int
    dimensionality of z
start, stop, step : int
    control the choice of vectors in B to use

Returns
-------
z is modified in place to reflect the application of
the Householder reflectors, B[:,range(start,stop,step)],
and the inclusion of values in y.

Notes
-----
Principle calling routine is gmres(...) and fgmres(...) in krylov.py

References
----------
See pages 164-167 in Saad, "Iterative Methods for Sparse Linear Systems")pbdoc");

    m.def("apply_givens", &_apply_givens<int, float, float>,
        py::arg("B").noconvert(), py::arg("x").noconvert(), py::arg("n"), py::arg("nrot"));
    m.def("apply_givens", &_apply_givens<int, double, double>,
        py::arg("B").noconvert(), py::arg("x").noconvert(), py::arg("n"), py::arg("nrot"));
    m.def("apply_givens", &_apply_givens<int, std::complex<float>, float>,
        py::arg("B").noconvert(), py::arg("x").noconvert(), py::arg("n"), py::arg("nrot"));
    m.def("apply_givens", &_apply_givens<int, std::complex<double>, double>,
        py::arg("B").noconvert(), py::arg("x").noconvert(), py::arg("n"), py::arg("nrot"),
R"pbdoc(
Apply the first nrot Givens rotations in B to x

Parameters
----------
x : array
    n-vector to be operated on
B : array
    Each 4 entries represent a Givens rotation
    length nrot*4
n : int
    dimensionality of x
nrot : int
    number of rotations in B

Returns
-------
x is modified in place to reflect the application of the nrot
rotations in B.  It is assumed that the first rotation operates on
degrees of freedom 0 and 1.  The second rotation operates on dof's 1 and 2,
and so on

Notes
-----
Principle calling routine is gmres(...) and fgmres(...) in krylov.py)pbdoc");

}

