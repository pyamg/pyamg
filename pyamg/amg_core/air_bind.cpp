// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "air.h"

namespace py = pybind11;

template<class I, class T>
void _one_point_interpolation(
      py::array_t<I> & Pp,
      py::array_t<I> & Pj,
      py::array_t<T> & Px,
      py::array_t<I> & Cp,
      py::array_t<I> & Cj,
      py::array_t<T> & Cx,
py::array_t<I> & splitting
                              )
{
    auto py_Pp = Pp.mutable_unchecked();
    auto py_Pj = Pj.mutable_unchecked();
    auto py_Px = Px.mutable_unchecked();
    auto py_Cp = Cp.unchecked();
    auto py_Cj = Cj.unchecked();
    auto py_Cx = Cx.unchecked();
    auto py_splitting = splitting.unchecked();
    I *_Pp = py_Pp.mutable_data();
    I *_Pj = py_Pj.mutable_data();
    T *_Px = py_Px.mutable_data();
    const I *_Cp = py_Cp.data();
    const I *_Cj = py_Cj.data();
    const T *_Cx = py_Cx.data();
    const I *_splitting = py_splitting.data();

    return one_point_interpolation<I, T>(
                      _Pp, Pp.shape(0),
                      _Pj, Pj.shape(0),
                      _Px, Px.shape(0),
                      _Cp, Cp.shape(0),
                      _Cj, Cj.shape(0),
                      _Cx, Cx.shape(0),
               _splitting, splitting.shape(0)
                                         );
}

template<class I>
void _approx_ideal_restriction_pass1(
      py::array_t<I> & Rp,
      py::array_t<I> & Cp,
      py::array_t<I> & Cj,
    py::array_t<I> & Cpts,
py::array_t<I> & splitting,
         const I distance
                                     )
{
    auto py_Rp = Rp.mutable_unchecked();
    auto py_Cp = Cp.unchecked();
    auto py_Cj = Cj.unchecked();
    auto py_Cpts = Cpts.unchecked();
    auto py_splitting = splitting.unchecked();
    I *_Rp = py_Rp.mutable_data();
    const I *_Cp = py_Cp.data();
    const I *_Cj = py_Cj.data();
    const I *_Cpts = py_Cpts.data();
    const I *_splitting = py_splitting.data();

    return approx_ideal_restriction_pass1<I>(
                      _Rp, Rp.shape(0),
                      _Cp, Cp.shape(0),
                      _Cj, Cj.shape(0),
                    _Cpts, Cpts.shape(0),
               _splitting, splitting.shape(0),
                 distance
                                             );
}

template<class I, class T>
void _approx_ideal_restriction_pass2(
      py::array_t<I> & Rp,
      py::array_t<I> & Rj,
      py::array_t<T> & Rx,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<I> & Cp,
      py::array_t<I> & Cj,
      py::array_t<T> & Cx,
    py::array_t<I> & Cpts,
py::array_t<I> & splitting,
         const I distance,
        const I use_gmres,
          const I maxiter,
     const I precondition
                                     )
{
    auto py_Rp = Rp.unchecked();
    auto py_Rj = Rj.mutable_unchecked();
    auto py_Rx = Rx.mutable_unchecked();
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_Cp = Cp.unchecked();
    auto py_Cj = Cj.unchecked();
    auto py_Cx = Cx.unchecked();
    auto py_Cpts = Cpts.unchecked();
    auto py_splitting = splitting.unchecked();
    const I *_Rp = py_Rp.data();
    I *_Rj = py_Rj.mutable_data();
    T *_Rx = py_Rx.mutable_data();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    const I *_Cp = py_Cp.data();
    const I *_Cj = py_Cj.data();
    const T *_Cx = py_Cx.data();
    const I *_Cpts = py_Cpts.data();
    const I *_splitting = py_splitting.data();

    return approx_ideal_restriction_pass2<I, T>(
                      _Rp, Rp.shape(0),
                      _Rj, Rj.shape(0),
                      _Rx, Rx.shape(0),
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Cp, Cp.shape(0),
                      _Cj, Cj.shape(0),
                      _Cx, Cx.shape(0),
                    _Cpts, Cpts.shape(0),
               _splitting, splitting.shape(0),
                 distance,
                use_gmres,
                  maxiter,
             precondition
                                                );
}

template<class I, class T>
void _block_approx_ideal_restriction_pass2(
      py::array_t<I> & Rp,
      py::array_t<I> & Rj,
      py::array_t<T> & Rx,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<I> & Cp,
      py::array_t<I> & Cj,
      py::array_t<T> & Cx,
    py::array_t<I> & Cpts,
py::array_t<I> & splitting,
        const I blocksize,
         const I distance,
        const I use_gmres,
          const I maxiter,
     const I precondition
                                           )
{
    auto py_Rp = Rp.unchecked();
    auto py_Rj = Rj.mutable_unchecked();
    auto py_Rx = Rx.mutable_unchecked();
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_Cp = Cp.unchecked();
    auto py_Cj = Cj.unchecked();
    auto py_Cx = Cx.unchecked();
    auto py_Cpts = Cpts.unchecked();
    auto py_splitting = splitting.unchecked();
    const I *_Rp = py_Rp.data();
    I *_Rj = py_Rj.mutable_data();
    T *_Rx = py_Rx.mutable_data();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    const I *_Cp = py_Cp.data();
    const I *_Cj = py_Cj.data();
    const T *_Cx = py_Cx.data();
    const I *_Cpts = py_Cpts.data();
    const I *_splitting = py_splitting.data();

    return block_approx_ideal_restriction_pass2<I, T>(
                      _Rp, Rp.shape(0),
                      _Rj, Rj.shape(0),
                      _Rx, Rx.shape(0),
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Cp, Cp.shape(0),
                      _Cj, Cj.shape(0),
                      _Cx, Cx.shape(0),
                    _Cpts, Cpts.shape(0),
               _splitting, splitting.shape(0),
                blocksize,
                 distance,
                use_gmres,
                  maxiter,
             precondition
                                                      );
}

PYBIND11_MODULE(air, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for air.h

    Methods
    -------
    one_point_interpolation
    sort_2nd
    approx_ideal_restriction_pass1
    approx_ideal_restriction_pass2
    block_approx_ideal_restriction_pass2
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("one_point_interpolation", &_one_point_interpolation<int, float>,
        py::arg("Pp").noconvert(), py::arg("Pj").noconvert(), py::arg("Px").noconvert(), py::arg("Cp").noconvert(), py::arg("Cj").noconvert(), py::arg("Cx").noconvert(), py::arg("splitting").noconvert());
    m.def("one_point_interpolation", &_one_point_interpolation<int, double>,
        py::arg("Pp").noconvert(), py::arg("Pj").noconvert(), py::arg("Px").noconvert(), py::arg("Cp").noconvert(), py::arg("Cj").noconvert(), py::arg("Cx").noconvert(), py::arg("splitting").noconvert(),
R"pbdoc(
Interpolate C-points by value and each F-point by value from its strongest
connected C-neighbor.

Parameters
----------
     Rp : const array<int>
         Pre-determined row-pointer for P in CSR format
     Rj : array<int>
         Empty array for column indices for P in CSR format
     Cp : const array<int>
         Row pointer for SOC matrix, C
     Cj : const array<int>
         Column indices for SOC matrix, C
     Cx : const array<float>
         Data array for SOC matrix, C
     splitting : const array<int>
         Boolean array with 1 denoting C-points and 0 F-points

Returns
-------
Nothing, Rj[] modified in place.)pbdoc");

    m.def("approx_ideal_restriction_pass1", &_approx_ideal_restriction_pass1<int>,
        py::arg("Rp").noconvert(), py::arg("Cp").noconvert(), py::arg("Cj").noconvert(), py::arg("Cpts").noconvert(), py::arg("splitting").noconvert(), py::arg("distance"),
R"pbdoc(
Build row_pointer for approximate ideal restriction in CSR or BSR form.

Parameters
----------
     Rp : array<int>
         Empty row-pointer for R
     Cp : const array<int>
         Row pointer for SOC matrix, C
     Cj : const array<int>
         Column indices for SOC matrix, C
     Cpts : array<int>
         List of global C-point indices
     splitting : const array<int>
         Boolean array with 1 denoting C-points and 0 F-points
     distance : int, default 2
         Distance of F-point neighborhood to consider, options are 1 and 2.

Returns
-------
Nothing, Rp[] modified in place.)pbdoc");

    m.def("approx_ideal_restriction_pass2", &_approx_ideal_restriction_pass2<int, float>,
        py::arg("Rp").noconvert(), py::arg("Rj").noconvert(), py::arg("Rx").noconvert(), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Cp").noconvert(), py::arg("Cj").noconvert(), py::arg("Cx").noconvert(), py::arg("Cpts").noconvert(), py::arg("splitting").noconvert(), py::arg("distance"), py::arg("use_gmres"), py::arg("maxiter"), py::arg("precondition"));
    m.def("approx_ideal_restriction_pass2", &_approx_ideal_restriction_pass2<int, double>,
        py::arg("Rp").noconvert(), py::arg("Rj").noconvert(), py::arg("Rx").noconvert(), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Cp").noconvert(), py::arg("Cj").noconvert(), py::arg("Cx").noconvert(), py::arg("Cpts").noconvert(), py::arg("splitting").noconvert(), py::arg("distance"), py::arg("use_gmres"), py::arg("maxiter"), py::arg("precondition"),
R"pbdoc(
Build column indices and data array for approximate ideal restriction
in CSR format.

Parameters
----------
     Rp : const array<int>
         Pre-determined row-pointer for R in CSR format
     Rj : array<int>
         Empty array for column indices for R in CSR format
     Rx : array<float>
         Empty array for data for R in CSR format
     Ap : const array<int>
         Row pointer for matrix A
     Aj : const array<int>
         Column indices for matrix A
     Ax : const array<float>
         Data array for matrix A
     Cp : const array<int>
         Row pointer for SOC matrix, C
     Cj : const array<int>
         Column indices for SOC matrix, C
     Cx : const array<float>
         Data array for SOC matrix, C
     Cpts : array<int>
         List of global C-point indices
     splitting : const array<int>
         Boolean array with 1 denoting C-points and 0 F-points
     distance : int, default 2
         Distance of F-point neighborhood to consider, options are 1 and 2.
     use_gmres : bool, default 0
         Use GMRES for local dense solve
     maxiter : int, default 10
         Maximum GMRES iterations
     precondition : bool, default True
         Diagonally precondition GMRES

Returns
-------
Nothing, Rj[] and Rx[] modified in place.

Notes
-----
Rx[] must be passed in initialized to zero.)pbdoc");

    m.def("block_approx_ideal_restriction_pass2", &_block_approx_ideal_restriction_pass2<int, float>,
        py::arg("Rp").noconvert(), py::arg("Rj").noconvert(), py::arg("Rx").noconvert(), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Cp").noconvert(), py::arg("Cj").noconvert(), py::arg("Cx").noconvert(), py::arg("Cpts").noconvert(), py::arg("splitting").noconvert(), py::arg("blocksize"), py::arg("distance"), py::arg("use_gmres"), py::arg("maxiter"), py::arg("precondition"));
    m.def("block_approx_ideal_restriction_pass2", &_block_approx_ideal_restriction_pass2<int, double>,
        py::arg("Rp").noconvert(), py::arg("Rj").noconvert(), py::arg("Rx").noconvert(), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Cp").noconvert(), py::arg("Cj").noconvert(), py::arg("Cx").noconvert(), py::arg("Cpts").noconvert(), py::arg("splitting").noconvert(), py::arg("blocksize"), py::arg("distance"), py::arg("use_gmres"), py::arg("maxiter"), py::arg("precondition"),
R"pbdoc(
Build column indices and data array for approximate ideal restriction
in CSR format.

Parameters
----------
     Rp : const array<int>
         Pre-determined row-pointer for R in CSR format
     Rj : array<int>
         Empty array for column indices for R in CSR format
     Rx : array<float>
         Empty array for data for R in CSR format
     Ap : const array<int>
         Row pointer for matrix A
     Aj : const array<int>
         Column indices for matrix A
     Ax : const array<float>
         Data array for matrix A
     Cp : const array<int>
         Row pointer for SOC matrix, C
     Cj : const array<int>
         Column indices for SOC matrix, C
     Cx : const array<float>
         Data array for SOC matrix, C
     Cpts : array<int>
         List of global C-point indices
     splitting : const array<int>
         Boolean array with 1 denoting C-points and 0 F-points
     blocksize : int
         Blocksize of matrix (assume square blocks)
     distance : int, default 2
         Distance of F-point neighborhood to consider, options are 1 and 2.
     use_gmres : bool, default 0
         Use GMRES for local dense solve
     maxiter : int, default 10
         Maximum GMRES iterations
     precondition : bool, default True
         Diagonally precondition GMRES

Returns
-------
Nothing, Rj[] and Rx[] modified in place.

Notes
-----
Rx[] must be passed in initialized to zero.)pbdoc");

}

