// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "ruge_stuben.h"

namespace py = pybind11;

template<class I, class T, class F>
void _classical_strength_of_connection_abs(
            const I n_row,
            const F theta,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx
                                           )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_Sp = Sp.mutable_unchecked();
    auto py_Sj = Sj.mutable_unchecked();
    auto py_Sx = Sx.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    I *_Sp = py_Sp.mutable_data();
    I *_Sj = py_Sj.mutable_data();
    T *_Sx = py_Sx.mutable_data();

    return classical_strength_of_connection_abs<I, T, F>(
                    n_row,
                    theta,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0)
                                                         );
}

template<class I, class T>
void _classical_strength_of_connection_min(
            const I n_row,
            const T theta,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx
                                           )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_Sp = Sp.mutable_unchecked();
    auto py_Sj = Sj.mutable_unchecked();
    auto py_Sx = Sx.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    I *_Sp = py_Sp.mutable_data();
    I *_Sj = py_Sj.mutable_data();
    T *_Sx = py_Sx.mutable_data();

    return classical_strength_of_connection_min<I, T>(
                    n_row,
                    theta,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0)
                                                      );
}

template<class I, class T, class F>
void _maximum_row_value(
            const I n_row,
       py::array_t<T> & x,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax
                        )
{
    auto py_x = x.mutable_unchecked();
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    T *_x = py_x.mutable_data();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();

    return maximum_row_value<I, T, F>(
                    n_row,
                       _x, x.shape(0),
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0)
                                      );
}

template<class I>
void _rs_cf_splitting(
          const I n_nodes,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<I> & Tp,
      py::array_t<I> & Tj,
py::array_t<I> & influence,
py::array_t<I> & splitting
                      )
{
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Tp = Tp.unchecked();
    auto py_Tj = Tj.unchecked();
    auto py_influence = influence.unchecked();
    auto py_splitting = splitting.mutable_unchecked();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    const I *_Tp = py_Tp.data();
    const I *_Tj = py_Tj.data();
    const I *_influence = py_influence.data();
    I *_splitting = py_splitting.mutable_data();

    return rs_cf_splitting<I>(
                  n_nodes,
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Tp, Tp.shape(0),
                      _Tj, Tj.shape(0),
               _influence, influence.shape(0),
               _splitting, splitting.shape(0)
                              );
}

template<class I>
void _rs_cf_splitting_pass2(
          const I n_nodes,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
py::array_t<I> & splitting
                            )
{
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_splitting = splitting.mutable_unchecked();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    I *_splitting = py_splitting.mutable_data();

    return rs_cf_splitting_pass2<I>(
                  n_nodes,
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
               _splitting, splitting.shape(0)
                                    );
}

template<class I>
void _cljp_naive_splitting(
                const I n,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<I> & Tp,
      py::array_t<I> & Tj,
py::array_t<I> & splitting,
        const I colorflag
                           )
{
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Tp = Tp.unchecked();
    auto py_Tj = Tj.unchecked();
    auto py_splitting = splitting.mutable_unchecked();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    const I *_Tp = py_Tp.data();
    const I *_Tj = py_Tj.data();
    I *_splitting = py_splitting.mutable_data();

    return cljp_naive_splitting<I>(
                        n,
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Tp, Tp.shape(0),
                      _Tj, Tj.shape(0),
               _splitting, splitting.shape(0),
                colorflag
                                   );
}

template<class I>
void _rs_direct_interpolation_pass1(
          const I n_nodes,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
py::array_t<I> & splitting,
      py::array_t<I> & Pp
                                    )
{
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_splitting = splitting.unchecked();
    auto py_Pp = Pp.mutable_unchecked();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    const I *_splitting = py_splitting.data();
    I *_Pp = py_Pp.mutable_data();

    return rs_direct_interpolation_pass1<I>(
                  n_nodes,
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
               _splitting, splitting.shape(0),
                      _Pp, Pp.shape(0)
                                            );
}

template<class I, class T>
void _rs_direct_interpolation_pass2(
          const I n_nodes,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx,
py::array_t<I> & splitting,
      py::array_t<I> & Pp,
      py::array_t<I> & Pj,
      py::array_t<T> & Px
                                    )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Sx = Sx.unchecked();
    auto py_splitting = splitting.unchecked();
    auto py_Pp = Pp.unchecked();
    auto py_Pj = Pj.mutable_unchecked();
    auto py_Px = Px.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    const T *_Sx = py_Sx.data();
    const I *_splitting = py_splitting.data();
    const I *_Pp = py_Pp.data();
    I *_Pj = py_Pj.mutable_data();
    T *_Px = py_Px.mutable_data();

    return rs_direct_interpolation_pass2<I, T>(
                  n_nodes,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0),
               _splitting, splitting.shape(0),
                      _Pp, Pp.shape(0),
                      _Pj, Pj.shape(0),
                      _Px, Px.shape(0)
                                               );
}

template<class I, class T>
void _cr_helper(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
       py::array_t<T> & B,
       py::array_t<T> & e,
 py::array_t<I> & indices,
py::array_t<I> & splitting,
   py::array_t<T> & gamma,
          const T thetacs
                )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_B = B.unchecked();
    auto py_e = e.mutable_unchecked();
    auto py_indices = indices.mutable_unchecked();
    auto py_splitting = splitting.mutable_unchecked();
    auto py_gamma = gamma.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_B = py_B.data();
    T *_e = py_e.mutable_data();
    I *_indices = py_indices.mutable_data();
    I *_splitting = py_splitting.mutable_data();
    T *_gamma = py_gamma.mutable_data();

    return cr_helper<I, T>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                       _B, B.shape(0),
                       _e, e.shape(0),
                 _indices, indices.shape(0),
               _splitting, splitting.shape(0),
                   _gamma, gamma.shape(0),
                  thetacs
                           );
}

template<class I>
void _rs_classical_interpolation_pass1(
          const I n_nodes,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
py::array_t<I> & splitting,
      py::array_t<I> & Pp
                                       )
{
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_splitting = splitting.unchecked();
    auto py_Pp = Pp.mutable_unchecked();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    const I *_splitting = py_splitting.data();
    I *_Pp = py_Pp.mutable_data();

    return rs_classical_interpolation_pass1<I>(
                  n_nodes,
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
               _splitting, splitting.shape(0),
                      _Pp, Pp.shape(0)
                                               );
}

template<class I, class T>
void _remove_strong_FF_connections(
          const I n_nodes,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx,
py::array_t<I> & splitting
                                   )
{
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Sx = Sx.mutable_unchecked();
    auto py_splitting = splitting.unchecked();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    T *_Sx = py_Sx.mutable_data();
    const I *_splitting = py_splitting.data();

    return remove_strong_FF_connections<I, T>(
                  n_nodes,
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0),
               _splitting, splitting.shape(0)
                                              );
}

template<class I, class T>
void _rs_classical_interpolation_pass2(
          const I n_nodes,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx,
py::array_t<I> & splitting,
      py::array_t<I> & Pp,
      py::array_t<I> & Pj,
      py::array_t<T> & Px,
      const bool modified
                                       )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Sx = Sx.unchecked();
    auto py_splitting = splitting.unchecked();
    auto py_Pp = Pp.unchecked();
    auto py_Pj = Pj.mutable_unchecked();
    auto py_Px = Px.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    const T *_Sx = py_Sx.data();
    const I *_splitting = py_splitting.data();
    const I *_Pp = py_Pp.data();
    I *_Pj = py_Pj.mutable_data();
    T *_Px = py_Px.mutable_data();

    return rs_classical_interpolation_pass2<I, T>(
                  n_nodes,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0),
               _splitting, splitting.shape(0),
                      _Pp, Pp.shape(0),
                      _Pj, Pj.shape(0),
                      _Px, Px.shape(0),
                 modified
                                                  );
}

PYBIND11_MODULE(ruge_stuben, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for ruge_stuben.h

    Methods
    -------
    classical_strength_of_connection_abs
    classical_strength_of_connection_min
    maximum_row_value
    rs_cf_splitting
    rs_cf_splitting_pass2
    cljp_naive_splitting
    rs_direct_interpolation_pass1
    rs_direct_interpolation_pass2
    cr_helper
    rs_classical_interpolation_pass1
    remove_strong_FF_connections
    rs_classical_interpolation_pass2
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("classical_strength_of_connection_abs", &_classical_strength_of_connection_abs<int, float, float>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("classical_strength_of_connection_abs", &_classical_strength_of_connection_abs<int, double, double>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("classical_strength_of_connection_abs", &_classical_strength_of_connection_abs<int, std::complex<float>, float>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("classical_strength_of_connection_abs", &_classical_strength_of_connection_abs<int, std::complex<double>, double>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(),
R"pbdoc(
Compute a strength of connection matrix using the classical strength
 of connection measure by Ruge and Stuben. Both the input and output
 matrices are stored in CSR format.  An off-diagonal nonzero entry
 A[i,j] is considered strong if:

 ..
     |A[i,j]| >= theta * max( |A[i,k]| )   where k != i

Otherwise, the connection is weak.

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
Sp : array
    CSR row pointer
Sj : array
    CSR index array
Sx : array
    CSR data array

Returns
-------
Nothing, S will be stored in Sp, Sj, Sx

Notes
-----
Storage for S must be preallocated.  Since S will consist of a subset
of A's nonzero values, a conservative bound is to allocate the same
storage for S as is used by A.)pbdoc");

    m.def("classical_strength_of_connection_min", &_classical_strength_of_connection_min<int, float>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert());
    m.def("classical_strength_of_connection_min", &_classical_strength_of_connection_min<int, double>,
        py::arg("n_row"), py::arg("theta"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(),
R"pbdoc(
)pbdoc");

    m.def("maximum_row_value", &_maximum_row_value<int, float, float>,
        py::arg("n_row"), py::arg("x").noconvert(), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert());
    m.def("maximum_row_value", &_maximum_row_value<int, double, double>,
        py::arg("n_row"), py::arg("x").noconvert(), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert());
    m.def("maximum_row_value", &_maximum_row_value<int, std::complex<float>, float>,
        py::arg("n_row"), py::arg("x").noconvert(), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert());
    m.def("maximum_row_value", &_maximum_row_value<int, std::complex<double>, double>,
        py::arg("n_row"), py::arg("x").noconvert(), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(),
R"pbdoc(
Compute the maximum in magnitude row value for a CSR matrix

Parameters
----------
num_rows : int
    number of rows in A
x : array, inplace
     num_rows array
Ap : array
    CSR row pointer
Aj : array
    CSR index array
Ax : array
    CSR data array

Returns
-------
Nothing, x[i] will hold row i's maximum magnitude entry)pbdoc");

    m.def("rs_cf_splitting", &_rs_cf_splitting<int>,
        py::arg("n_nodes"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Tp").noconvert(), py::arg("Tj").noconvert(), py::arg("influence").noconvert(), py::arg("splitting").noconvert(),
R"pbdoc(
Compute a C/F (coarse-fine( splitting using the classical coarse grid
selection method of Ruge and Stuben.  The strength of connection matrix S,
and its transpose T, are stored in CSR format.  Upon return, the  splitting
array will consist of zeros and ones, where C-nodes (coarse nodes) are
marked with the value 1 and F-nodes (fine nodes) with the value 0.

Parameters
----------
n_nodes : int
    number of rows in A
Sp : array
    CSR row pointer array for SOC matrix
Sj : array
    CSR column index array for SOC matrix
Tp : array
    CSR row pointer array for transpose of SOC matrix
Tj : array
    CSR column index array for transpose of SOC matrix
influence : array
    array that influences splitting (values stored here are
    added to lambda for each point)
splitting : array, inplace
    array to store the C/F splitting

Notes
-----
The splitting array must be preallocated)pbdoc");

    m.def("rs_cf_splitting_pass2", &_rs_cf_splitting_pass2<int>,
        py::arg("n_nodes"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("splitting").noconvert(),
R"pbdoc(
)pbdoc");

    m.def("cljp_naive_splitting", &_cljp_naive_splitting<int>,
        py::arg("n"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Tp").noconvert(), py::arg("Tj").noconvert(), py::arg("splitting").noconvert(), py::arg("colorflag"),
R"pbdoc(
)pbdoc");

    m.def("rs_direct_interpolation_pass1", &_rs_direct_interpolation_pass1<int>,
        py::arg("n_nodes"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("splitting").noconvert(), py::arg("Pp").noconvert(),
R"pbdoc(
Produce the Ruge-Stuben prolongator using "Direct Interpolation"


The first pass uses the strength of connection matrix 'S'
and C/F splitting to compute the row pointer for the prolongator.

The second pass fills in the nonzero entries of the prolongator

Parameters
----------
n_nodes : int
    Number of nodes
Sp : array
    Strength matrix row pointer array
Sj : array
    Strength matrix column index array
splitting : array
    C/F splitting
Pp : array, inplace
    Row pointer array

References
----------
Page 479 of Multigrid)pbdoc");

    m.def("rs_direct_interpolation_pass2", &_rs_direct_interpolation_pass2<int, float>,
        py::arg("n_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("splitting").noconvert(), py::arg("Pp").noconvert(), py::arg("Pj").noconvert(), py::arg("Px").noconvert());
    m.def("rs_direct_interpolation_pass2", &_rs_direct_interpolation_pass2<int, double>,
        py::arg("n_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("splitting").noconvert(), py::arg("Pp").noconvert(), py::arg("Pj").noconvert(), py::arg("Px").noconvert(),
R"pbdoc(
)pbdoc");

    m.def("cr_helper", &_cr_helper<int, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("B").noconvert(), py::arg("e").noconvert(), py::arg("indices").noconvert(), py::arg("splitting").noconvert(), py::arg("gamma").noconvert(), py::arg("thetacs"));
    m.def("cr_helper", &_cr_helper<int, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("B").noconvert(), py::arg("e").noconvert(), py::arg("indices").noconvert(), py::arg("splitting").noconvert(), py::arg("gamma").noconvert(), py::arg("thetacs"),
R"pbdoc(
Helper function for compatible relaxation to perform steps 3.1d - 3.1f
in Falgout / Brannick (2010).

Parameters
----------
Ap : array
     Row pointer for sparse matrix in CSR format.
Aj : array
     Column indices for sparse matrix in CSR format.
B : array
     Target near null space vector for computing candidate set measure.
e : array, inplace
     Relaxed vector for computing candidate set measure.
indices : array, inplace
     Array of indices, where indices[0] = the number of F indices, nf,
     followed by F indices in elements 1:nf, and C indices in (nf+1):n.
splitting : array, inplace
     Integer array with current C/F splitting of nodes, 0 = C-point,
     1 = F-point.
gamma : array, inplace
     Preallocated vector to store candidate set measure.
thetacs : float
     Threshold for coarse grid candidates from set measure.

Returns
-------
Nothing, updated C/F-splitting and corresponding indices modified in place.)pbdoc");

    m.def("rs_classical_interpolation_pass1", &_rs_classical_interpolation_pass1<int>,
        py::arg("n_nodes"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("splitting").noconvert(), py::arg("Pp").noconvert(),
R"pbdoc(
First pass of classical AMG interpolation to build row pointer for
P based on SOC matrix and CF-splitting.

Parameters
----------
n_nodes : int
    Number of rows in A
Sp : array
    Row pointer for SOC matrix, C
Sj : array
    Column indices for SOC matrix, C
splitting : array
    Boolean array with 1 denoting C-points and 0 F-points
Pp : array
    empty array to store row pointer for matrix P

Returns
-------
Nothing, Pp is modified in place.)pbdoc");

    m.def("remove_strong_FF_connections", &_remove_strong_FF_connections<int, float>,
        py::arg("n_nodes"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("splitting").noconvert());
    m.def("remove_strong_FF_connections", &_remove_strong_FF_connections<int, double>,
        py::arg("n_nodes"), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("splitting").noconvert(),
R"pbdoc(
Remove strong F-to-F connections that do NOT have a common C-point from
the set of strong connections. Specifically, set the data value in CSR
format to 0. Removing zero entries afterwards will adjust row pointer
and column indices.

Parameters
----------
n_nodes : int
    Number of rows in A
Sp : array
    Row pointer for SOC matrix, C
Sj : array
    Column indices for SOC matrix, C
Sx : array
    Data array for SOC matrix, C
splitting : array
    Boolean array with 1 denoting C-points and 0 F-points

Returns
-------
Nothing, Sx[] is set to zero to eliminate connections.)pbdoc");

    m.def("rs_classical_interpolation_pass2", &_rs_classical_interpolation_pass2<int, float>,
        py::arg("n_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("splitting").noconvert(), py::arg("Pp").noconvert(), py::arg("Pj").noconvert(), py::arg("Px").noconvert(), py::arg("modified"));
    m.def("rs_classical_interpolation_pass2", &_rs_classical_interpolation_pass2<int, double>,
        py::arg("n_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("splitting").noconvert(), py::arg("Pp").noconvert(), py::arg("Pj").noconvert(), py::arg("Px").noconvert(), py::arg("modified"),
R"pbdoc(
Produce a classical AMG interpolation operator for the case in which
two strongly connected F -points do NOT have a common C-neighbor. Formula
can be found in Sec. 3 Eq. (8) of [1] for modified=False and Eq. (9)
for modified=True.

Parameters
----------
Ap : array
    Row pointer for matrix A
Aj : array
    Column indices for matrix A
Ax : array
    Data array for matrix A
Sp : array
    Row pointer for SOC matrix, C
Sj : array
    Column indices for SOC matrix, C
Sx : array
    Data array for SOC matrix, C -- MUST HAVE VALUES OF A
splitting : array
    Boolean array with 1 denoting C-points and 0 F-points
Pp : array
    Row pointer for matrix P
Pj : array
    Column indices for matrix P
Px : array
    Data array for matrix P
modified : bool
    Use modified interpolation formula

Notes
-----
For modified interpolation, it is assumed that SOC matrix C is
passed in WITHOUT any F-to-F connections that do not share a
common C-point neighbor. Any SOC matrix C can be set as such by
calling remove_strong_FF_connections().

Returns
-------
Nothing, Pj[] and Px[] modified in place.

References
----------
[0] V. E. Henson and U. M. Yang, BoomerAMG: a parallel algebraic multigrid
     solver and preconditioner, Applied Numerical Mathematics 41 (2002).

[1] "Distance-Two Interpolation for Parallel Algebraic Multigrid,"
     H. De Sterck, R. Falgout, J. Nolting, U. M. Yang, (2008).)pbdoc");

}

