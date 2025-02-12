// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "relaxation.h"

namespace py = pybind11;

template<class I, class T, class F>
void _gauss_seidel(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
        const I row_start,
         const I row_stop,
         const I row_step
                   )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();

    return gauss_seidel<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                row_start,
                 row_stop,
                 row_step
                                 );
}

template<class I, class T, class F>
void _sor_gauss_seidel(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
        const I row_start,
         const I row_stop,
         const I row_step,
            const F omega
                       )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();

    return sor_gauss_seidel<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                row_start,
                 row_stop,
                 row_step,
                    omega
                                     );
}

template<class I, class T, class F>
void _bsr_gauss_seidel(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
        const I row_start,
         const I row_stop,
         const I row_step,
        const I blocksize
                       )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();

    return bsr_gauss_seidel<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                row_start,
                 row_stop,
                 row_step,
                blocksize
                                     );
}

template<class I, class T, class F>
void _jacobi(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
    py::array_t<T> & temp,
        const I row_start,
         const I row_stop,
         const I row_step,
   py::array_t<T> & omega
             )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_temp = temp.mutable_unchecked();
    auto py_omega = omega.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    T *_temp = py_temp.mutable_data();
    const T *_omega = py_omega.data();

    return jacobi<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                    _temp, temp.shape(0),
                row_start,
                 row_stop,
                 row_step,
                   _omega, omega.shape(0)
                           );
}

template<class I, class T, class F>
void _jacobi_indexed(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
 py::array_t<I> & indices,
   py::array_t<T> & omega
                     )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_indices = indices.unchecked();
    auto py_omega = omega.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    const I *_indices = py_indices.data();
    const T *_omega = py_omega.data();

    return jacobi_indexed<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                 _indices, indices.shape(0),
                   _omega, omega.shape(0)
                                   );
}

template<class I, class T, class F>
void _bsr_jacobi(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
    py::array_t<T> & temp,
        const I row_start,
         const I row_stop,
         const I row_step,
        const I blocksize,
   py::array_t<T> & omega
                 )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_temp = temp.mutable_unchecked();
    auto py_omega = omega.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    T *_temp = py_temp.mutable_data();
    const T *_omega = py_omega.data();

    return bsr_jacobi<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                    _temp, temp.shape(0),
                row_start,
                 row_stop,
                 row_step,
                blocksize,
                   _omega, omega.shape(0)
                               );
}

template<class I, class T, class F>
void _bsr_jacobi_indexed(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
 py::array_t<I> & indices,
        const I blocksize,
   py::array_t<T> & omega
                         )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_indices = indices.unchecked();
    auto py_omega = omega.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    const I *_indices = py_indices.data();
    const T *_omega = py_omega.data();

    return bsr_jacobi_indexed<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                 _indices, indices.shape(0),
                blocksize,
                   _omega, omega.shape(0)
                                       );
}

template<class I, class T, class F>
void _gauss_seidel_indexed(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
      py::array_t<I> & Id,
        const I row_start,
         const I row_stop,
         const I row_step
                           )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_Id = Id.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    const I *_Id = py_Id.data();

    return gauss_seidel_indexed<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                      _Id, Id.shape(0),
                row_start,
                 row_stop,
                 row_step
                                         );
}

template<class I, class T, class F>
void _jacobi_ne(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
      py::array_t<T> & Tx,
    py::array_t<T> & temp,
        const I row_start,
         const I row_stop,
         const I row_step,
   py::array_t<T> & omega
                )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_Tx = Tx.unchecked();
    auto py_temp = temp.mutable_unchecked();
    auto py_omega = omega.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    const T *_Tx = py_Tx.data();
    T *_temp = py_temp.mutable_data();
    const T *_omega = py_omega.data();

    return jacobi_ne<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                      _Tx, Tx.shape(0),
                    _temp, temp.shape(0),
                row_start,
                 row_stop,
                 row_step,
                   _omega, omega.shape(0)
                              );
}

template<class I, class T, class F>
void _gauss_seidel_ne(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
        const I row_start,
         const I row_stop,
         const I row_step,
      py::array_t<T> & Tx,
            const F omega
                      )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_Tx = Tx.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    const T *_Tx = py_Tx.data();

    return gauss_seidel_ne<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                row_start,
                 row_stop,
                 row_step,
                      _Tx, Tx.shape(0),
                    omega
                                    );
}

template<class I, class T, class F>
void _gauss_seidel_nr(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & z,
        const I col_start,
         const I col_stop,
         const I col_step,
      py::array_t<T> & Tx,
            const F omega
                      )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_z = z.mutable_unchecked();
    auto py_Tx = Tx.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    T *_z = py_z.mutable_data();
    const T *_Tx = py_Tx.data();

    return gauss_seidel_nr<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _z, z.shape(0),
                col_start,
                 col_stop,
                 col_step,
                      _Tx, Tx.shape(0),
                    omega
                                    );
}

template<class I, class T, class F>
void _block_jacobi(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
      py::array_t<T> & Tx,
    py::array_t<T> & temp,
        const I row_start,
         const I row_stop,
         const I row_step,
   py::array_t<T> & omega,
        const I blocksize
                   )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_Tx = Tx.unchecked();
    auto py_temp = temp.mutable_unchecked();
    auto py_omega = omega.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    const T *_Tx = py_Tx.data();
    T *_temp = py_temp.mutable_data();
    const T *_omega = py_omega.data();

    return block_jacobi<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                      _Tx, Tx.shape(0),
                    _temp, temp.shape(0),
                row_start,
                 row_stop,
                 row_step,
                   _omega, omega.shape(0),
                blocksize
                                 );
}

template<class I, class T, class F>
void _block_jacobi_indexed(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
      py::array_t<T> & Tx,
 py::array_t<I> & indices,
   py::array_t<T> & omega,
        const I blocksize
                           )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_Tx = Tx.unchecked();
    auto py_indices = indices.unchecked();
    auto py_omega = omega.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    const T *_Tx = py_Tx.data();
    const I *_indices = py_indices.data();
    const T *_omega = py_omega.data();

    return block_jacobi_indexed<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                      _Tx, Tx.shape(0),
                 _indices, indices.shape(0),
                   _omega, omega.shape(0),
                blocksize
                                         );
}

template<class I, class T, class F>
void _block_gauss_seidel(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
      py::array_t<T> & Tx,
        const I row_start,
         const I row_stop,
         const I row_step,
        const I blocksize
                         )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_Tx = Tx.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    const T *_Tx = py_Tx.data();

    return block_gauss_seidel<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                      _Tx, Tx.shape(0),
                row_start,
                 row_stop,
                 row_step,
                blocksize
                                       );
}

template<class I, class T, class F>
void _extract_subblocks(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<T> & Tx,
      py::array_t<I> & Tp,
      py::array_t<I> & Sj,
      py::array_t<I> & Sp,
        const I nsdomains,
            const I nrows
                        )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_Tx = Tx.mutable_unchecked();
    auto py_Tp = Tp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Sp = Sp.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_Tx = py_Tx.mutable_data();
    const I *_Tp = py_Tp.data();
    const I *_Sj = py_Sj.data();
    const I *_Sp = py_Sp.data();

    return extract_subblocks<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Tx, Tx.shape(0),
                      _Tp, Tp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sp, Sp.shape(0),
                nsdomains,
                    nrows
                                      );
}

template<class I, class T, class F>
void _overlapping_schwarz_csr(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
      py::array_t<T> & Tx,
      py::array_t<I> & Tp,
      py::array_t<I> & Sj,
      py::array_t<I> & Sp,
              I nsdomains,
                  I nrows,
              I row_start,
               I row_stop,
               I row_step
                              )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_Tx = Tx.unchecked();
    auto py_Tp = Tp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Sp = Sp.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    const T *_Tx = py_Tx.data();
    const I *_Tp = py_Tp.data();
    const I *_Sj = py_Sj.data();
    const I *_Sp = py_Sp.data();

    return overlapping_schwarz_csr<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                      _Tx, Tx.shape(0),
                      _Tp, Tp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sp, Sp.shape(0),
                nsdomains,
                    nrows,
                row_start,
                 row_stop,
                 row_step
                                            );
}

PYBIND11_MODULE(relaxation, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for relaxation.h

    Methods
    -------
    gauss_seidel
    sor_gauss_seidel
    bsr_gauss_seidel
    jacobi
    jacobi_indexed
    bsr_jacobi
    bsr_jacobi_indexed
    gauss_seidel_indexed
    jacobi_ne
    gauss_seidel_ne
    gauss_seidel_nr
    block_jacobi
    block_jacobi_indexed
    block_gauss_seidel
    extract_subblocks
    overlapping_schwarz_csr
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("gauss_seidel", &_gauss_seidel<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("gauss_seidel", &_gauss_seidel<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("gauss_seidel", &_gauss_seidel<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("gauss_seidel", &_gauss_seidel<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"),
R"pbdoc(
Gauss-Seidel iteration.

Perform one iteration of Gauss-Seidel relaxation on the linear
system Ax = b, where A is stored in CSR format and x and b
are column vectors.

Parameters
----------
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
Ax : array
    CSR data array.
x : array
    Approximate solution.
b : array
    Right hand side.
row_start : int
    Beginning of the sweep.
row_stop : int
    End of the sweep (i.e. one past the last unknown).
row_step : int
    Stride used during the sweep (may be negative).

Returns
-------
None
    Array x will be modified inplace.

Notes
-----
The unknowns are swept through according to the slice defined
by row_start, row_end, and row_step.  These options are used
to implement standard forward and backward sweeps, or sweeping
only a subset of the unknowns.  A forward sweep is implemented
with ``gauss_seidel(Ap, Aj, Ax, x, b, 0, N, 1)`` where N is the
number of rows in matrix A.  Similarly, a backward sweep is
implemented with ``gauss_seidel(Ap, Aj, Ax, x, b, N, -1, -1)``.)pbdoc");

    m.def("sor_gauss_seidel", &_sor_gauss_seidel<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega"));
    m.def("sor_gauss_seidel", &_sor_gauss_seidel<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega"));
    m.def("sor_gauss_seidel", &_sor_gauss_seidel<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega"));
    m.def("sor_gauss_seidel", &_sor_gauss_seidel<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega"),
R"pbdoc(
SOR iteration.

Perform one iteration of SOR relaxation on the linear
system Ax = b, where A is stored in CSR format and x and b
are column vectors.

Parameters
----------
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
Ax : array
    CSR data array.
x : array
    Approximate solution.
b : array
    Right hand side.
row_start : int
    Beginning of the sweep.
row_stop : int
    End of the sweep (i.e. one past the last unknown).
row_step : int
    Stride used during the sweep (may be negative).
omega : float
    Relaxation parameter for SOR.

Returns
-------
None
    Array x will be modified inplace.

Notes
-----
Nearly identical to ``gauss_seidel`` with a relaxation parameter ``omega``.)pbdoc");

    m.def("bsr_gauss_seidel", &_bsr_gauss_seidel<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("blocksize"));
    m.def("bsr_gauss_seidel", &_bsr_gauss_seidel<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("blocksize"));
    m.def("bsr_gauss_seidel", &_bsr_gauss_seidel<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("blocksize"));
    m.def("bsr_gauss_seidel", &_bsr_gauss_seidel<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("blocksize"),
R"pbdoc(
Gauss-Seidel iteration with BSR arrays.

Perform one iteration of Gauss-Seidel relaxation on the linear
system Ax = b, where A is stored in Block CSR format and x and b
are column vectors.  This method applies point-wise relaxation
to the BSR as opposed to \"block relaxation\".

Refer to gauss_seidel for additional information regarding
row_start, row_stop, and row_step.

Parameters
----------
Ap : array
    BSR row pointer.
Aj : array
    BSR index array.
Ax : array
    BSR data array.
x : array
    Approximate solution.
b : array
    Right hand side.
row_start : int
    Beginning of the sweep (block row index).
row_stop : int
    End of the sweep (i.e. one past the last unknown).
row_step : int
    Stride used during the sweep (may be negative).
blocksize : int
    BSR blocksize (blocks must be square).

Returns
-------
None
    Array x will be modified inplace.)pbdoc");

    m.def("jacobi", &_jacobi<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega").noconvert());
    m.def("jacobi", &_jacobi<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega").noconvert());
    m.def("jacobi", &_jacobi<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega").noconvert());
    m.def("jacobi", &_jacobi<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega").noconvert(),
R"pbdoc(
Weighted Jacobi iteration.

Perform one iteration of Jacobi relaxation on the linear
system Ax = b, where A is stored in CSR format and x and b
are column vectors.  Damping is controlled by the omega
parameter.

Refer to gauss_seidel for additional information regarding
row_start, row_stop, and row_step.

Parameters
----------
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
Ax : array
    CSR data array.
x : array
    Approximate solution.
b : array
    Right hand side.
temp : array
    Temporary vector the same size as x.
row_start : int
    Beginning of the sweep.
row_stop : int
    End of the sweep (i.e. one past the last unknown).
row_step : int
    Stride used during the sweep (may be negative).
omega : float
    Damping parameter.

Returns
-------
None
    Array x will be modified inplace.)pbdoc");

    m.def("jacobi_indexed", &_jacobi_indexed<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("indices").noconvert(), py::arg("omega").noconvert());
    m.def("jacobi_indexed", &_jacobi_indexed<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("indices").noconvert(), py::arg("omega").noconvert());
    m.def("jacobi_indexed", &_jacobi_indexed<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("indices").noconvert(), py::arg("omega").noconvert());
    m.def("jacobi_indexed", &_jacobi_indexed<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("indices").noconvert(), py::arg("omega").noconvert(),
R"pbdoc(
Indexed weighted Jacobi iteration.

Perform one iteration of Jacobi relaxation on the linear
system Ax = b for a given set of row indices, where A is
stored in CSR format and x and b are column vectors.
Damping is controlled by the omega parameter.

Parameters
----------
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
Ax : array
    CSR data array.
x : array
    Approximate solution.
b : array
    Right hand side.
temp : array
    Temporary vector the same size as x.
indices : array
    List of row indices to perform Jacobi on, e.g. F-points.
omega : array
    Damping parameter.

Returns
-------
None
    Array x will be modified in place.)pbdoc");

    m.def("bsr_jacobi", &_bsr_jacobi<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("blocksize"), py::arg("omega").noconvert());
    m.def("bsr_jacobi", &_bsr_jacobi<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("blocksize"), py::arg("omega").noconvert());
    m.def("bsr_jacobi", &_bsr_jacobi<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("blocksize"), py::arg("omega").noconvert());
    m.def("bsr_jacobi", &_bsr_jacobi<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("blocksize"), py::arg("omega").noconvert(),
R"pbdoc(
Weighted Jacobi iteration on BSR arrays.

Perform one iteration of Jacobi relaxation on the linear
system Ax = b, where A is stored in Block CSR format and x and b
are column vectors.  This method applies point-wise relaxation
to the BSR as opposed to \"block relaxation\".

Refer to jacobi for additional information regarding
row_start, row_stop, and row_step.

Parameters
----------
Ap : array
    BSR row pointer.
Aj : array
    BSR index array.
Ax : array
    BSR data array.
x : array
    Approximate solution.
b : array
    Right hand side.
temp : array
    Temporary vector the same size as x.
row_start : int
    Beginning of the sweep (block row index).
row_stop : int
    End of the sweep (i.e. one past the last unknown).
row_step : int
    Stride used during the sweep (may be negative).
blocksize : int
    BSR blocksize (blocks must be square).
omega : float
    Damping parameter.

Returns
-------
None
    Array x will be modified inplace.)pbdoc");

    m.def("bsr_jacobi_indexed", &_bsr_jacobi_indexed<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("indices").noconvert(), py::arg("blocksize"), py::arg("omega").noconvert());
    m.def("bsr_jacobi_indexed", &_bsr_jacobi_indexed<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("indices").noconvert(), py::arg("blocksize"), py::arg("omega").noconvert());
    m.def("bsr_jacobi_indexed", &_bsr_jacobi_indexed<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("indices").noconvert(), py::arg("blocksize"), py::arg("omega").noconvert());
    m.def("bsr_jacobi_indexed", &_bsr_jacobi_indexed<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("indices").noconvert(), py::arg("blocksize"), py::arg("omega").noconvert(),
R"pbdoc(
Indexed weighted Jacobi on BSR arrays.

Perform one iteration of Jacobi relaxation on the linear
system Ax = b for a given set of row indices, where A is
stored in Block CSR format and x and b are column vectors.
This method applies point-wise relaxation to the BSR matrix
for a given set of row block indices, as opposed to "block
relaxation".

Parameters
----------
Ap : array
   BSR row pointer.
Aj : array
   BSR index array.
Ax : array
   BSR data array.
x : array
   Approximate solution.
b : array
   Right hand side.
indices : array
   List of row indices to perform Jacobi on, e.g., F-points.
   Note, it is assumed that indices correspond to blocks in A.
blocksize : int
    BSR blocksize (blocks must be square).
omega : float
    Damping parameter.

Returns
-------
None
    Array x will be modified in place.)pbdoc");

    m.def("gauss_seidel_indexed", &_gauss_seidel_indexed<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Id").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("gauss_seidel_indexed", &_gauss_seidel_indexed<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Id").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("gauss_seidel_indexed", &_gauss_seidel_indexed<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Id").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("gauss_seidel_indexed", &_gauss_seidel_indexed<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Id").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"),
R"pbdoc(
Indexed Gauss-Seidel iteration.

Perform one iteration of Gauss-Seidel relaxation on the linear
system Ax = b, where A is stored in CSR format and x and b
are column vectors.

Parameters
----------
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
Ax : array
    CSR data array.
x : array
    Approximate solution.
b : array
    Right hand side.
Id : array
    Index array representing the.
row_start : int
    Beginning of the sweep (in array Id).
row_stop : int
    End of the sweep (in array Id).
row_step : int
    Stride used during the sweep (may be negative).

Returns
-------
None
    Array x will be modified inplace.

Notes
-----
Unlike gauss_seidel, which is restricted to updating a slice
of the unknowns (defined by row_start, row_start, and row_step),
this method updates unknowns according to the rows listed in
an index array.  This allows and arbitrary set of the unknowns
to be updated in an arbitrary order, as is necessary for the
relaxation steps in the Compatible Relaxation method.

In this method the slice arguments are used to define the subset
of the index array Id which is to be considered.)pbdoc");

    m.def("jacobi_ne", &_jacobi_ne<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega").noconvert());
    m.def("jacobi_ne", &_jacobi_ne<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega").noconvert());
    m.def("jacobi_ne", &_jacobi_ne<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega").noconvert());
    m.def("jacobi_ne", &_jacobi_ne<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega").noconvert(),
R"pbdoc(
Jacobi NE iteration.

Perform NE Jacobi on the linear system A x = b
This effectively carries out weighted-Jacobi on A^TA x = A^T b
(also known as Cimmino's relaxation)

Parameters
----------
Ap : array
    Index pointer for CSR matrix A.
Aj : array
    Column indices for CSR matrix A.
Ax : array
    Value array for CSR matrix A.
x : array
    Current guess to the linear system.
b : array
    Right hand side.
Tx : array
    Scaled residual D_A^{-1} (b - Ax).
temp : array
    Work space.
row_start : int
    Controls which rows to start on.
row_stop : int
    Controls which rows to stop on.
row_step : int
    Controls which rows to iterate over.
omega : array
    Size one array that contains the weighted-jacobi
    parameter.  An array must be used to pass in omega to
    account for the case where omega may be complex.

Returns
-------
None
    x is modified inplace in an additive, not overwriting fashion.)pbdoc");

    m.def("gauss_seidel_ne", &_gauss_seidel_ne<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("Tx").noconvert(), py::arg("omega"));
    m.def("gauss_seidel_ne", &_gauss_seidel_ne<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("Tx").noconvert(), py::arg("omega"));
    m.def("gauss_seidel_ne", &_gauss_seidel_ne<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("Tx").noconvert(), py::arg("omega"));
    m.def("gauss_seidel_ne", &_gauss_seidel_ne<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("Tx").noconvert(), py::arg("omega"),
R"pbdoc(
Gauss-Seidel NE iteration.

Perform NE Gauss-Seidel on the linear system A x = b
This effectively carries out Gauss-Seidel on A A.H y = b,
where x = A.h y.

Parameters
----------
Ap : array
    Index pointer for CSR matrix A.
Aj : array
    Column indices for CSR matrix A.
Ax : array
    Value array for CSR matrix A.
x : array
    Current guess to the linear system.
b : array
    Right hand side.
Tx : array
    Inverse(diag(A A.H)).
omega : float
    Relaxation parameter (if not 1.0, then algorithm becomes SOR).
row_start,stop,step : int
    Controls which rows to iterate over.

Returns
-------
None
    Array x is modified inplace in an additive, not overwriting fashion.)pbdoc");

    m.def("gauss_seidel_nr", &_gauss_seidel_nr<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("z").noconvert(), py::arg("col_start"), py::arg("col_stop"), py::arg("col_step"), py::arg("Tx").noconvert(), py::arg("omega"));
    m.def("gauss_seidel_nr", &_gauss_seidel_nr<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("z").noconvert(), py::arg("col_start"), py::arg("col_stop"), py::arg("col_step"), py::arg("Tx").noconvert(), py::arg("omega"));
    m.def("gauss_seidel_nr", &_gauss_seidel_nr<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("z").noconvert(), py::arg("col_start"), py::arg("col_stop"), py::arg("col_step"), py::arg("Tx").noconvert(), py::arg("omega"));
    m.def("gauss_seidel_nr", &_gauss_seidel_nr<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("z").noconvert(), py::arg("col_start"), py::arg("col_stop"), py::arg("col_step"), py::arg("Tx").noconvert(), py::arg("omega"),
R"pbdoc(
Gauss-Seidel NR iteration.

Perform NR Gauss-Seidel on the linear system A x = b
This effectively carries out Gauss-Seidel on A.H A x = A.H b

Parameters
----------
Ap : array
    Index pointer for CSC matrix A.
Aj : array
    Row indices for CSC matrix A.
Ax : array
    Value array for CSC matrix A.
x : array
    Current guess to the linear system.
z : array
    Initial residual.
Tx : array
    Inverse(diag(A.H A)).
omega : float
    Relaxation parameter (if not 1.0, then algorithm becomes SOR).
col_start,stop,step : int
    Controls which rows to iterate over.

Returns
-------
None
    Array x is modified inplace in an additive, not overwriting fashion.)pbdoc");

    m.def("block_jacobi", &_block_jacobi<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega").noconvert(), py::arg("blocksize"));
    m.def("block_jacobi", &_block_jacobi<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega").noconvert(), py::arg("blocksize"));
    m.def("block_jacobi", &_block_jacobi<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega").noconvert(), py::arg("blocksize"));
    m.def("block_jacobi", &_block_jacobi<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("temp").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("omega").noconvert(), py::arg("blocksize"),
R"pbdoc(
Block Jacobi iteration.

Perform one iteration of block Jacobi relaxation on the linear
system Ax = b, where A is stored in BSR format and x and b
are column vectors.  Damping is controlled by the omega
parameter.

Refer to gauss_seidel for additional information regarding
row_start, row_stop, and row_step.

Parameters
----------
Ap : array
    BSR row pointer.
Aj : array
    BSR index array.
Ax : array
    BSR data array, blocks assumed square.
x : array
    Approximate solution.
b : array
    Right hand side.
Tx : array
    Inverse of each diagonal block of A stored
    as a (n/blocksize, blocksize, blocksize) array.
temp : array
    Temporary vector the same size as x.
row_start : int
    Beginning of the sweep.
row_stop : int
    End of the sweep (i.e. one past the last unknown).
row_step : int
    Stride used during the sweep (may be negative).
omega : float
    Damping parameter.
blocksize : int
    Dimension of square blocks in BSR matrix A.

Returns
-------
None
    Result in place.)pbdoc");

    m.def("block_jacobi_indexed", &_block_jacobi_indexed<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("indices").noconvert(), py::arg("omega").noconvert(), py::arg("blocksize"));
    m.def("block_jacobi_indexed", &_block_jacobi_indexed<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("indices").noconvert(), py::arg("omega").noconvert(), py::arg("blocksize"));
    m.def("block_jacobi_indexed", &_block_jacobi_indexed<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("indices").noconvert(), py::arg("omega").noconvert(), py::arg("blocksize"));
    m.def("block_jacobi_indexed", &_block_jacobi_indexed<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("indices").noconvert(), py::arg("omega").noconvert(), py::arg("blocksize"),
R"pbdoc(
Indexed Block Jacobi iteration.

Perform one iteration of block Jacobi relaxation on the linear
system Ax = b for a given set of (block) row indices. A is
stored in BSR format and x and b are column vectors. Damping
is controlled by the parameter omega.

Parameters
----------
Ap : array
    BSR row pointer.
Aj : array
    BSR index array.
Ax : array
    BSR data array, blocks assumed square.
x : array
    Approximate solution.
b : array
    Right hand side.
Tx : array
    Inverse of each diagonal block of A stored
    as a (n/blocksize, blocksize, blocksize) array.
indices : array
    Indices.
omega : float
    Damping parameter.
blocksize : int
    Dimension of square blocks in BSR matrix A.

Returns
-------
None
    Array x will be modified in place.)pbdoc");

    m.def("block_gauss_seidel", &_block_gauss_seidel<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("blocksize"));
    m.def("block_gauss_seidel", &_block_gauss_seidel<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("blocksize"));
    m.def("block_gauss_seidel", &_block_gauss_seidel<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("blocksize"));
    m.def("block_gauss_seidel", &_block_gauss_seidel<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"), py::arg("blocksize"),
R"pbdoc(
Block Gauss-Seidel iteration.

Perform one iteration of block Gauss-Seidel relaxation on
the linear system Ax = b, where A is stored in BSR format
and x and b are column vectors.

Refer to gauss_seidel for additional information regarding
row_start, row_stop, and row_step.

Parameters
----------
Ap : array
    BSR row pointer.
Aj : array
    BSR index array.
Ax : array
    BSR data array, blocks assumed square.
x : array
    Approximate solution.
b : array
    Right hand side.
Tx : array
    Inverse of each diagonal block of A stored
    as a (n/blocksize, blocksize, blocksize) array.
row_start : int
    Beginning of the sweep.
row_stop : int
    End of the sweep (i.e. one past the last unknown).
row_step : int
    Stride used during the sweep (may be negative).
blocksize : int
    Dimension of square blocks in BSR matrix A.

Returns
-------
None
    Result in place.)pbdoc");

    m.def("extract_subblocks", &_extract_subblocks<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Tx").noconvert(), py::arg("Tp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sp").noconvert(), py::arg("nsdomains"), py::arg("nrows"));
    m.def("extract_subblocks", &_extract_subblocks<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Tx").noconvert(), py::arg("Tp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sp").noconvert(), py::arg("nsdomains"), py::arg("nrows"));
    m.def("extract_subblocks", &_extract_subblocks<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Tx").noconvert(), py::arg("Tp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sp").noconvert(), py::arg("nsdomains"), py::arg("nrows"));
    m.def("extract_subblocks", &_extract_subblocks<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Tx").noconvert(), py::arg("Tp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sp").noconvert(), py::arg("nsdomains"), py::arg("nrows"),
R"pbdoc(
Extract diagonal blocks from A and insert into a linear array.

This is a helper function for overlapping_schwarz_csr.

Parameters
----------
Ap : array
    CSR row pointer.
Aj : array
    CSR index array. Must be sorted for each row.
Ax : array
    CSR data array, blocks assumed square.
Tx : array
    Inverse of each diagonal block of A, stored in row major.
Tp : array
    Pointer array into Tx indicating where the diagonal blocks start and stop.
Sj : array
    Indices of each subdomain. Must be sorted over each subdomain.
Sp : array
    Pointer array indicating where each subdomain starts and stops.
nsdomains : int
    Number of subdomains.
nrows : int
    Number of rows.

Returns
-------
None
    Array Tx will be modified inplace.)pbdoc");

    m.def("overlapping_schwarz_csr", &_overlapping_schwarz_csr<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("Tp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sp").noconvert(), py::arg("nsdomains"), py::arg("nrows"), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("overlapping_schwarz_csr", &_overlapping_schwarz_csr<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("Tp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sp").noconvert(), py::arg("nsdomains"), py::arg("nrows"), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("overlapping_schwarz_csr", &_overlapping_schwarz_csr<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("Tp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sp").noconvert(), py::arg("nsdomains"), py::arg("nrows"), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"));
    m.def("overlapping_schwarz_csr", &_overlapping_schwarz_csr<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Tx").noconvert(), py::arg("Tp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sp").noconvert(), py::arg("nsdomains"), py::arg("nrows"), py::arg("row_start"), py::arg("row_stop"), py::arg("row_step"),
R"pbdoc(
Overlapping Schwarz iteration.

Perform one iteration of an overlapping Schwarz relaxation on
the linear system Ax = b, where A is stored in CSR format
and x and b are column vectors.

Refer to gauss_seidel for additional information regarding
row_start, row_stop, and row_step.

Parameters
----------
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
Ax : array
    CSR data array, blocks assumed square.
x : array
    Approximate solution.
b : array
    Right hand side.
Tx : array
    Inverse of each diagonal block of A, stored in row major.
Tp : array
    Pointer array into Tx indicating where the diagonal blocks start and stop.
Sj : array
    Indices of each subdomain. Must be sorted over each subdomain.
Sp : array
    Pointer array indicating where each subdomain starts and stops.
nsdomains : int
    Number of subdomains.
nrows : int
    Number of rows.
row_start : int
    Subdomain processing start index.
row_stop : int
    Subdomain processing stop index.
row_step : int
    Subdomain processing step index.

Returns
-------
None
    Array x will be modified inplace.)pbdoc");

}

