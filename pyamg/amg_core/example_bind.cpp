#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <iostream>
#include "relaxation.h"

namespace py = pybind11;

template<class I, class T, class F>
void _gauss_seidel(py::array_t<I> &Ap,
                   py::array_t<I> &Aj,
                   py::array_t<T> &Ax,
                   py::array_t<T>  &x,
                   py::array_t<T>  &b,
                   I row_start,
                   I row_stop,
                   I row_step)
{
    auto rrAp = Ap.unchecked();
    auto rrAj = Aj.unchecked();
    auto rrAx = Ax.unchecked();
    auto  rrx =  x.mutable_unchecked();
    auto  rrb =  b.unchecked();

    const I *_Ap = rrAp.data(0);
    const I *_Aj = rrAj.data(0);
    const T *_Ax = rrAx.data(0);
          T *_x  =  rrx.mutable_data(0);
    const T *_b  =  rrb.data(0);

    gauss_seidel<I,T,F>(_Ap, Ap.size(),
                        _Aj, Aj.size(),
                        _Ax, Ax.size(),
                         _x,  x.size(),
                         _b,  b.size(),
                        row_start,
                        row_stop,
                        row_step);
}

#define NC py::arg().noconvert()
#define YC py::arg()
PYBIND11_PLUGIN(relaxation) {
    py::module m("relaxation", R"pbdoc(
    pybind11 wrappers for relxation.h

    Relaxation Methods
    ------------------
    Gauss Seidel
    Jacobi
    )pbdoc");

    m.def("gauss_seidel", &_gauss_seidel<int, float, float>, NC, NC, NC, NC, NC, YC, YC, YC, "A function which adds two numbers (float)");
    m.def("gauss_seidel", &_gauss_seidel<int, double, double>, NC, NC, NC, NC, NC, YC, YC, YC, "A function which adds two numbers (double)");
    //m.def("gauss_seidel", &_gauss_seidel<int, std::complex<float>, float>);
    //m.def("gauss_seidel", &_gauss_seidel<int, std::complex<double>, double>);

    return m.ptr();
}
#undef NC
#undef YC
