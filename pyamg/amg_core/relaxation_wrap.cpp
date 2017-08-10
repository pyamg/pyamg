#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

template<class I, class T, class F>
void gauss_seidel_orig(const I Ap[], const int Ap_size,
                   const I Aj[], const int Aj_size,
                   const T Ax[], const int Ax_size,
                         T  x[], const int  x_size,
                   const T  b[], const int  b_size,
                   const I row_start,
                   const I row_stop,
                   const I row_step)
{
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        T rsum = 0;
        T diag = 0;

        for(I jj = start; jj < end; jj++){
            I j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*x[j];
        }

        if (diag != (F) 0.0){
            x[i] = (b[i] - rsum)/diag;
        }
    }
}

template<class I, class T, class F>
void gauss_seidel1(py::array &Ap,
                  const py::array &Aj,
                  const py::array &Ax,
                        py::array  &x,
                  const py::array  &b,
                  const I row_start,
                  const I row_stop,
                  const I row_step)
{
    auto rrAp = Ap.unchecked<I,1>();
    auto rrAj = Aj.unchecked<I,1>();
    auto rrAx = Ax.unchecked<T,1>();
    auto  rrx =  x.mutable_unchecked<T,1>();
    auto  rrb =  b.unchecked<T,1>();

    const I *_Ap;
    const I *_Aj;
    const T *_Ax;
          T *_x;
    const T *_b;

    _Ap = rrAp.data(0);
    _Aj = rrAj.data(0);
    _Ax = rrAx.data(0);
     _x =  rrx.mutable_data(0);
     _b =  rrb.data(0);

    gauss_seidel_orig<I,T,F>(_Ap, Ap.size(),
                  _Aj, Aj.size(),
                  _Ax, Ax.size(),
                   _x,  x.size(),
                   _b,  b.size(),
                   row_start,
                   row_stop,
                   row_step);
}

template<class T>
void tmp(py::array &v)
{
    auto rr = v.unchecked<T,1>();
    const T *vv;
    vv = rr.data(0);
}

template<class I, class T, class F>
void gauss_seidel2(py::array &Apraw,
                   py::array &Ajraw,
                   py::array &Axraw,
                   py::array &xraw,
                   py::array &braw,
                   const I row_start,
                   const I row_stop,
                   const I row_step)
{
    auto rrAp = Apraw.unchecked<I,1>();
    auto rrAj = Ajraw.unchecked<I,1>();
    auto rrAx = Axraw.unchecked<T,1>();
    auto  rrx =  xraw.mutable_unchecked<T,1>();
    auto  rrb =  braw.unchecked<T,1>();

    const I *Ap;
    const I *Aj;
    const T *Ax;
          T *x;
    const T *b;

    Ap = rrAp.data(0);
    Aj = rrAj.data(0);
    Ax = rrAx.data(0);
     x =  rrx.mutable_data(0);
     b =  rrb.data(0);

    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        T rsum = 0;
        T diag = 0;

        for(I jj = start; jj < end; jj++){
            I j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*x[j];
        }

        if (diag != (F) 0.0){
            x[i] = (b[i] - rsum)/diag;
        }
    }
}

template<class I, class T, class F>
void gauss_seidel3(py::array &Apraw,
                   py::array &Ajraw,
                   py::array &Axraw,
                   py::array &xraw,
                   py::array &braw,
                   const I row_start,
                   const I row_stop,
                   const I row_step)
{
    auto rrAp = Apraw.unchecked<I,1>();
    auto rrAj = Ajraw.unchecked<I,1>();
    auto rrAx = Axraw.unchecked<T,1>();
    auto  rrx =  xraw.mutable_unchecked<T,1>();
    auto  rrb =  braw.unchecked<T,1>();
}


PYBIND11_PLUGIN(relaxation_wrap) {
    py::module m("relaxation_wrap", "pybind11 example plugin");

    m.def("gauss_seidel_proto1", gauss_seidel1<int, double, double>, "Gauss-Seidel");
    m.def("gauss_seidel_proto2", gauss_seidel2<int, double, double>, "Gauss-Seidel");
    m.def("gauss_seidel_proto3", gauss_seidel3<int, double, double>, "Gauss-Seidel");
    //m.def("gauss_seidel_proto", tmp<double>, "Gauss-Seidel");

    return m.ptr();
}
