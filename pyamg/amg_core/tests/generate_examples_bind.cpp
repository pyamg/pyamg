// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "generate_examples.h"

namespace py = pybind11;

template <class I>
int _test1(
                const I n
           )
{
    return test1 <I>(
                        n
                     );
}

template <class I>
int _test2(
                const I n
           )
{
    return test2 <I>(
                        n
                     );
}

template <class I>
int _test3(
                const I n
           )
{
    return test3 <I>(
                        n
                     );
}

template <class I>
int _test4(
                const I n
           )
{
    return test4 <I>(
                        n
                     );
}

template <class I>
int _test5(
                const I n
           )
{
    return test5 <I>(
                        n
                     );
}

int _test6(
              const int n
           )
{
    return test6(
                        n
                 );
}

int _test7(
              const int n
           )
{
    return test7(
                        n
                 );
}

int _test8(
              const int n,
                    int m,
  py::array_t<double> & x,
     py::array_t<int> & J
           )
{
    auto py_x = x.mutable_unchecked();
    auto py_J = J.mutable_unchecked();
    double *_x = py_x.mutable_data();
    int *_J = py_J.mutable_data();

    return test8(
                        n,
                        m,
                       _x, x.size(),
                       _J, J.size()
                 );
}

template <class I, class T, class F>
int _test9(
       py::array_t<I> & J,
       py::array_t<T> & x,
       py::array_t<F> & y
           )
{
    auto py_J = J.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_y = y.mutable_unchecked();
    const I *_J = py_J.data();
    T *_x = py_x.mutable_data();
    F *_y = py_y.mutable_data();

    return test9 <I, T, F>(
                       _J, J.size(),
                       _x, x.size(),
                       _y, y.size()
                           );
}

PYBIND11_MODULE(generate_examples, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for generate_examples.h

    Methods
    -------
    test1
    test2
    test3
    test4
    test5
    test6
    test7
    test8
    test9
    )pbdoc";

    m.def("test1", &_test1<int>,
        py::arg("n"));
    m.def("test1", &_test1<long int>,
        py::arg("n"));
    m.def("test1", &_test1<double>,
        py::arg("n"),
R"pbdoc(
//
// Testing docstring
//
)pbdoc");

    m.def("test2", &_test2<int>,
        py::arg("n"));
    m.def("test2", &_test2<long int>,
        py::arg("n"));
    m.def("test2", &_test2<double>,
        py::arg("n"),
R"pbdoc(
// Testing docstring
)pbdoc");

    m.def("test3", &_test3<int>,
        py::arg("n"));
    m.def("test3", &_test3<long int>,
        py::arg("n"));
    m.def("test3", &_test3<double>,
        py::arg("n"),
R"pbdoc(
/*
 * Testing a docstring
 *
 */
)pbdoc");

    m.def("test4", &_test4<int>,
        py::arg("n"));
    m.def("test4", &_test4<long int>,
        py::arg("n"));
    m.def("test4", &_test4<double>,
        py::arg("n"),
R"pbdoc(
/* Testing a docstring */
)pbdoc");

    m.def("test5", &_test5<int>,
        py::arg("n"));
    m.def("test5", &_test5<long int>,
        py::arg("n"));
    m.def("test5", &_test5<double>,
        py::arg("n"),
R"pbdoc(
)pbdoc");

    m.def("test6", &_test6,
        py::arg("n"),
R"pbdoc(
/* Testing a docstring */
)pbdoc");

    m.def("test7", &_test7,
        py::arg("n"),
R"pbdoc(
)pbdoc");

    m.def("test8", &_test8,
        py::arg("n"), py::arg("m"), py::arg("x").noconvert(), py::arg("J").noconvert(),
R"pbdoc(
// untemplated
)pbdoc");

    m.def("test9", &_test9<int, double, std::complex<double>>,
        py::arg("J").noconvert(), py::arg("x").noconvert(), py::arg("y").noconvert(),
R"pbdoc(
// some class
)pbdoc");

}

