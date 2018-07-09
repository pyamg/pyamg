// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "graph.h"

namespace py = pybind11;

template<class I, class T>
I _maximal_independent_set_serial(
         const I num_rows,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
           const T active,
                const T C,
                const T F,
       py::array_t<T> & x
                                  )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_x = x.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    T *_x = py_x.mutable_data();

    return maximal_independent_set_serial<I, T>(
                 num_rows,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                   active,
                        C,
                        F,
                       _x, x.shape(0)
                                                );
}

template<class I, class T, class R>
I _maximal_independent_set_parallel(
         const I num_rows,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
           const T active,
                const T C,
                const T F,
       py::array_t<T> & x,
       py::array_t<R> & y,
        const I max_iters
                                    )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_y = y.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    T *_x = py_x.mutable_data();
    const R *_y = py_y.data();

    return maximal_independent_set_parallel<I, T, R>(
                 num_rows,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                   active,
                        C,
                        F,
                       _x, x.shape(0),
                       _y, y.shape(0),
                max_iters
                                                     );
}

template<class I, class T>
T _vertex_coloring_mis(
         const I num_rows,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
       py::array_t<T> & x
                       )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_x = x.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    T *_x = py_x.mutable_data();

    return vertex_coloring_mis<I, T>(
                 num_rows,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                       _x, x.shape(0)
                                     );
}

template<class I, class T, class R>
T _vertex_coloring_jones_plassmann(
         const I num_rows,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
       py::array_t<T> & x,
       py::array_t<R> & z
                                   )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_z = z.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    T *_x = py_x.mutable_data();
    R *_z = py_z.mutable_data();

    return vertex_coloring_jones_plassmann<I, T, R>(
                 num_rows,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                       _x, x.shape(0),
                       _z, z.shape(0)
                                                    );
}

template<class I, class T, class R>
T _vertex_coloring_LDF(
         const I num_rows,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
       py::array_t<T> & x,
       py::array_t<R> & y
                       )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_y = y.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    T *_x = py_x.mutable_data();
    const R *_y = py_y.data();

    return vertex_coloring_LDF<I, T, R>(
                 num_rows,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                       _x, x.shape(0),
                       _y, y.shape(0)
                                        );
}

template<class I, class T>
void _bellman_ford(
         const I num_rows,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<I> & z
                   )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_z = z.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    I *_z = py_z.mutable_data();

    return bellman_ford<I, T>(
                 num_rows,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _z, z.shape(0)
                              );
}

template<class I, class T>
void _lloyd_cluster(
         const I num_rows,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
        const I num_seeds,
       py::array_t<T> & x,
       py::array_t<I> & w,
       py::array_t<I> & z
                    )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_w = w.mutable_unchecked();
    auto py_z = z.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    I *_w = py_w.mutable_data();
    I *_z = py_z.mutable_data();

    return lloyd_cluster<I, T>(
                 num_rows,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                num_seeds,
                       _x, x.shape(0),
                       _w, w.shape(0),
                       _z, z.shape(0)
                               );
}

template<class I, class T, class R>
void _maximal_independent_set_k_parallel(
         const I num_rows,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
                const I k,
       py::array_t<T> & x,
       py::array_t<R> & y,
        const I max_iters
                                         )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_y = y.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    T *_x = py_x.mutable_data();
    const R *_y = py_y.data();

    return maximal_independent_set_k_parallel<I, T, R>(
                 num_rows,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                        k,
                       _x, x.shape(0),
                       _y, y.shape(0),
                max_iters
                                                       );
}

template <class I>
void _breadth_first_search(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
             const I seed,
   py::array_t<I> & order,
   py::array_t<I> & level
                           )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_order = order.mutable_unchecked();
    auto py_level = level.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    I *_order = py_order.mutable_data();
    I *_level = py_level.mutable_data();

    return breadth_first_search <I>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                     seed,
                   _order, order.shape(0),
                   _level, level.shape(0)
                                    );
}

template <class I>
I _connected_components(
        const I num_nodes,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
py::array_t<I> & components
                        )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_components = components.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    I *_components = py_components.mutable_data();

    return connected_components <I>(
                num_nodes,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
              _components, components.shape(0)
                                    );
}

PYBIND11_MODULE(graph, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for graph.h

    Methods
    -------
    maximal_independent_set_serial
    maximal_independent_set_parallel
    vertex_coloring_mis
    vertex_coloring_jones_plassmann
    vertex_coloring_LDF
    bellman_ford
    lloyd_cluster
    maximal_independent_set_k_parallel
    breadth_first_search
    connected_components
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("maximal_independent_set_serial", &_maximal_independent_set_serial<int, int>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("active"), py::arg("C"), py::arg("F"), py::arg("x").noconvert(),
R"pbdoc(
Compute a maximal independent set for a graph stored in CSR format
 using a greedy serial algorithm

 Parameters
     num_rows   - number of rows in A (number of vertices)
     Ap[]       - CSR row pointer
     Aj[]       - CSR index array
     active     - value used for active vertices        (input)
      C         - value used to mark non-MIS vertices   (output)
      F         - value used to mark MIS vertices       (output)
     x[]        - state of each vertex


 Returns:
     The number of nodes in the MIS.

 Notes:
     Only the vertices with values with x[i] == active are considered
     when determining the MIS.  Upon return, all active vertices will
     be assigned the value C or F depending on whether they are in the
     MIS or not.)pbdoc");

    m.def("maximal_independent_set_parallel", &_maximal_independent_set_parallel<int, int, double>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("active"), py::arg("C"), py::arg("F"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("max_iters"),
R"pbdoc(
Compute a maximal independent set for a graph stored in CSR format
 using a variant of Luby's parallel MIS algorithm

 Parameters
     num_rows   - number of rows in A (number of vertices)
     Ap[]       - CSR row pointer
     Aj[]       - CSR index array
     active     - value used for active vertices        (input)
      C         - value used to mark non-MIS vertices   (output)
      F         - value used to mark MIS vertices       (output)
     x[]        - state of each vertex
     y[]        - random values for each vertex
     max_iters  - maximum number of iterations
                  by default max_iters=-1 and no limit
                  is imposed

 Returns:
     The number of nodes in the MIS.

 Notes:
     Only the vertices with values with x[i] == active are considered
     when determining the MIS.  Upon return, all active vertices will
     be assigned the value C or F depending on whether they are in the
     MIS or not.)pbdoc");

    m.def("vertex_coloring_mis", &_vertex_coloring_mis<int, int>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("x").noconvert(),
R"pbdoc(
Compute a vertex coloring for a graph stored in CSR format.

 The coloring is computed by removing maximal independent sets
 of vertices from the graph.  Specifically, at iteration i an
 independent set of the remaining subgraph is constructed and
 assigned color i.

 Returns the K, the number of colors used in the coloring.
 On return x[i] \in [0,1, ..., K - 1] will contain the color
 of the i-th vertex.)pbdoc");

    m.def("vertex_coloring_jones_plassmann", &_vertex_coloring_jones_plassmann<int, int, double>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("x").noconvert(), py::arg("z").noconvert(),
R"pbdoc(
Compute a vertex coloring of a graph using the Jones-Plassmann algorithm

 Parameters
     num_rows   - number of rows in A (number of vertices)
     Ap[]       - CSR row pointer
     Aj[]       - CSR index array
     x[]        - color of each vertex
     y[]        - initial random values for each vertex

 Notes:
     Arrays x and y will be overwritten

 References:
     Mark T. Jones and Paul E. Plassmann
     A Parallel Graph Coloring Heuristic
     SIAM Journal on Scientific Computing 14:3 (1993) 654--669
     http://citeseer.ist.psu.edu/jones92parallel.html)pbdoc");

    m.def("vertex_coloring_LDF", &_vertex_coloring_LDF<int, int, double>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("x").noconvert(), py::arg("y").noconvert(),
R"pbdoc(
Compute a vertex coloring of a graph using the parallel
Largest-Degree-First (LDF) algorithm

 Parameters
     num_rows   - number of rows in A (number of vertices)
     Ap[]       - CSR row pointer
     Aj[]       - CSR index array
     x[]        - color of each vertex
     y[]        - initial random values for each vertex

  References:
    J. R. Allwright and R. Bordawekar and P. D. Coddington and K. Dincer and C. L. Martin
    A Comparison of Parallel Graph Coloring Algorithms
    DRAFT SCCS-666
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.4650)pbdoc");

    m.def("bellman_ford", &_bellman_ford<int, int>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("z").noconvert());
    m.def("bellman_ford", &_bellman_ford<int, float>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("z").noconvert());
    m.def("bellman_ford", &_bellman_ford<int, double>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("z").noconvert(),
R"pbdoc(
Apply one iteration of Bellman-Ford iteration on a distance
graph stored in CSR format.

 Parameters
     num_rows   - number of rows in A (number of vertices)
     Ap[]       - CSR row pointer
     Aj[]       - CSR index array
     Ax[]       - CSR data array (edge lengths)
     x[]        - (current) distance to nearest center
     y[]        - (current) index of nearest center

 References:
     http://en.wikipedia.org/wiki/Bellman-Ford_algorithm)pbdoc");

    m.def("lloyd_cluster", &_lloyd_cluster<int, int>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("num_seeds"), py::arg("x").noconvert(), py::arg("w").noconvert(), py::arg("z").noconvert());
    m.def("lloyd_cluster", &_lloyd_cluster<int, float>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("num_seeds"), py::arg("x").noconvert(), py::arg("w").noconvert(), py::arg("z").noconvert());
    m.def("lloyd_cluster", &_lloyd_cluster<int, double>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("num_seeds"), py::arg("x").noconvert(), py::arg("w").noconvert(), py::arg("z").noconvert(),
R"pbdoc(
Perform Lloyd clustering on a distance graph

 Parameters
     num_rows       - number of rows in A (number of vertices)
     Ap[]           - CSR row pointer
     Aj[]           - CSR index array
     Ax[]           - CSR data array (edge lengths)
     x[num_rows]    - distance to nearest seed
     y[num_rows]    - cluster membership
     z[num_centers] - cluster centers

 References
     Nathan Bell
     Algebraic Multigrid for Discrete Differential Forms
     PhD thesis (UIUC), August 2008)pbdoc");

    m.def("maximal_independent_set_k_parallel", &_maximal_independent_set_k_parallel<int, int, double>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("k"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("max_iters"),
R"pbdoc(
Compute a distance-k maximal independent set for a graph stored
 in CSR format using a parallel algorithm.  An MIS-k is a set of
 vertices such that all vertices in the MIS-k are separated by a
 path of at least K+1 edges and no additional vertex can be added
 to the set without destroying this property.  A standard MIS
 is therefore a MIS-1.

 Parameters
     num_rows   - number of rows in A (number of vertices)
     Ap[]       - CSR row pointer
     Aj[]       - CSR index array
     k          - minimum separation between MIS vertices
     x[]        - state of each vertex (1 if in the MIS, 0 otherwise)
     y[]        - random values used during parallel MIS algorithm
     max_iters  - maximum number of iterations to use (default, no limit))pbdoc");

    m.def("breadth_first_search", &_breadth_first_search<int>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("seed"), py::arg("order").noconvert(), py::arg("level").noconvert(),
R"pbdoc(
Compute a breadth first search of a graph in CSR format
 beginning at a given seed vertex.

 Parameters
     num_rows         - number of rows in A (number of vertices)
     Ap[]             - CSR row pointer
     Aj[]             - CSR index array
     order[num_rows]  - records the order in which vertices were searched
     level[num_rows]  - records the level set of the searched vertices (i.e. the minimum distance to the seed)

 Notes:
     The values of the level must be initialized to -1)pbdoc");

    m.def("connected_components", &_connected_components<int>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("components").noconvert(),
R"pbdoc(
Compute the connected components of a graph stored in CSR format.

 Vertices belonging to each component are marked with a unique integer
 in the range [0,K), where K is the number of components.

 Parameters
     num_rows             - number of rows in A (number of vertices)
     Ap[]                 - CSR row pointer
     Aj[]                 - CSR index array
     components[num_rows] - component labels)pbdoc");

}

