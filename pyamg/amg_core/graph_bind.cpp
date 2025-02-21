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
void _floyd_warshall(
        const I num_nodes,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & D,
       py::array_t<I> & P,
       py::array_t<I> & C,
       py::array_t<I> & L,
       py::array_t<I> & m,
                const I a,
                const I N
                     )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_D = D.mutable_unchecked();
    auto py_P = P.mutable_unchecked();
    auto py_C = C.unchecked();
    auto py_L = L.unchecked();
    auto py_m = m.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_D = py_D.mutable_data();
    I *_P = py_P.mutable_data();
    const I *_C = py_C.data();
    const I *_L = py_L.data();
    const I *_m = py_m.data();

    return floyd_warshall<I, T>(
                num_nodes,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _D, D.shape(0),
                       _P, P.shape(0),
                       _C, C.shape(0),
                       _L, L.shape(0),
                       _m, m.shape(0),
                        a,
                        N
                                );
}

template<class I, class T>
bool _center_nodes(
        const I num_nodes,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
    py::array_t<I> & Cptr,
       py::array_t<T> & D,
       py::array_t<I> & P,
       py::array_t<I> & C,
       py::array_t<I> & L,
       py::array_t<T> & q,
       py::array_t<I> & c,
       py::array_t<T> & d,
       py::array_t<I> & m,
       py::array_t<I> & p,
      py::array_t<I> & pc,
       py::array_t<I> & s
                   )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_Cptr = Cptr.mutable_unchecked();
    auto py_D = D.mutable_unchecked();
    auto py_P = P.mutable_unchecked();
    auto py_C = C.mutable_unchecked();
    auto py_L = L.mutable_unchecked();
    auto py_q = q.mutable_unchecked();
    auto py_c = c.mutable_unchecked();
    auto py_d = d.mutable_unchecked();
    auto py_m = m.mutable_unchecked();
    auto py_p = p.mutable_unchecked();
    auto py_pc = pc.mutable_unchecked();
    auto py_s = s.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    I *_Cptr = py_Cptr.mutable_data();
    T *_D = py_D.mutable_data();
    I *_P = py_P.mutable_data();
    I *_C = py_C.mutable_data();
    I *_L = py_L.mutable_data();
    T *_q = py_q.mutable_data();
    I *_c = py_c.mutable_data();
    T *_d = py_d.mutable_data();
    I *_m = py_m.mutable_data();
    I *_p = py_p.mutable_data();
    I *_pc = py_pc.mutable_data();
    I *_s = py_s.mutable_data();

    return center_nodes<I, T>(
                num_nodes,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                    _Cptr, Cptr.shape(0),
                       _D, D.shape(0),
                       _P, P.shape(0),
                       _C, C.shape(0),
                       _L, L.shape(0),
                       _q, q.shape(0),
                       _c, c.shape(0),
                       _d, d.shape(0),
                       _m, m.shape(0),
                       _p, p.shape(0),
                      _pc, pc.shape(0),
                       _s, s.shape(0)
                              );
}

template<class I, class T>
void _bellman_ford(
        const I num_nodes,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<I> & c,
       py::array_t<T> & d,
       py::array_t<I> & m,
       py::array_t<I> & p
                   )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_c = c.unchecked();
    auto py_d = d.mutable_unchecked();
    auto py_m = m.mutable_unchecked();
    auto py_p = p.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    const I *_c = py_c.data();
    T *_d = py_d.mutable_data();
    I *_m = py_m.mutable_data();
    I *_p = py_p.mutable_data();

    return bellman_ford<I, T>(
                num_nodes,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _c, c.shape(0),
                       _d, d.shape(0),
                       _m, m.shape(0),
                       _p, p.shape(0)
                              );
}

template<class I, class T>
bool _bellman_ford_balanced(
        const I num_nodes,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<I> & c,
       py::array_t<T> & d,
       py::array_t<I> & m,
       py::array_t<I> & p,
      py::array_t<I> & pc,
       py::array_t<I> & s,
   const bool tiebreaking
                            )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_c = c.unchecked();
    auto py_d = d.mutable_unchecked();
    auto py_m = m.mutable_unchecked();
    auto py_p = p.mutable_unchecked();
    auto py_pc = pc.mutable_unchecked();
    auto py_s = s.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    const I *_c = py_c.data();
    T *_d = py_d.mutable_data();
    I *_m = py_m.mutable_data();
    I *_p = py_p.mutable_data();
    I *_pc = py_pc.mutable_data();
    I *_s = py_s.mutable_data();

    return bellman_ford_balanced<I, T>(
                num_nodes,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _c, c.shape(0),
                       _d, d.shape(0),
                       _m, m.shape(0),
                       _p, p.shape(0),
                      _pc, pc.shape(0),
                       _s, s.shape(0),
              tiebreaking
                                       );
}

template<class I, class T>
bool _most_interior_nodes(
        const I num_nodes,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<I> & c,
       py::array_t<T> & d,
       py::array_t<I> & m,
       py::array_t<I> & p
                          )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_c = c.mutable_unchecked();
    auto py_d = d.mutable_unchecked();
    auto py_m = m.mutable_unchecked();
    auto py_p = p.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    I *_c = py_c.mutable_data();
    T *_d = py_d.mutable_data();
    I *_m = py_m.mutable_data();
    I *_p = py_p.mutable_data();

    return most_interior_nodes<I, T>(
                num_nodes,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _c, c.shape(0),
                       _d, d.shape(0),
                       _m, m.shape(0),
                       _p, p.shape(0)
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
    coreassert
    maximal_independent_set_serial
    maximal_independent_set_parallel
    vertex_coloring_mis
    vertex_coloring_jones_plassmann
    vertex_coloring_LDF
    floyd_warshall
    center_nodes
    bellman_ford
    bellman_ford_balanced
    most_interior_nodes
    maximal_independent_set_k_parallel
    breadth_first_search
    connected_components
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("maximal_independent_set_serial", &_maximal_independent_set_serial<int, int>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("active"), py::arg("C"), py::arg("F"), py::arg("x").noconvert(),
R"pbdoc(
Serial maximal independent set.

Compute a maximal independent set for a graph stored in CSR format
using a greedy serial algorithm

Parameters
----------
num_rows : int
    Number of rows in A (number of vertices).
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
active : float
    Value used for active vertices.
C : float
    Value used to mark non-MIS vertices.
F : float
    Value used to mark MIS vertices.
x : array
    State of each vertex.

Returns
-------
int
    The number of nodes in the MIS.

Notes
-----
Only the vertices with values with x[i] == active are considered
when determining the MIS.  Upon return, all active vertices will
be assigned the value C or F depending on whether they are in the
MIS or not.)pbdoc");

    m.def("maximal_independent_set_parallel", &_maximal_independent_set_parallel<int, int, double>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("active"), py::arg("C"), py::arg("F"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("max_iters"),
R"pbdoc(
Parallel maximal independent set.

Compute a maximal independent set for a graph stored in CSR format
using a variant of Luby's parallel MIS algorithm.

Parameters
----------
num_rows : int
    Number of rows in A (number of vertices).
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
active : float
    Value used for active vertices.
C : float
    Value used to mark non-MIS vertices.
F : float
    Value used to mark MIS vertices.
x : array, output
    State of each vertex.
y : array
    Random values for each vertex.
max_iters : int
    Maximum number of iterations By default max_iters=-1 and no limit is imposed.

Returns
-------
int
    The number of nodes in the MIS.

Notes
-----
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
Compute a vertex coloring of a graph using the Jones-Plassmann algorithm.

Parameters
----------
num_rows : int
    Number of rows in A (number of vertices).
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
x : array, inplace
    Color of each vertex.
z : array
    Initial random values for each vertex.

Notes
-----
    Arrays x and z will be overwritten.

References
----------
.. [Jones92] Mark T. Jones and Paul E. Plassmann
   A Parallel Graph Coloring Heuristic
   SIAM Journal on Scientific Computing 14:3 (1993) 654--669
   http://citeseer.ist.psu.edu/jones92parallel.html)pbdoc");

    m.def("vertex_coloring_LDF", &_vertex_coloring_LDF<int, int, double>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("x").noconvert(), py::arg("y").noconvert(),
R"pbdoc(
Compute a vertex coloring of a graph using parallel Largest-Degree-First (LDF).

Parameters
----------
num_rows : int
    Number of rows in A (number of vertices).
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
x : array
    Color of each vertex.
y : array
    Initial random values for each vertex.

References
----------
.. [LDF] J. R. Allwright and R. Bordawekar and P. D. Coddington and K. Dincer and C. L. Martin
   A Comparison of Parallel Graph Coloring Algorithms
   DRAFT SCCS-666
   http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.4650)pbdoc");

    m.def("floyd_warshall", &_floyd_warshall<int, int>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("D").noconvert(), py::arg("P").noconvert(), py::arg("C").noconvert(), py::arg("L").noconvert(), py::arg("m").noconvert(), py::arg("a"), py::arg("N"));
    m.def("floyd_warshall", &_floyd_warshall<int, float>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("D").noconvert(), py::arg("P").noconvert(), py::arg("C").noconvert(), py::arg("L").noconvert(), py::arg("m").noconvert(), py::arg("a"), py::arg("N"));
    m.def("floyd_warshall", &_floyd_warshall<int, double>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("D").noconvert(), py::arg("P").noconvert(), py::arg("C").noconvert(), py::arg("L").noconvert(), py::arg("m").noconvert(), py::arg("a"), py::arg("N"),
R"pbdoc(
Floyd-Warshall on a subgraph or cluster of nodes in A.

Parameters
----------
num_nodes : int
    Number of nodes (number of rows in A).
Ap : array
    CSR row pointer for A, (num_nodes, 1).
Aj : array
    CSR column index for A, (num_edges, 1).
Ax : array
    CSR data array (edge weights), (num_edges, 1).
D : array
    FW distance array, (max_size, max_size).
P : array
    FW predecessor array, (max_size, max_size).
C : array
    FW global index for current cluster, (N, 1).
L : array
    FW local index for current cluster, (num_nodes, 1).
m : array
    Cluster index, (num_nodes, 1).
a : array
    Center of current cluster.
N : int
    Size of current cluster.

Notes
-----
    - There are no checks within this kernel
    - There is no initialization within this kernel
    - Ax > 0 is assumed
    - Only a slice of C is passed to Floyd–Warshall.  See center_nodes.
    - C[i] is the global index of i for i=0, ..., N in the current cluster
    - N = |C|
    - L = local indices, nx1 (-1 if not in the cluster)
    - assumes a fully connected (directed) graph

References
----------
.. [1] Graph Center:   https://en.wikipedia.org/wiki/Graph_center
.. [2] Floyd-Warshall: https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
.. [3] Graph Distance: https://en.wikipedia.org/wiki/Distance_(graph_theory))pbdoc");

    m.def("center_nodes", &_center_nodes<int, int>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Cptr").noconvert(), py::arg("D").noconvert(), py::arg("P").noconvert(), py::arg("C").noconvert(), py::arg("L").noconvert(), py::arg("q").noconvert(), py::arg("c").noconvert(), py::arg("d").noconvert(), py::arg("m").noconvert(), py::arg("p").noconvert(), py::arg("pc").noconvert(), py::arg("s").noconvert());
    m.def("center_nodes", &_center_nodes<int, float>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Cptr").noconvert(), py::arg("D").noconvert(), py::arg("P").noconvert(), py::arg("C").noconvert(), py::arg("L").noconvert(), py::arg("q").noconvert(), py::arg("c").noconvert(), py::arg("d").noconvert(), py::arg("m").noconvert(), py::arg("p").noconvert(), py::arg("pc").noconvert(), py::arg("s").noconvert());
    m.def("center_nodes", &_center_nodes<int, double>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Cptr").noconvert(), py::arg("D").noconvert(), py::arg("P").noconvert(), py::arg("C").noconvert(), py::arg("L").noconvert(), py::arg("q").noconvert(), py::arg("c").noconvert(), py::arg("d").noconvert(), py::arg("m").noconvert(), py::arg("p").noconvert(), py::arg("pc").noconvert(), py::arg("s").noconvert(),
R"pbdoc(
Update center nodes for a cluster

Parameters
----------
  num_nodes  : (IN) number of nodes (number of rows in A)
  Ap[]       : (IN) CSR row pointer for A                              (num_nodes x 1)
  Aj[]       : (IN) CSR column index for A                             (num_edges x 1)
  Ax[]       : (IN) CSR data array (edge weights)                      (num_edges x 1)
Cptr[]       : (INOUT) ptr to start of indices in C for each cluster   (num_clusters x 1)
   D[]       : (INOUT) FW distance array                               (max_size x max_size)
   P[]       : (INOUT) FW predecessor array                            (max_size x max_size)
   C[]       : (INOUT) FW global index for current cluster             (num_nodes x 1)
   L[]       : (INOUT) FW local index for current cluster              (num_nodes x 1)
   q         : (INOUT) FW work array for D**2                          (max_size x max_size)
   c         : (INOUT) cluster center                                  (num_clusters x 1)
   d         : (INOUT) distance to cluster center                      (num_nodes x 1)
   m         : (INOUT) cluster index                                   (num_nodes x 1)
   p         : (INOUT) predecessor on shortest path to center          (num_nodes x 1)
   pc        : (INOUT) predecessor count                               (num_nodes x 1)
   s         : (INOUT) cluster size                                    (num_clusters x 1)

Returns
-------
changed : flag to indicate a change in arrays D or P

Notes
-----
- sort into clusters first O(n)
    s: [4           2     4               ....
 Cptr: [0           4     6              11 ...
        |           |     |              |
        v           v     v              v
    C: [87 99 4  6  82 13 15 9  12 55 66 77 ...]
              ^  ^           ^
              |  |________   |_____
              |_____      |        |
                    |     |        |
    L: [            2     3        1
        ^  ^  ^  ^  ^  ^  ^  ^  ^  ^  ^
        |  |  |  |  |  |  |  |  |  |  |  ...
        0  1  2  3  4  5  6  7  8  9  10 ...
- pass pointer to start of each C[start,...., start+N]
- N is the cluster size)pbdoc");

    m.def("bellman_ford", &_bellman_ford<int, int>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("c").noconvert(), py::arg("d").noconvert(), py::arg("m").noconvert(), py::arg("p").noconvert());
    m.def("bellman_ford", &_bellman_ford<int, float>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("c").noconvert(), py::arg("d").noconvert(), py::arg("m").noconvert(), py::arg("p").noconvert());
    m.def("bellman_ford", &_bellman_ford<int, double>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("c").noconvert(), py::arg("d").noconvert(), py::arg("m").noconvert(), py::arg("p").noconvert(),
R"pbdoc(
Apply one iteration of Bellman-Ford iteration on a distance graph stored in CSR format.

Parameters
----------
num_nodes : int
    Number of nodes (number of rows in A).
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
Ax : array
    CSR data array (edge lengths).
c : array
    Cluster center.
d : array, inplace
    Distance to nearest center.
m : array, inplace
    Cluster index for each node.
p : array, inplace
    Predecssor on the shortest path to center.

Notes
-----
- There are no checks within this kernel.
- Ax is assumed to be positive

Initializations::

 - d[i] = 0 if i is a center, else inf
 - m[i] = 0 .. num_clusters if in a cluster, else -1
 - p[i] = -1

See Also
--------
pyamg.graph.bellman_ford

References
----------
.. [1] Bellman-Ford Wikipedia: http://en.wikipedia.org/wiki/Bellman-Ford_algorithm)pbdoc");

    m.def("bellman_ford_balanced", &_bellman_ford_balanced<int, int>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("c").noconvert(), py::arg("d").noconvert(), py::arg("m").noconvert(), py::arg("p").noconvert(), py::arg("pc").noconvert(), py::arg("s").noconvert(), py::arg("tiebreaking"));
    m.def("bellman_ford_balanced", &_bellman_ford_balanced<int, float>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("c").noconvert(), py::arg("d").noconvert(), py::arg("m").noconvert(), py::arg("p").noconvert(), py::arg("pc").noconvert(), py::arg("s").noconvert(), py::arg("tiebreaking"));
    m.def("bellman_ford_balanced", &_bellman_ford_balanced<int, double>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("c").noconvert(), py::arg("d").noconvert(), py::arg("m").noconvert(), py::arg("p").noconvert(), py::arg("pc").noconvert(), py::arg("s").noconvert(), py::arg("tiebreaking"),
R"pbdoc(
Bellman-Ford with a heuristic to balance cluster sizes

 This version is modified to break distance ties by assigning nodes
 to the cluster with the fewest points, while preserving cluster
 connectivity. This results in more balanced cluster sizes.

 Parameters
 ----------
  num_nodes  : (IN) number of nodes (number of rows in A)
  Ap[]       : (IN) CSR row pointer for A                              (num_nodes x 1)
  Aj[]       : (IN) CSR column index for A                             (num_edges x 1)
  Ax[]       : (IN) CSR data array (edge weights)                      (num_edges x 1)
   c         : (INOUT) cluster center                                  (num_clusters x 1)
   d         : (INOUT) distance to cluster center                      (num_nodes x 1)
   m         : (INOUT) cluster index                                   (num_nodes x 1)
   p         : (INOUT) predecessor on shortest path to center          (num_nodes x 1)
   pc        : (INOUT) number of predecessors                          (num_nodes x 1)
   s         : (INOUT) cluster size                                    (num_clusters x 1)

 Notes
 -----
 - There are no checks within this kernel.
 - Ax > 0 is assumed

 Initializations
 ---------------
 d[i] = 0 if i is a center, else 0
 m[i] = 0, ..., nclusters if i is in a cluster, else -1
 p = -1
 pc = 0
 s = 1

 See Also
 --------
 pyamg.graph.bellman_ford)pbdoc");

    m.def("most_interior_nodes", &_most_interior_nodes<int, int>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("c").noconvert(), py::arg("d").noconvert(), py::arg("m").noconvert(), py::arg("p").noconvert());
    m.def("most_interior_nodes", &_most_interior_nodes<int, float>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("c").noconvert(), py::arg("d").noconvert(), py::arg("m").noconvert(), py::arg("p").noconvert());
    m.def("most_interior_nodes", &_most_interior_nodes<int, double>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("c").noconvert(), py::arg("d").noconvert(), py::arg("m").noconvert(), py::arg("p").noconvert(),
R"pbdoc(
Find the most interior nodes.

Parameters
----------
num_nodes : int
    Number of nodes (number of rows in A).
Ap : array
    CSR row pointer for adjacency matrix A, (num_nodes, 1).
Aj : array
    CSR index array, (num_edges, 1).
Ax : array
    CSR data array (edge lengths), (num_edges, 1).
c : array, num_cluster
    Cluster centers, (num_clusters, 1).
d : array, num_nodes
    Distance to nearest seed, (num_nodes, 1).
m : array, num_nodes
    Cluster index for each node, (num_nodes, 1).
p : array
    Predecessor on shortest path to center, (num_nodes, 1).

Notes
-----
- There are no checks within this kernel.
- Ax is assumed to be positive

References
----------
.. [Bell2008] Nathan Bell, Algebraic Multigrid for Discrete Differential Forms
   PhD thesis (UIUC), August 2008.)pbdoc");

    m.def("maximal_independent_set_k_parallel", &_maximal_independent_set_k_parallel<int, int, double>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("k"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("max_iters"),
R"pbdoc(
Compute MIS-k.

Parameters
----------
num_rows : int
    Number of rows in A (number of vertices).
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
k : int
    Minimum separation between MIS vertices.
x : array, inplace
    State of each vertex (1 if in the MIS, 0 otherwise).
y : array
    Random values used during parallel MIS algorithm.
max_iters : int
    Maximum number of iterations to use (default, no limit).

Returns
-------
None
    In place.

Notes
-----
Compute a distance-k maximal independent set for a graph stored
in CSR format using a parallel algorithm.  An MIS-k is a set of
vertices such that all vertices in the MIS-k are separated by a
path of at least K+1 edges and no additional vertex can be added
to the set without destroying this property.  A standard MIS
is therefore a MIS-1.)pbdoc");

    m.def("breadth_first_search", &_breadth_first_search<int>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("seed"), py::arg("order").noconvert(), py::arg("level").noconvert(),
R"pbdoc(
Breadth first search.

Compute a breadth first search of a graph in CSR format
beginning at a given seed vertex.

Parameters
----------
num_rows : int
    Number of rows in A (number of vertices).
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
order : array, num_rows, inplace
    Records the order in which vertices were searched.
level : array, num_rows, inplace
    Records the level set of the searched vertices (i.e. the minimum distance to the seed).

Returns
-------
None
    In place.

Notes
-----
The values of the level must be initialized to -1.)pbdoc");

    m.def("connected_components", &_connected_components<int>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("components").noconvert(),
R"pbdoc(
Compute the connected components of a graph stored in CSR format.

Parameters
----------
num_rows : int
    Number of rows in A (number of vertices).
Ap : array
    CSR row pointer.
Aj : array
    CSR index array.
components : array, num_rows
    Component labels.

Returns
-------
None
    In place.

Notes
-----
Vertices belonging to each component are marked with a unique integer
in the range [0,K), where K is the number of components.)pbdoc");

}

