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

template<class I>
void _cluster_node_incidence(
        const I num_nodes,
     const I num_clusters,
      py::array_t<I> & cm,
     py::array_t<I> & ICp,
     py::array_t<I> & ICi,
       py::array_t<I> & L
                             )
{
    auto py_cm = cm.unchecked();
    auto py_ICp = ICp.mutable_unchecked();
    auto py_ICi = ICi.mutable_unchecked();
    auto py_L = L.mutable_unchecked();
    const I *_cm = py_cm.data();
    I *_ICp = py_ICp.mutable_data();
    I *_ICi = py_ICi.mutable_data();
    I *_L = py_L.mutable_data();

    return cluster_node_incidence<I>(
                num_nodes,
             num_clusters,
                      _cm, cm.shape(0),
                     _ICp, ICp.shape(0),
                     _ICi, ICi.shape(0),
                       _L, L.shape(0)
                                     );
}

template<class I, class T>
I _cluster_center(
                const I a,
        const I num_nodes,
     const I num_clusters,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<I> & cm,
     py::array_t<I> & ICp,
     py::array_t<I> & ICi,
       py::array_t<I> & L
                  )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_cm = cm.unchecked();
    auto py_ICp = ICp.unchecked();
    auto py_ICi = ICi.unchecked();
    auto py_L = L.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    const I *_cm = py_cm.data();
    const I *_ICp = py_ICp.data();
    const I *_ICi = py_ICi.data();
    const I *_L = py_L.data();

    return cluster_center<I, T>(
                        a,
                num_nodes,
             num_clusters,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _cm, cm.shape(0),
                     _ICp, ICp.shape(0),
                     _ICi, ICi.shape(0),
                       _L, L.shape(0)
                                );
}

template<class I, class T>
void _bellman_ford(
        const I num_nodes,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & d,
      py::array_t<I> & cm
                   )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_d = d.mutable_unchecked();
    auto py_cm = cm.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_d = py_d.mutable_data();
    I *_cm = py_cm.mutable_data();

    return bellman_ford<I, T>(
                num_nodes,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _d, d.shape(0),
                      _cm, cm.shape(0)
                              );
}

template<class I, class T>
void _lloyd_cluster(
        const I num_nodes,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
     const I num_clusters,
       py::array_t<T> & d,
      py::array_t<I> & cm,
       py::array_t<I> & c
                    )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_d = d.mutable_unchecked();
    auto py_cm = cm.mutable_unchecked();
    auto py_c = c.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_d = py_d.mutable_data();
    I *_cm = py_cm.mutable_data();
    I *_c = py_c.mutable_data();

    return lloyd_cluster<I, T>(
                num_nodes,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
             num_clusters,
                       _d, d.shape(0),
                      _cm, cm.shape(0),
                       _c, c.shape(0)
                               );
}

template<class I, class T>
void _lloyd_cluster_exact(
        const I num_nodes,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
     const I num_clusters,
       py::array_t<T> & d,
      py::array_t<I> & cm,
       py::array_t<I> & c
                          )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_d = d.mutable_unchecked();
    auto py_cm = cm.mutable_unchecked();
    auto py_c = c.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_d = py_d.mutable_data();
    I *_cm = py_cm.mutable_data();
    I *_c = py_c.mutable_data();

    return lloyd_cluster_exact<I, T>(
                num_nodes,
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
             num_clusters,
                       _d, d.shape(0),
                      _cm, cm.shape(0),
                       _c, c.shape(0)
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
    cluster_node_incidence
    cluster_center
    bellman_ford
    lloyd_cluster
    lloyd_cluster_exact
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
----------
num_rows : int
    Number of rows in A (number of vertices)
Ap : array
    CSR row pointer
Aj : array
    CSR index array
active : float-like
    Value used for active vertices
C : float-like
    Value used to mark non-MIS vertices
F : float-like
    Value used to mark MIS vertices
x : array, inplace output
    State of each vertex

Returns
-------
N : int
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
Compute a maximal independent set for a graph stored in CSR format
 using a variant of Luby's parallel MIS algorithm

 Parameters
 ----------
 num_rows : int
     number of rows in A (number of vertices)
 Ap : array
     CSR row pointer
 Aj : array
     CSR index array
 active : float
     value used for active vertices
 C : float
     value used to mark non-MIS vertices
 F : float
     value used to mark MIS vertices
 x : array, output
     state of each vertex
 y : array
     random values for each vertex
 max_iters : int
     maximum number of iterations By default max_iters=-1 and no limit is imposed

 Returns
 -------
 N : int
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
Compute a vertex coloring of a graph using the Jones-Plassmann algorithm

Parameters
----------
num_rows : int
    number of rows in A (number of vertices)
Ap : array
    CSR row pointer
Aj : array
    CSR index array
x : array, inplace
    color of each vertex
y : array
    initial random values for each vertex

Notes
-----
    Arrays x and y will be overwritten

References
----------
.. [Jones92] Mark T. Jones and Paul E. Plassmann
   A Parallel Graph Coloring Heuristic
   SIAM Journal on Scientific Computing 14:3 (1993) 654--669
   http://citeseer.ist.psu.edu/jones92parallel.html)pbdoc");

    m.def("vertex_coloring_LDF", &_vertex_coloring_LDF<int, int, double>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("x").noconvert(), py::arg("y").noconvert(),
R"pbdoc(
Compute a vertex coloring of a graph using the parallel
Largest-Degree-First (LDF) algorithm

Parameters
----------
num_rows : int
    number of rows in A (number of vertices)
Ap : array
    CSR row pointer
Aj : array
    CSR index array
x : array
    color of each vertex
y : array
    initial random values for each vertex

References
----------
.. [LDF] J. R. Allwright and R. Bordawekar and P. D. Coddington and K. Dincer and C. L. Martin
   A Comparison of Parallel Graph Coloring Algorithms
   DRAFT SCCS-666
   http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.4650)pbdoc");

    m.def("cluster_node_incidence", &_cluster_node_incidence<int>,
        py::arg("num_nodes"), py::arg("num_clusters"), py::arg("cm").noconvert(), py::arg("ICp").noconvert(), py::arg("ICi").noconvert(), py::arg("L").noconvert(),
R"pbdoc(
Compute the incidence matrix for a clustering

Parameters
----------
num_nodes : int
    number of nodes
num_clusters : int
    number of clusters
cm : array, num_nodes
    cluster index for each node
ICp : arrayt, num_clusters+1, inplace
    CSC column pointer array for I
ICi : array, num_nodes, inplace
    CSC column indexes for I
L : array, num_nodes, inplace
    Local index mapping

Notes
-----
I = Incidence matrix between nodes and clusters (num_nodes x num_clusters)
I[i,a] = 1 if node i is in cluster a, otherwise 0

Cluster indexes: a,b,c in 1..num_clusters
Global node indexes: i,j,k in 1..num_rows
Local node indexes: pair (a,m) where a is cluster and m in 1..num_nodes_in_cluster

We store I in both CSC and CSR formats because we want to be able
to map global <-> local node indexes. However, I in CSR format is
simply the cm array, so we only need to compute CSC format.

IC = (ICp,ICi)    = I in CSC format (don't store ICx because it's always 1).

IR = (IRa) = (cm) = I in CSR format (don't store IRp because we
have exactly one nonzero entry per row, and don't store IRx because it's always 1). This is
just the cm array.

Converting local (a,m) -> global i:   i = ICi[ICp[a] + m]
Converting global i -> local (a,m):   a = cm[i], m = L[i]

L is an additional vector (length num_rows) to store local indexes.)pbdoc");

    m.def("cluster_center", &_cluster_center<int, int>,
        py::arg("a"), py::arg("num_nodes"), py::arg("num_clusters"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("cm").noconvert(), py::arg("ICp").noconvert(), py::arg("ICi").noconvert(), py::arg("L").noconvert());
    m.def("cluster_center", &_cluster_center<int, float>,
        py::arg("a"), py::arg("num_nodes"), py::arg("num_clusters"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("cm").noconvert(), py::arg("ICp").noconvert(), py::arg("ICi").noconvert(), py::arg("L").noconvert());
    m.def("cluster_center", &_cluster_center<int, double>,
        py::arg("a"), py::arg("num_nodes"), py::arg("num_clusters"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("cm").noconvert(), py::arg("ICp").noconvert(), py::arg("ICi").noconvert(), py::arg("L").noconvert(),
R"pbdoc(
Apply Floyd–Warshall to cluster "a" and use the result to find the
cluster center

Parameters
----------
a : int
    cluster index to find the center of
num_nodes : int
    number of nodes
num_clusters : int
    number of clusters
Ap : array
    CSR row pointer
Aj : array
    CSR index array
Ax : array
    CSR data array (edge lengths)
cm : array, num_nodes
    cluster index for each node
ICp : array, num_clusters+1
    CSC column pointer array for I
ICi : array, num_nodes
    CSC column indexes for I
L : array, num_nodes
    Local index mapping

Returns
-------
i : int
    global node index of center of cluster a

References
----------
.. [1] Graph Center:   https://en.wikipedia.org/wiki/Graph_center
.. [2] Floyd-Warshall: https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
.. [3] Graph Distance: https://en.wikipedia.org/wiki/Distance_(graph_theory))pbdoc");

    m.def("bellman_ford", &_bellman_ford<int, int>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("d").noconvert(), py::arg("cm").noconvert());
    m.def("bellman_ford", &_bellman_ford<int, float>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("d").noconvert(), py::arg("cm").noconvert());
    m.def("bellman_ford", &_bellman_ford<int, double>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("d").noconvert(), py::arg("cm").noconvert(),
R"pbdoc(
Apply one iteration of Bellman-Ford iteration on a distance
graph stored in CSR format.

Parameters
----------
num_nodes : int
    number of nodes (number of rows in A)
Ap : array
    CSR row pointer
Aj : array
    CSR index array
Ax : array
    CSR data array (edge lengths)
d : array, inplace
    distance to nearest center
cm : array, inplace
    cluster index for each node

References
----------
.. [1] Bellman-Ford Wikipedia: http://en.wikipedia.org/wiki/Bellman-Ford_algorithm)pbdoc");

    m.def("lloyd_cluster", &_lloyd_cluster<int, int>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("num_clusters"), py::arg("d").noconvert(), py::arg("cm").noconvert(), py::arg("c").noconvert());
    m.def("lloyd_cluster", &_lloyd_cluster<int, float>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("num_clusters"), py::arg("d").noconvert(), py::arg("cm").noconvert(), py::arg("c").noconvert());
    m.def("lloyd_cluster", &_lloyd_cluster<int, double>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("num_clusters"), py::arg("d").noconvert(), py::arg("cm").noconvert(), py::arg("c").noconvert(),
R"pbdoc(
Perform one iteration of Lloyd clustering on a distance graph

Parameters
----------
num_nodes : int
    number of nodes (number of rows in A)
Ap : array
    CSR row pointer for adjacency matrix A
Aj : array
    CSR index array
Ax : array
    CSR data array (edge lengths)
num_clusters : int
    number of clusters (seeds)
d : array, num_nodes
    distance to nearest seed
cm : array, num_nodes
    cluster index for each node
c : array, num_clusters
    cluster centers

References
----------
.. [Bell2008] Nathan Bell, Algebraic Multigrid for Discrete Differential Forms
   PhD thesis (UIUC), August 2008)pbdoc");

    m.def("lloyd_cluster_exact", &_lloyd_cluster_exact<int, int>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("num_clusters"), py::arg("d").noconvert(), py::arg("cm").noconvert(), py::arg("c").noconvert());
    m.def("lloyd_cluster_exact", &_lloyd_cluster_exact<int, float>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("num_clusters"), py::arg("d").noconvert(), py::arg("cm").noconvert(), py::arg("c").noconvert());
    m.def("lloyd_cluster_exact", &_lloyd_cluster_exact<int, double>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("num_clusters"), py::arg("d").noconvert(), py::arg("cm").noconvert(), py::arg("c").noconvert(),
R"pbdoc(
Perform one iteration of Lloyd clustering on a distance graph using
exact centers

Parameters
----------
num_nodes : int
    number of rows in A (number of vertices)
Ap : array
    CSR row pointer
Aj : array
    CSR index array
Ax : array
    CSR data array (edge lengths)
num_clusters : int
    number of clusters = number of seeds
d : array, num_nodes
    distance to nearest seed
cm : array, num_nodes
    cluster index for each node
c : array, num_clusters
    cluster centers

Notes
-----
This version computes exact cluster centers with Floyd-Warshall and
also uses a balanced version of Bellman-Ford to try and find
nearly-equal-sized clusters.)pbdoc");

    m.def("maximal_independent_set_k_parallel", &_maximal_independent_set_k_parallel<int, int, double>,
        py::arg("num_rows"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("k"), py::arg("x").noconvert(), py::arg("y").noconvert(), py::arg("max_iters"),
R"pbdoc(
Compute MIS-k.

Parameters
----------
num_rows : int
    number of rows in A (number of vertices)
Ap : array
    CSR row pointer
Aj : array
    CSR index array
k : int
    minimum separation between MIS vertices
x : array, inplace
    state of each vertex (1 if in the MIS, 0 otherwise)
y : array
    random values used during parallel MIS algorithm
max_iters : int
    maximum number of iterations to use (default, no limit)

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
Compute a breadth first search of a graph in CSR format
beginning at a given seed vertex.

Parameters
----------
num_rows : int
    number of rows in A (number of vertices)
Ap : array
    CSR row pointer
Aj : array
    CSR index array
order : array, num_rows, inplace
    records the order in which vertices were searched
level : array, num_rows, inplace
    records the level set of the searched vertices (i.e. the minimum distance to the seed)

Notes
-----
The values of the level must be initialized to -1)pbdoc");

    m.def("connected_components", &_connected_components<int>,
        py::arg("num_nodes"), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("components").noconvert(),
R"pbdoc(
Compute the connected components of a graph stored in CSR format.

Parameters
----------
num_rows : int
    number of rows in A (number of vertices)
Ap : array
    CSR row pointer
Aj : array
    CSR index array
components : array, num_rows
    component labels

Notes
-----
Vertices belonging to each component are marked with a unique integer
in the range [0,K), where K is the number of components.)pbdoc");

}

