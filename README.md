![](Docs/logo/PyAMG_logo.png)

PyAMG is developed by **[Nathan Bell](http://graphics.cs.uiuc.edu/~wnbell/)**, **[Luke Olson](http://www.cs.uiuc.edu/homes/lukeo/)**, and **[Jacob Schroder](http://grandmaster.colorado.edu/~jacob/index.html)**, in the **[Deparment of Computer Science](http://www.cs.uiuc.edu)** at the **[University of Illinois at Urbana-Champaign](http://www.illinois.edu)**.  Portions of the project were partially supported by the [NSF](http://www.nsf.gov) under award DMS-0612448.

![](Docs/logo/CS_logo.png)

----

[![Build Status](https://travis-ci.org/pyamg/pyamg.png?branch=master)](https://travis-ci.org/pyamg/pyamg)

# Introduction

PyAMG is a library of **Algebraic Multigrid (AMG)** solvers with a convenient Python interface.  

# Citing

<pre>
@MISC{BeOlSc2011,
      author = "Bell, W. N. and Olson, L. N. and Schroder, J. B.",
      title = "{PyAMG}: Algebraic Multigrid Solvers in {Python} v2.0",
      year = "2011",
      url = "http://www.pyamg.org",
      note = "Release 2.0"
      }
</pre>

# Getting Help

Contact the [pyamg-user group](http://groups.google.com/group/pyamg-user)

Look at the [Tutorial](https://github.com/pyamg/pyamg/wiki/Tutorial) or the [Examples](https://github.com/pyamg/pyamg/wiki/Examples) (for instance  the [0STARTHERE](https://github.com/pyamg/pyamg-examples/blob/master/0STARTHERE/demo.py) example)

# What is AMG?

 AMG is a multilevel technique for solving large-scale linear systems with optimal or near-optimal efficiency.  Unlike geometric multigrid, AMG requires little or no geometric information about the underlying problem and develops a sequence of coarser grids directly from the input matrix.  This feature is especially important for problems discretized on unstructured meshes and irregular grids.

# PyAMG Features

PyAMG features implementations of:

- **Ruge-Stuben (RS)** or *Classical AMG*
- AMG based on **Smoothed Aggregation (SA)**

and experimental support for:

- **Adaptive Smoothed Aggregation (αSA)**
- **Compatible Relaxation (CR)**

The predominant portion of PyAMG is written in Python with a smaller amount of supporting C++ code for performance critical operations.

# PyAMG Objectives

**ease of use**

- interface is accessible to non-experts
- extensive documentation and references

**speed**

- solves problems with millions of unknowns efficiently
- core multigrid algorithms are implemented in C++ and translated through SWIG
- sparse matrix support provided by scipy.sparse

**readability**

- source code is organized into intuitive components

**extensibility**

- core components can be reused to implement additional techniques
- new features are easy integrated

**experimentation**

- facilitates rapid prototyping and analysis of multigrid methods

**portability**

- tested on several platforms
- relies only on Python, NumPy, and SciPy

# Example Usage

PyAMG is easy to use!  The following code constructs a two-dimensional Poisson problem and solves the resulting linear system with Classical AMG.

````python
from scipy import *
from scipy.linalg import *
from pyamg import *
from pyamg.gallery import *
A = poisson((500,500), format='csr')     # 2D Poisson problem on 500x500 grid
ml = ruge_stuben_solver(A)               # construct the multigrid hierarchy
print ml                                 # print hierarchy information
b = rand(A.shape[0])                     # pick a random right hand side
x = ml.solve(b, tol=1e-10)               # solve Ax=b to a tolerance of 1e-8
print "residual norm is", norm(b - A*x)  # compute norm of residual vector
````

Program output:

<pre>
    multilevel_solver
    Number of Levels:     6
    Operator Complexity:  2.198
    Grid Complexity:      1.666
    Coarse Solver:        'pinv2'
      level   unknowns     nonzeros
        0       250000      1248000 [45.50%]
        1       125000      1121002 [40.87%]
        2        31252       280662 [10.23%]
        3         7825        70657 [ 2.58%]
        4         1937        17973 [ 0.66%]
        5          484         4728 [ 0.17%]
    
    residual norm is 1.86112114946e-06
</pre>
