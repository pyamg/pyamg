The .cpp files in this directory are generated using the script
`generate_bindings.py`.  Pybind11 is used to for the Python bindings.

0. Run
    ./generate_bindings.py SOMEFILE.h
to generate `SOMEFILE_bind.cpp`.  See `generate_bindings.py` for more details.

1. Setup will builld each `*_bind.cpp` file.

2. Import everything in `__init__.py`
