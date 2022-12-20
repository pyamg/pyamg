The `_bind.cpp` files in this directory are generated using the script
`generate_bindings.py`.  Pybind11 is used to for the Python bindings.

0. Include C/C++ functions to be bound to Python in `instantiate.yml`, add any `.h` files to `generate.sh`.

1. Run `./generate.sh` to generate `SOMEFILE_bind.cpp` files for every `SOMEFILE.h` listed in `generate.sh`. Alternatively, call `python bindthem.py SOMEFILE.h` to bind a specific file. 

2. Import everything in `__init__.py`
