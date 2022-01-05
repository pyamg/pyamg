Steps.

First get the `pybind11` suppressions:
- `curl -s https://raw.githubusercontent.com/python/cpython/main/Misc/valgrind-python.supp > valgrind-python.supp'
- `curl -s https://raw.githubusercontent.com/pybind/pybind11/master/tests/valgrind-numpy-scipy.supp > valgrind-numpy-scipy.supp'
- remove the `###` lines from `valgrind-python.supp`

Then generate any extra suppressions from numpy and others:
- `PYTHONMALLOC=malloc valgrind --leak-check=full --log-file="valgrind-numpy-extras.txt" --show-leak-kinds=definite,indirect --gen-suppressions=all --suppressions="valgrind-numpy-scipy.supp" python -c "import numpy"`
- dump the suppressions into `valgrind-extras.supp`

Then run with `valgrind_run.sh`
