Requirements:
    - sphinx

The files in pyamg/Docs such as

    - `pyamg/docs/source/conf.py`
    - `pyamg/docs/source/index.rst`
    - `pyamg/docs/Makefile`

are created with sphinx-quickstart and are in the repo.  The files
`pyamg/docs/source/pyamg.*.rst` are generated with

```
sphinx-apidoc -f -o source ../pyamg "../**/test*.py"
```
