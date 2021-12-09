Requirements:
    - sphinx

The files in pyamg/Docs such as

    - `pyamg/Docs/source/conf.py`
    - `pyamg/Docs/source/index.rst`
    - `pyamg/Docs/Makefile`

are created with sphinx-quickstart and are in the repo.  The files
`pyamg/Docs/source/pyamg.*.rst` are generated with

```
sphinx-apidoc -f -o source ../pyamg
```
