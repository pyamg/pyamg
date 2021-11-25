Release Steps:
- with no staged commits and a clean status...
- add notes to `CHANGELOG.md`
- `mkvirtualenv releasetest`
- check that `pip install .` and `python -c "import pyamg; pyamg.test()"` pass (outside source directory)
- remove untracked files `git ls-files . --ignored --exclude-standard --others`
- `git tag -a v3.2.0 -m "version 3.2.0"`
- `git push`
- `git push --tags`
- first github:
  - create source distribution: `python3 setup.py sdist`
  - `gh release create v1.2.3 --notes "bugfix release, see CHANGELOG.md"`
  - `gh release upload v1.2.3 ./dist/*.tgz`
- now pypi:
    - `git clean -xdf`
    - `twine upload --skip-existing dist/*` (no register needed)

Testing notes:
- do not use seeds such as 0, 1, 42, 100
- for each needed seed, generate a "random" int
- `python3 -c "import numpy as np; np.random.seed(); seeds = np.random.randint(0, 2**32 - 1, 5); print(seeds)"`
