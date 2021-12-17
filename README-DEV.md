Release Steps:
- with no staged commits and a clean status...
- add notes to `CHANGELOG.md`
- `mkvirtualenv releasetest`
- check that `pip install .` and `python -c "import pyamg; pyamg.test()"` pass (outside source directory)
- remove untracked files `git clean -xdf`
- `git tag -a v3.2.0 -m "version 3.2.0"`
- `git push`
- `git push --tags`
- then release the version on Github: `gh release create v1.2.3 --notes "bugfix release, see CHANGELOG.md"`
  - This will trigger the GHA `.github/workflows/wheels.yml` which builds wheels and a source distribution, and publishes to pypi

Testing notes:
- do not use seeds such as 0, 1, 42, 100
- for each needed seed, generate a "random" int
- `python3 -c "import numpy as np; np.random.seed(); seeds = np.random.randint(0, 2**32 - 1, 5); print(seeds)"`
- check memory leaks with `scripts/checkvalgrind.sh`
- lint with `flake8 pyamg`
- `pip install -e .` then lint `pylint --rcfile setup.cfg pyamg`
