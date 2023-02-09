Release Steps:
- suppose the current tag is 4.2.2 and the next is 4.2.3
- with no staged commits and a clean status...
- meld a summary of `git log $(git tag --sort version:refname | tail -n 1)..HEAD --oneline` (all commits since last tag) with whatever hash with `[4.2.3]` in changelog
- commit, push
- `mkvirtualenv releasetest`
- check that `pip install .` and `python -c "import pyamg; pyamg.test()"` pass (outside source directory)
- remove untracked files `git clean -xdf`
- the following can be done with a pre-release, `v4.2.3-alpha.6`, for testing.  It will not become the default on pypi and `gh release create` can be marked with `--prerelease` (below)
- mark `fallback_version` in `pyproject.toml`
- commit, push
- `git tag -a v4.2.3 -m "version 4.2.3"`
- `git push`
- `git push --tags`
- then release the version on Github: `gh release create v4.2.3 --notes "see changelog.md"`
  - This will trigger the GHA `.github/workflows/wheels.yml` which builds wheels and a source distribution, and publishes to pypi

Testing notes:
- do not use seeds such as 0, 1, 42, 100
- for each needed seed, generate a "random" int
- `python3 -c "import numpy as np; np.random.seed(); seeds = np.random.randint(0, 2**32 - 1, 5); print(seeds)"`
