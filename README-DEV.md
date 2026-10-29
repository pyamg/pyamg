Release Steps:
- suppose the current tag is 4.2.2 and the next is 4.2.3
- with no staged commits and a clean status...
- meld a summary of `git log $(git describe --tags --abbrev=0)..HEAD --oneline` (all commits since last tag) with whatever hash with `[4.2.3]` in changelog
- commit, push
- wait for CI to pass on `main`: `ci.yml` (including `Test-min-deps`), `lint.yml`, `docstrings.yml`
  (`gh run list --branch main`)
- make a virtual environment and activate
- check that `pip install .` and `python -c "import pyamg; pyamg.test()"` pass (outside source directory)
- remove untracked files: preview with `git clean -xdn`, then `git clean -xdf`
- build the wheels *before* tagging (see "Building wheels" below):
  `gh workflow run wheels.yml --ref main`, then check the run and fix until it is clean
- the following can be done with a pre-release, `v4.2.3-alpha.6`, for testing.  It will not become the default on pypi and `gh release create` can be marked with `--prerelease` (below)
  - a pre-release still uploads to the real pypi (as `4.2.3a6`)
- if the wheel build needed fixes: commit, push (and wait for CI again)
- `git tag -a v4.2.3 -m "version 4.2.3"`
- `git push`
- `git push --tags`
- then release the version on Github: `gh release create v4.2.3 --notes "see changelog.md"`
  - This will trigger the GHA `.github/workflows/wheels.yml` which builds wheels and a source distribution, and publishes to pypi

Building wheels (`.github/workflows/wheels.yml`):
- runs only on a published GitHub release or a manual run; never on push
- a release builds the release tag and uploads to pypi
- a manual run only builds; wheels and sdist are saved as run artifacts, nothing is uploaded
  - `gh workflow run wheels.yml --ref main` builds the tip of `main`
  - `gh workflow run wheels.yml --ref main -f ref=<tag-or-commit>` builds a specific tag or commit
  - `gh run list --workflow wheels.yml` to watch, `gh run download <run-id>` to fetch the artifacts
  - an untagged commit gets a dev version from versioningit (e.g. `5.3.1.dev3+g1234abc`)
-  fix wheel problems with manual runs *before* tagging, then tag and release once

If the release is already tagged and something minor needs fixing:
- pypi files can never be replaced: once `pyamg-5.3.1-...whl` is uploaded, that filename is taken for good (even if deleted)
- if nothing reached pypi (`upload_all` is skipped when any build fails), move the tag:
  - `git tag -f -a v5.3.1 -m "version 5.3.1" <commit>` then `git push --force origin v5.3.1`
  - moving a tag does not re-trigger the release; delete and re-create the GitHub release
    (`gh release delete v5.3.1` then `gh release create ...`) to build and upload
- if files are already on pypi, do not move the tag; release `v5.3.2` (or `v5.3.1.post1`)
- a release run uses `wheels.yml` from the *tagged commit*, so edits to it on `main` do not
  affect re-runs of that release; to add only *missing* wheels to an existing release, the
  tagged commit's `wheels.yml` needs `skip-existing: true`

Testing notes:
- do not use seeds such as 0, 1, 42, 100
- for each needed seed, generate a "random" int
- `python3 -c "import numpy as np; np.random.seed(); seeds = np.random.randint(0, 2**32 - 1, 5); print(seeds)"`
