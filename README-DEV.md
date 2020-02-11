Release Steps:
- with no staged commits...
- check that `python setup.py test` passes
- remove build, dist, .egg-info, etc
- change `isreleased` to True in `setup.py`
- change `version` in `setup.py`
- git commit -a -m "version 3.2.0"
- first github release:
    - git tag -a v3.2.0 -m "version 3.2.0"
    - git push --tags
    - on github under release: draft new release (with the new tag): https://github.com/blog/1547-release-your-software
    - release title: v3.2.0
    - add summary of changes to the notes
- now pypi:
    - `git clean -xdf`
    - `python2 setup.py sdist bdist_wheel`
    - `python3 setup.py sdist bdist_wheel`
    - `twine upload --skip-existing dist/*` (no register needed)
- change `isreleased` to False in `setup.py`
- git commit -a -m "remove isreleased"

Testing notes:
- do not use seeds such as 0, 1, 42, 100
- for each needed seed, generate a "random" int
- `python3 -c "import numpy as np; np.random.seed(); seeds = np.random.randint(0, 2**32 - 1, 5); print(seeds)"`
