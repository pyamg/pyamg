Random notes for development

Release steps:
* python setup.py install in the master
* pyamg.test() passes
* change ISRELEASED to True in setup.py
* change version in setup.py
* python setup.py sdist in the master
* check that the tarball version, that it installs and passes
* first github release:
    * git tag -a v2.1.0 -m "version 2.1.0"
    * git push --tags
    * on github under release: draft new release (with the new tag)
* now pypi:
    * python setup.py register
    * python setup.py sdist upload
* change ISRELEASED back to FALSE
* add new (latest release) to github: https://github.com/blog/1547-release-your-software
