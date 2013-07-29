Random notes for development

Release steps:
* python setup.py install in the master
* pyamg.test() passes
* python setup.py sdist in the master
* check that the tarball installs and passes
* change ISRELEASED to True in setup.py
* first github release:
* git tag
* git push
* now pypi:
* python setup.py register
* python setup.py sdist upload
