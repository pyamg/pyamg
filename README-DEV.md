Random notes for development

Release steps:
* python setup.py install in the master
* pyamg.test() passes
* python setup.py sdist in the master
* check that the tarball installs and passes
* change ISRELEASED to True in setup.py
* change version in setup.py
* first github release:
    * git tag -a v2.1.0 -m "version 2.1.0"
    * git push origin v2.1.0
* now pypi:
    * python setup.py register
    * python setup.py sdist upload
* change ISRELEASED back to FALSE
