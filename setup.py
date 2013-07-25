#!/usr/bin/env python
"""PyAMG: Algebraic Multigrid Solvers in Python

PyAMG is a library of Algebraic Multigrid (AMG) solvers 
with a convenient Python interface.

"""

import os
import sys

DOCLINES = __doc__.split('\n')

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Environment :: Console
Environment :: X11 Applications
Intended Audience :: Developers
Intended Audience :: Education
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Natural Language :: English
Operating System :: OS Independent
Programming Language :: C++
Programming Language :: Python
Topic :: Education
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Mathematics
Topic :: Software Development :: Libraries :: Python Modules
"""

NAME = 'pyamg'
AUTHOR = 'Nathan Bell, Luke OLson, and Jacob Schroder'
AUTHOR_EMAIL = 'luke.olson@gmail.com'
MAINTAINER = 'Luke Olson'
MAINTAINER_EMAIL = 'luke.olson@gmail.com'
URL = 'https://github.com/pyamg/pyamg'
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = '\n'.join(DOCLINES[2:])
DOWNLOAD_URL = 'https://github.com/pyamg/pyamg/releases'
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix']
LICENSE = 'BSD'

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('pyamg')
    config.add_data_files(('pyamg','*.txt'))

    config.get_version(os.path.join('pyamg','version.py')) # sets config.version

    return config

def setup_package():
    from numpy.distutils.core import setup

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)
    sys.path.insert(0,os.path.join(local_path,'pyamg')) # to retrive version

    try:
        setup(
            name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DOCLINES[0],
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            platforms=PLATFORMS,
            configuration=configuration )
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return

if __name__ == '__main__':
    setup_package()
