#!/usr/bin/env python
"""PyAMG: Algebraic Multigrid Solvers in Python

PyAMG is a library of Algebraic Multigrid (AMG) solvers 
with a convenient Python interface.

"""

DOCLINES = __doc__.split("\n")

import os
import sys

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: Education
License :: OSI Approved :: BSD License
Programming Language :: C++
Programming Language :: Python
Topic :: Education
Topic :: Software Development
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Mathematics
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

#os.environ['NO_PYAMG_IMPORT']='SciPy/setup.py'

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
    from numpy.distutils.misc_util import Configuration

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)
    sys.path.insert(0,os.path.join(local_path,'pyamg')) # to retrive version

    try:
        setup(
            name = 'pyamg',
            maintainer = "PyAMG Developers",
            maintainer_email = "wnbell@gmail.com",
            description = DOCLINES[0],
            long_description = "\n".join(DOCLINES[2:]),
            url = "http://code.google.com/p/pyamg",
            download_url = "http://code.google.com/p/pyamg/downloads/list",
            license = 'BSD',
            classifiers=filter(None, CLASSIFIERS.split('\n')),
            platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
            configuration=configuration )
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return

if __name__ == '__main__':
    setup_package()
