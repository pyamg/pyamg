#!/usr/bin/env python
"""PyAMG: Algebraic Multigrid Solvers in Python

PyAMG is a library of Algebraic Multigrid (AMG) solvers
with a convenient Python interface.
"""

import os
import sys
import subprocess

#if sys.version_info[:2] < (2, 6) or (3, 0) <= sys.version_info[0:2]:
#    raise RuntimeError("Python version 2.6, 2.7 required.")

if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins

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
Programming Language :: Python :: 2
Programming Language :: Python :: 2.6
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.3
Programming Language :: Python :: 3.4
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
DOWNLOAD_URL = 'https://github.com/pyamg/pyamg/releases'
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = '\n'.join(DOCLINES[2:])
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix']
LICENSE = 'BSD'
MAJOR = 3
MINOR = 0
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
INSTALL_REQUIRES = ['nose', 'numpy', 'scipy']

# Use NumPy/SciPy style versioning
#
# This changes the pyamg/version.py file directly
#
#       notes for the future:
#       should we try to use a tag if it exists? probably not
#       could also add -devN:C:D:R to the versioning
#       N: revisions numbers approximated with git rev-list HEAD --count
#       C: unique SHA1 generated with git describe --always
#       D: date in format YYYYMMDDHHMMSS
#       R: which repository (fork) this came from with git remote show origin
#       then add this to say fullversion.py

# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'describe', '--always'])
        GIT_REVISION = out.decode('utf-8', 'strict').strip()
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

# This is a bit hackish: we are setting a global variable so that the main
# pyamg __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
builtins.__PYAMG_SETUP__ = True


def write_version_py(filename='pyamg/version.py'):
    cnt = """
# this file is generated from pyamg in setup.py
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of pyamg.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('pyamg/version.py'):
        # must be a source distribution, use existing version file
        try:
            from pyamg.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "pyamg/version.py and the build directory "
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('pyamg')

    config.get_version('pyamg/version.py')  # sets config.version

    return config


def setup_package():

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)

    # rewrite the version file
    write_version_py()

    # Run build
    try:
        from numpy.distutils.core import setup
    except ImportError:
        from setuptools import setup

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
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            configuration=configuration)
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return

if __name__ == '__main__':
    setup_package()
