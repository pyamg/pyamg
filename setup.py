#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyAMG: Algebraic Multigrid Solvers in Python

PyAMG is a library of Algebraic Multigrid (AMG)
solvers with a convenient Python interface.

PyAMG features implementations of:

- Ruge-Stuben (RS) or Classical AMG
- AMG based on Smoothed Aggregation (SA)
- Adaptive Smoothed Aggregation (αSA)
- Compatible Relaxation (CR)
- Krylov methods such as CG, GMRES, FGMRES, BiCGStab, MINRES, etc

PyAMG is primarily written in Python with
supporting C++ code for performance critical operations.
"""

import os
import subprocess

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

version = '4.1.0'
isreleased = False

install_requires = (
    'numpy>=1.7.0',
    'scipy>=0.12.0',
    'pytest>=2',
)


# set the version information
# https://github.com/numpy/numpy/commits/master/setup.py
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
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
        out = _minimal_ext_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        GIT_BRANCH = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = 'Unknown'
        GIT_BRANCH = ''

    return GIT_REVISION


def set_version_info(VERSION, ISRELEASED):
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('pyamg/version.py'):
        try:
            import imp
            version = imp.load_source("pyamg.version", "pyamg/version.py")
            GIT_REVISION = version.git_revision
        except ImportError:
            raise ImportError('Unable to read version information.')
    else:
        GIT_REVISION = 'Unknown'
        GIT_BRANCH = ''

    FULLVERSION = VERSION
    if not ISRELEASED:
        FULLVERSION += '.dev0' + '+' + GIT_REVISION[:7]

    print(GIT_REVISION)
    print(FULLVERSION)
    return FULLVERSION, GIT_REVISION


def write_version_py(VERSION,
                     FULLVERSION,
                     GIT_REVISION,
                     ISRELEASED,
                     filename='pyamg/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


fullversion, git_revision = set_version_info(version, isreleased)
write_version_py(version, fullversion, git_revision, isreleased,
                 filename='pyamg/version.py')

# add module for each c++ file
amg_core_headers = ['evolution_strength',
                    'graph',
                    'krylov',
                    'linalg',
                    'relaxation',
                    'ruge_stuben',
                    'smoothed_aggregation']
ext_modules = [
    Pybind11Extension(
        f"pyamg.amg_core.{f}",
        sources=[f"pyamg/amg_core/{f}_bind.cpp"]
    )
    for f in amg_core_headers
]
ext_modules += [
    Pybind11Extension(
        "pyamg.amg_core.tests.bind_examples",
        sources=[f"pyamg/amg_core/tests/bind_examples_bind.cpp"]
    )
]

setup(
    name='pyamg',
    version=fullversion,
    keywords=['algebraic multigrid AMG sparse matrix preconditioning'],
    author='Nathan Bell, Luke Olson, and Jacob Schroder',
    author_email='luke.olson@gmail.com',
    maintainer='Luke Olson',
    maintainer_email='luke.olson@gmail.com',
    url='https://github.com/pyamg/pyamg',
    download_url='https://github.com/pyamg/pyamg/releases',
    license='MIT',
    platforms=['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    description=__doc__.split('\n')[0],
    long_description=__doc__,
    #
    packages=find_packages(exclude=['doc']),
    package_data={'pyamg': ['gallery/example_data/*.mat', 'gallery/mesh_data/*.npz']},
    include_package_data=False,
    install_requires=install_requires,
    zip_safe=False,
    #
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    setup_requires=['numpy', 'pybind11'],
    #
    tests_require=['pytest'],
    #
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Environment :: X11 Applications',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
