#!/usr/bin/env python
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
import sys

import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand

install_requires = (
    'numpy>=1.7.0',
    'scipy>=0.12.0',
    'pytest>=2',
)


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        pytest.main(self.test_args)


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        l_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        ct = self.compiler.compiler_type
        c_opts = self.c_opts.get(ct, [])
        l_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            c_opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                c_opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.extra_compile_args = c_opts
            ext.extra_link_args = l_opts
            ext.define_macros = [('VERSION_INFO', f'"{self.distribution.get_version()}"')]

        build_ext.build_extensions(self)

    # identify extension modules
    # since numpy is needed (for the path), need to bootstrap the setup
    # http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    def finalize_options(self):
        build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


class get_pybind_include:
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        # The issue:
        # https://github.com/pybind/pybind11/issues/1067
        #
        # pybind11 will install files to
        # TMP/pybind11-version.egg/*.h
        # TMP/pybind11-version.egg/detail/*.h
        #
        # We need this to look like
        # TMP/pybind11/*.h
        # TMP/pybind11/detail/*.h
        # TMPDIR/pybind11-2.2.4-py3.7.egg/pybind11/__init__.py
        f = pybind11.__file__

        # TMPDIR/pybind11-2.2.4-py3.7.egg/pybind11/
        d = os.path.dirname(f)

        # TMPDIR/pybind11-2.2.4-py3.7.egg
        dd = os.path.dirname(d)

        # TMPDIR
        tmpdir = os.path.dirname(dd)

        # check if not a half-install
        if not os.path.exists(os.path.join(dd, 'pybind11.h')):
            return pybind11.get_include(self.user)

        # if it *is* a half-install
        # Then copy all files to
        # TMPDIR/pybind11
        if not os.path.isdir(os.path.join(tmpdir, 'pybind11')):
            import shutil
            shutil.copytree(dd, os.path.join(tmpdir, 'pybind11'))

        return tmpdir


amg_core_headers = ['evolution_strength.h',
                    'graph.h',
                    'krylov.h',
                    'linalg.h',
                    'relaxation.h',
                    'ruge_stuben.h',
                    'smoothed_aggregation.h']
amg_core_headers = [f.replace('.h', '') for f in amg_core_headers]

ext_modules = [Extension('pyamg.amg_core.%s' % f,
                         sources=['pyamg/amg_core/%s_bind.cpp' % f],
                         include_dirs=[get_pybind_include(), get_pybind_include(user=True)],
                         undef_macros=['NDEBUG'],
                         language='c++') for f in amg_core_headers]

ext_modules += [Extension('pyamg.amg_core.tests.bind_examples',
                          sources=['pyamg/amg_core/tests/bind_examples_bind.cpp'],
                          include_dirs=[get_pybind_include(), get_pybind_include(user=True)],
                          language='c++')]

setup(
    name='pyamg',
    use_scm_version={
        "version_scheme": lambda _: "4.1.0",
        "write_to": "pyamg/version.py"
    },
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
    cmdclass={'build_ext': BuildExt, 'test': PyTest},
    setup_requires=['numpy', 'pybind11', 'setuptools_scm'],
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
