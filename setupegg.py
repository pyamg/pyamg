#!/usr/bin/env python
"""
A setup.py script to use setuptools
"""

from setuptools import setup

if sys.version_info[0] >= 3:
    import imp
    setupfile = imp.load_source('setupfile', 'setup.py')
    setupfile.setup_package()
else:
    exec(compile(open('setup.py').read(), 'setup.py', 'exec'))
