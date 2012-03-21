#!/usr/bin/env python
"""
usage:
python setup.py build_ext --inplace
"""
import commands
flag = commands.getstatusoutput('swig -c++ -python splinalg.i')
if flag[0]!=0:
    print flag
    exit()

from numpy.distutils.core import setup, Extension
splinalg_module = Extension('_splinalg', sources=['splinalg_wrap.cxx'], define_macros=[('__STDC_FORMAT_MACROS', 1)],)
setup (name = 'splinalg',
       version = '0.1',
       author      = "Luke Olson",
       description = """basic sparse linear algebra""",
       ext_modules = [splinalg_module],
       py_modules = ["splinalg"],
       )

#   def configuration(parent_package='',top_path=None):
#       from numpy.distutils.misc_util import Configuration
#   
#       config = Configuration()
#   
#       config.add_extension('_splinalg',
#               define_macros=[('__STDC_FORMAT_MACROS', 1)],
#               sources=['splinalg_wrap.cxx'])
#   
#       return config
#   
#   if __name__ == '__main__':
#       from numpy.distutils.core import setup
#       setup(**configuration(top_path='').todict())
