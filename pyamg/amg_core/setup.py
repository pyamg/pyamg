#!/usr/bin/env python

if __name__ == '__main__':
    from setuptools import setup, Extension
    import numpy as np

    setup(ext_modules=[Extension('amg_core/amg_core',
                                 sources=['amg_core_wrap.cxx'],
                                 define_macros=[('__STDC_FORMAT_MACROS', 1)],
                                 include_dirs=[np.get_include()])],
          name='amg_core',
          package_dir={'amg_core': '.'},
          packages=['amg_core'])
