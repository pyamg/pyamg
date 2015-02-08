#!/usr/bin/env python


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    # lapack_opt = get_info('lapack_opt')
    config = Configuration('amg_core', parent_package, top_path)

    # extra_info = lapack_opt)
    config.add_extension('_amg_core',
                         define_macros=[('__STDC_FORMAT_MACROS', 1)],
                         sources=['amg_core_wrap.cxx'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
