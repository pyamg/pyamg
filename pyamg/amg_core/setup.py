#!/usr/bin/env python

def configuration(parent_package='',top_path=None):
    #from numpy.distutils.system_info import get_info, NotFoundError
    from numpy.distutils.misc_util import Configuration

    #lapack_opt = get_info('lapack_opt')
    config = Configuration('amg_core', parent_package, top_path)

    config.add_extension('_amg_core', 
            sources=['amg_core_wrap.cxx'])
            #extra_info = lapack_opt)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

