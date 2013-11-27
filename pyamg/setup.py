#!/usr/bin/env python


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('pyamg', parent_package, top_path)

    config.add_subpackage('aggregation')
    config.add_subpackage('classical')
    config.add_subpackage('gallery')
    config.add_subpackage('krylov')
    config.add_subpackage('amg_core')
    config.add_subpackage('relaxation')
    config.add_subpackage('testing')
    config.add_subpackage('util')
    config.add_subpackage('vis')

    config.add_data_dir('tests')

    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
