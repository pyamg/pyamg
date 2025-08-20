project('pyamg', 'cpp',
  version: run_command(find_program('python3'), '-m', 'versioningit', check: true).stdout().strip(),
  license: 'MIT',
  default_options: ['cpp_std=c++11'],
)

amg_core_headers = ['air',
                    'evolution_strength',
                    'graph',
                    'krylov',
                    'linalg',
                    'relaxation',
                    'ruge_stuben',
                    'smoothed_aggregation']

py = import('python').find_installation(pure: false)
pybind11_dep = dependency('pybind11')

foreach header : amg_core_headers
  py.extension_module(header,
      'pyamg/amg_core/' + header + '_bind.cpp',
      install: true,
      subdir: 'pyamg/amg_core',
      dependencies : [pybind11_dep],
  )
endforeach

py.extension_module('bind_examples',
    'pyamg/amg_core/tests/bind_examples_bind.cpp',
    install: true,
    subdir: 'pyamg/amg_core/tests',
    dependencies : [pybind11_dep],
)

install_subdir('pyamg',
  exclude_directories: 'amg_core',   # subdirectory of pyamg
  install_dir: py.get_install_dir(),
  )
py.install_sources([
  'pyamg/amg_core/__init__.py',
  'pyamg/amg_core/tests/__init__.py',
  'pyamg/amg_core/tests/test_bind_examples.py'
  ],
  preserve_path: true,
)
