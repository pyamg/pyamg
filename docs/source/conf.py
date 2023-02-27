#!/usr/bin/env python3
"""Sphinx configuration."""
import os
import sys
import pyamg
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.mathjax',
              'sphinx_automodapi.automodapi',
              'm2r2',
              'numpydoc']

numpydoc_show_inherited_class_members = False

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'special-members': True,
    'private-members': True,
}
autosummary_generate = True

templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'

project = 'PyAMG'
copyright = '2022, Luke Olson'  # pylint: disable=redefined-builtin
author = '2022, Luke Olson and Jacob Schroder'

version = pyamg.__version__
release = version

language = 'en'
exclude_patterns = ['README.md']

pygments_style = 'sphinx'

todo_include_todos = False

# for debugging
# keep_warnings=True

html_theme = 'pydata_sphinx_theme'
html_logo = '../logo/pyamg_logo.png'
html_static_path = ['_static']

html_theme_options = {
    'github_url': 'https://github.com/pyamg/pyamg',
    'logo': {'image_light': 'pyamg_logo.png',
             'image_dark': 'pyamg_logo.png',
             }
}

htmlhelp_basename = 'PyAMGdoc'

latex_elements = {
    # 'papersize': 'letterpaper',
    # 'pointsize': '10pt',
    # 'preamble': '',
    # 'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'pyamg.tex', 'PyAMG Documentation',
     'Luke Olson', 'manual'),
]

man_pages = [
    (master_doc, 'pyamg', 'PyAMG Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'PyAMG', 'PyAMG Documentation',
     author, 'PyAMG', 'Algebraic Multigrid Solvers in Python.',
     'Miscellaneous'),
]


def autodoc_skip_member(_app, _what, name, _obj, skip, _options):
    """Set skip member."""
    exclusions = ('__weakref__',  # special-members
                  '__doc__', '__module__', '__dict__',  # undoc-members
                  )
    exclude = name in exclusions
    return skip or exclude


def setup(app):
    """Define setup."""
    app.connect('autodoc-skip-member', autodoc_skip_member)
