#!/usr/bin/env python3
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.mathjax',
              'm2r',
              'numpydoc']

autodoc_default_flags = ['members', 'undoc-members', 'special-members', 'private-members']
autosummary_generate = True

templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'

project = 'PyAMG'
copyright = '2018, Luke Olson'
author = '2018, Luke Olson and Jacob Schroder'

import pyamg
version = pyamg.__version__
release = version

language = None
exclude_patterns = ['README.md']

pygments_style = 'sphinx'

todo_include_todos = False

# for debugging
# keep_warnings=True

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_logo = '../logo/versions/logo_dropshadow_small.png'
html_static_path = ['_static']

htmlhelp_basename = 'PyAMGdoc'

latex_elements = {
    # 'papersize': 'letterpaper',
    # 'pointsize': '10pt',
    # 'preamble': '',
    # 'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'PyAMG.tex', 'PyAMG Documentation',
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

def autodoc_skip_member(app, what, name, obj, skip, options):
    exclusions = ('__weakref__',  # special-members
                  '__doc__', '__module__', '__dict__',  # undoc-members
                  )
    exclude = name in exclusions
    return skip or exclude

def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member)
