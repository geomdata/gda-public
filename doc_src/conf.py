# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

import sys, os, re


needs_sphinx = '1.0'

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.



extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'matplotlib.sphinxext.plot_directive']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

master_doc = 'index'

# General information about the project.
project = 'GDA Toolbox'
copyright = '2015-2017, Geometric Data Analytics, Inc'
author = 'Geometric Data Analytics, Inc'
# The short X.Y version.
version = '1.0'
# The full version, including alpha/beta/rc tags.
release = '1.0'


# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
unused_docs = []
default_role = "autolink"
exclude_dirs = []
add_function_parentheses = True
add_module_names = True
show_authors = True
pygments_style = 'sphinx'


# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

themedir = '_theme'
if not os.path.isdir(themedir):
    raise RuntimeError("Get the scipy-sphinx-theme first, "
                       "and copy the _theme directory to doc_src")

html_theme = 'scipy'
html_theme_path = [themedir]
html_theme_options = {
        "edit_link": False,
        "sidebar": "right",
        "scipy_org_logo": False,
        "rootlinks": []
    }

html_title = "%s v%s Manual" % (project, version)
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'

html_use_modindex = True
html_copy_source = False
html_domain_indices = True
html_file_suffix = '.html'
intersphinx_mapping = {
    'python': ('https://docs.python.org/dev', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'pandas': ('http://pandas-docs.github.io/pandas-docs-travis/', None),
    'matplotlib': ('http://matplotlib.org', None),
    'cython': ('http://docs.cython.org', None)
}

import glob
autosummary_generate = glob.glob("*.rst")

# -----------------------------------------------------------------------------
# Coverage checker
# -----------------------------------------------------------------------------
coverage_ignore_modules = r"""
    """.split()
coverage_ignore_functions = r"""
    test($|_) (some|all)true bitwise_not cumproduct pkgload
    generic\.
    """.split()
coverage_ignore_classes = r"""
    """.split()

coverage_c_path = []
coverage_c_regexes = {}
coverage_ignore_c_items = {}

todo_include_todos=True

numpydoc_show_class_members = False
