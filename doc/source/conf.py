# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
# import guzzle_sphinx_theme
# import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

import pytorch_sphinx_theme

project = "torchlip"
copyright = (
    "2020, IRT Antoine de Saint Exupéry"
    " - All rights reserved. DEEL is a research program operated by IVADO, "
    "IRT Saint Exupéry, CRIAQ and ANITI."
)
author = (
    "Mathieu Serrurier (mathieu.serrurier@irt-saintexupery.com),\n"
    "Franck Mamalet (franck.mamalet@irt-saintexupery.com),\n"
    "Thibaut Boissin (thibaut.boissin@irt-saintexupery.com),\n"
    "Mikael Capelle (mikael.capelle@irt-saintexupery.com),\n"
    "Justin Plakoo (justin.plakoo@irt-saintexupery.com),"
)

# The full version, including alpha/beta/rc tags
release = "1.0.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
    "nbsphinx",
]

autoclass_content = "both"
autoapi_root = "../deel"

nbsphinx_requirejs_path = ""

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "build\\*",
    "..\\deel\\torchlip\\examples\\*",
    "..\\deel\\torchlip\\tests\\*",
]

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "titles_only": False,
    "canonical_url": "https://torchlip.readthedocs.io/en/latest/",
}

html_logo = "_static/images/logo_white.svg"
html_static_path = ["_static"]

# html_theme = "sphinx_rtd_theme"

# html_sidebars = {
#     "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
# }

html_context = {"get_started": "/", "github_url": "https://github.com/deel-ai/torchlip"}

html_context = {
    "css_files": [
        "_static/theme_overrides.css",
    ],  # override wide tables in RTD theme
}
