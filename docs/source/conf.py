# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FanInSAR"
copyright = "2023, Fan Chengyan (Fancy)"
author = "Fan Chengyan (Fancy)"
release = "v1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ['_static']


video_enforce_extra_source = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    ":show-inheritance:": True,
}

# -- faninsar package ----------------------------------------------------------
import os
import sys

# Location of Sphinx files
sys.path.insert(0, os.path.abspath("./../.."))
