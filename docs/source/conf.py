# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FanInSAR"
copyright = "2024, Fan Chengyan (Fancy)"
author = "Fan Chengyan (Fancy)"
release = "v0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = ["colon_fence"]
myst_url_schemes = ["http", "https", "mailto"]
suppress_warnings = ["mystnb.unknown_mime_type"]
nb_execution_mode = "off"
autodoc_inherit_docstrings = True
# templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo/logo.png"

html_theme_options = {
    "show_toc_level": 2,
    "show_nav_level": 2,
    "header_links_before_dropdown": 10,
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Fanchengyan/FanInSAR",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/FanInSAR",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
}

video_enforce_extra_source = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    ":show-inheritance:": True,
}
html_context = {
    "github_url": "https://github.com",
    "github_user": "Fanchengyan",
    "github_repo": "FanInSAR",
    "github_version": "main",
    "doc_path": "docs/source",
}


# connect docs in other projects
intersphinx_mapping = {
    "cartopy": (
        "https://scitools.org.uk/cartopy/docs/latest/",
        "https://scitools.org.uk/cartopy/docs/latest/objects.inv",
    ),
    "geopandas": (
        "https://geopandas.org/en/stable/",
        "https://geopandas.org/en/stable/objects.inv",
    ),
    "fiona": (
        "https://fiona.readthedocs.io/en/stable/",
        "https://fiona.readthedocs.io/en/stable/objects.inv",
    ),
    "matplotlib": (
        "https://matplotlib.org/stable/",
        "https://matplotlib.org/stable/objects.inv",
    ),
    "numpy": (
        "https://numpy.org/doc/stable/",
        "https://numpy.org/doc/stable/objects.inv",
    ),
    "pandas": (
        "https://pandas.pydata.org/pandas-docs/stable/",
        "https://pandas.pydata.org/pandas-docs/stable/objects.inv",
    ),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "pyproj": (
        "https://pyproj4.github.io/pyproj/stable/",
        "https://pyproj4.github.io/pyproj/stable/objects.inv",
    ),
    "python": (
        "https://docs.python.org/3",
        "https://docs.python.org/3/objects.inv",
    ),
    "rasterio": (
        "https://rasterio.readthedocs.io/en/stable/",
        "https://rasterio.readthedocs.io/en/stable/objects.inv",
    ),
    "shapely": (
        "https://shapely.readthedocs.io/en/stable/",
        "https://shapely.readthedocs.io/en/stable/objects.inv",
    ),
    "pyproj": (
        "https://pyproj4.github.io/pyproj/stable/",
        "https://pyproj4.github.io/pyproj/stable/objects.inv",
    ),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torchgeo": (
        "https://torchgeo.readthedocs.io/en/latest/",
        "https://torchgeo.readthedocs.io/en/latest/objects.inv",
    ),
}


# -- faninsar package ----------------------------------------------------------
import os
import sys

# Location of Sphinx files
sys.path.insert(0, os.path.abspath("./../.."))
