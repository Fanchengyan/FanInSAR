"""HTML, SVG, and compact representations for scientific values."""

from __future__ import annotations

from .formatting_html import array_repr, repr_inline
from .html_component import (
    HtmlDims,
    HtmlIndexes,
    HtmlProperties,
    HtmlTable,
    add_svg,
    add_svg_string,
)
from .svg_graph import PairsSVG

__all__ = [
    "HtmlDims",
    "HtmlIndexes",
    "HtmlProperties",
    "HtmlTable",
    "PairsSVG",
    "add_svg",
    "add_svg_string",
    "array_repr",
    "repr_inline",
]
