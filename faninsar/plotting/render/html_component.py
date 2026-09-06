"""Render HTML table for table."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Literal

from dominate.tags import (
    _input,
    body,
    div,
    dl,
    html_tag,
    label,
    li,
    p,
    span,
    table,
    td,
    th,
    thead,
    tr,
    ul,
)
from dominate.util import raw

from faninsar.plotting.render.formatting_html import short_index_repr_html

if TYPE_CHECKING:
    from collections.abc import Hashable

    import pandas as pd

li.is_inline = True
span.is_inline = True
div.is_inline = True
label.is_inline = True


def _icon(icon_name: str) -> str:
    """Return an SVG icon in HTML."""
    # icon_name should be defined in xarray/static/html/icon-svg-inline.html
    return (
        f"<svg class='icon xr-{icon_name}'><use xlink:href='#{icon_name}'></use></svg>"
    )


attrs_icon = raw(_icon("icon-file-text2"))
data_icon = raw(_icon("icon-database"))


class BaseHTML:
    """Base class for small HTML renderers."""

    def to_tag(self) -> html_tag:
        """Return the rendered dominate tag."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return the rendered HTML string."""
        return str(self.to_tag())


class HtmlTable(BaseHTML):
    """Render table to HTML."""

    def __init__(
        self,
        table: pd.DataFrame,
        svg: html_tag | None = None,
    ) -> None:
        """Initialize the Properties class.

        Parameters
        ----------
        table : pd.DataFrame
            The table to be rendered as a html table.
        svg: html_tag, optional
            The SVG tag to be shown in the right column. Default is None.

        """
        self.table = table
        self.svg = svg

    def to_tag(self) -> html_tag:
        """Return a rendered dominate tag containing a table of table."""
        table_tag = table(style="margin-left: 32px;")

        # Create header row
        header_row = tr()
        if self.table.index.name:
            header_row += th(self.table.index.name)
        for column in self.table.columns:
            header_row += th(column)
        table_tag += thead(header_row)

        # Create data rows
        for index, row in self.table.iterrows():
            data_row = tr()
            data_row += th(index)
            for value in row:
                data_row += td(value)
            table_tag += data_row

        if self.svg:
            table_tag = add_svg(table_tag, self.svg)

        return table_tag


class HtmlProperties(BaseHTML):
    """Render properties to HTML table."""

    def __init__(
        self,
        properties: dict,
        column: int = 1,
        width: str = "100%",
        margin: str = "0px",
        description: str | None = None,
        svg: html_tag | None = None,
    ) -> None:
        """Initialize the Properties class.

        Parameters
        ----------
        properties : dict
            The properties to be rendered as a html table.
        column: int, optional
            The column to be shown in the table. Default is 1.
        width: str, optional
            The width of the table. Default is "100%".
        margin: str, optional
            The margin of the table. Default is "0px".
        description: str
            The description of the property.
        svg: html_tag, optional
            The SVG tag to be shown in the right column. Default is None.

        """
        self.properties = properties
        self.column = column
        self.width = width
        self.margin = margin
        self.description = description
        self.svg = svg

    def to_tag(self) -> html_tag:
        """Return a rendered dominate tag containing a table of properties."""
        div_tag = div(style=f"width: {self.width}; margin: {self.margin};")
        table_tag = table(
            style="margin: 5px 2.5%; width: 95%;",
        )
        if self.description is not None:
            table_tag += p(
                f"{self.description}:", style="color: gray;margin-left: 1em;"
            )
        body_tag = body()
        for i, (key, value) in enumerate(self.properties.items()):
            if i % self.column == 0:
                row = tr()
            row += th(str(key))
            row += td(str(value))
            if i % self.column == self.column - 1:
                body_tag += row
        if i % self.column != self.column - 1:
            body_tag += row
        table_tag += body_tag

        if self.svg:
            table_tag = add_svg(table_tag, self.svg)

        div_tag += table_tag
        return div_tag


class HtmlDims(BaseHTML):
    """Render Dims in xarray html style."""

    def __init__(
        self,
        dims: dict[str, object],
        sep: Literal[":", "="] = ":",
    ) -> None:
        """Initialize the HtmlDims class.

        Parameters
        ----------
        dims : dict
            The dims to be rendered.
        sep: Literal[":", "="], optional
            The separator between key and value. Default is ":".

        """
        self.dims = dims
        if sep not in [":", "="]:
            msg = "The separator must be ':' or '='."
            raise ValueError(msg)
        if sep == ":":
            self.sep = ": "
        self.sep = sep

    def to_tag(self) -> html_tag:
        """Return a rendered dominate tag containing a table of properties."""
        ul_tag = ul(cls="xr-dim-list")
        for key, value in self.dims.items():
            ul_tag += li(span(f"{key}", cls="xr-has-index"), f"{self.sep}{value}")
        return ul_tag


class HtmlIndexes(BaseHTML):
    """Render indexes in xarray html style."""

    def __init__(
        self,
        indexes: dict[Hashable, object],
        format_func: callable,
        max_width: int = 50,
    ) -> None:
        """Initialize the HtmlIndexes class.

        Parameters
        ----------
        indexes : dict
            The indexes to be rendered.
        format_func: callable, optional
            The function to format the index data.
        max_width: int, optional
            The maximum width of the index. Default is 50.

        """
        self.indexes = indexes
        self.format_func = format_func
        self.max_width = max_width

    def to_tag(self) -> html_tag:
        """Return a rendered dominate tag containing a table of properties."""
        ul_tag = ul(cls="xr-var-list")
        for key, value in self.indexes.items():
            li_tag = li(cls="xr-var-item")

            li_tag += div(span(f"{key}", cls="xr-has-index"), cls="xr-var-name")
            # 1. using class name replace the type name 2. disable the dims
            # li_tag += div("", cls="xr-var-dims")
            li_tag += div(f"{value.__class__.__name__}", cls="xr-var-dtype")
            li_tag += div(
                f"{self.format_func(value, max_width=self.max_width)}",
                cls="xr-var-preview xr-preview",
            )

            attrs_id = "attrs-" + str(uuid.uuid4())
            data_id = "data-" + str(uuid.uuid4())
            disabled = True

            attrs_ul = div(dl(cls="xr-attrs"), cls="xr-var-attrs")
            if hasattr(value, "attrs") and len(value.attrs) > 0:
                # attrs_ul = raw(summarize_attrs(value.attrs))
                attrs_ul = HtmlProperties(
                    value.attrs, description="Attributes"
                ).to_tag()
                disabled = False
            data_repr = raw(short_index_repr_html(value))

            li_tag += _input(
                cls="xr-var-attrs-in",
                id=attrs_id,
                type="checkbox",
                disabled=disabled,
            )
            li_tag += label(attrs_icon, fr=attrs_id, title="Show/Hide attributes")
            li_tag += _input(id=data_id, cls="xr-var-data-in", type="checkbox")
            li_tag += label(data_icon, fr=data_id, title="Show/Hide data repr")
            li_tag += div(attrs_ul, cls="xr-var-attrs")
            li_tag += div(data_repr, cls="xr-var-data")

            ul_tag += li_tag
        return ul_tag


def add_svg(tag: html_tag, svg: html_tag) -> html_tag:
    """Add a SVG to the right of the table.

    Parameters
    ----------
    tag : html_tag
        The tag to be shown in the left column.
    svg : html_tag
        The SVG tag to be shown in the right column.

    Returns
    -------
    html_tag
        The table tag with SVG.

    """
    table_tag = table()
    table_tag += td(tag)
    table_tag += td(svg)
    return table_tag


def add_svg_string(tag: str, svg: str) -> str:
    """Add a SVG to the right of the table.

    This function is used when both tag and svg are string.

    Parameters
    ----------
    tag : str
        The tag to be shown in the left column.
    svg : str
        The SVG tag to be shown in the right column.

    Returns
    -------
    str
        The table tag with SVG.

    """
    return f"<table><td>{tag}</td><td>{svg}</td></table>"
