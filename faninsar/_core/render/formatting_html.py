# This file is a copy of xarray/core/formatting_html.py,
# with modifications to format FanInSAR objects as HTML.

"""Format FanInSAR objects as HTML for Jupyter notebooks."""

from __future__ import annotations

import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from typing import TYPE_CHECKING

from xarray.core.formatting import (
    format_array_flat,
    inline_index_repr,
    inline_variable_array_repr,
)
from xarray.core.options import _get_boolean_with_default  # noqa: PLC2701

from .formatting import short_data_repr

STATIC_FILES = (
    ("xarray.static.html", "icons-svg-inline.html"),
    ("xarray.static.css", "style.css"),
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping

    import numpy as np
    from xarray import DataArray, Dataset, Variable
    from xarray.core.coordinates import DatasetCoordinates
    from xarray.core.datatree import DataTree
    from xarray.core.indexes import Index, Indexes

    from faninsar import Pairs

    T_Xarray = DataArray | Variable | Dataset


@lru_cache(None)
def _load_static_files() -> list[str]:
    """Lazily load the resource files into memory the first time they are needed."""
    return [
        files(package).joinpath(resource).read_text(encoding="utf-8")
        for package, resource in STATIC_FILES
    ]


def short_data_repr_html(array: np.ndarray) -> str:
    """Format "data" for DataArray and Variable."""
    internal_data = getattr(array, "variable", array)._data
    if hasattr(internal_data, "_repr_html_"):
        return internal_data._repr_html_()
    text = escape(short_data_repr(array))
    return f"<pre>{text}</pre>"


def format_dims(
    dim_sizes: tuple[str, int],
    dims_with_index: Mapping[Hashable, object],
) -> str:
    """Format dimensions as HTML."""
    if not dim_sizes:
        return ""

    dim_css_map = {
        dim: " class='xr-has-index'" if dim in dims_with_index else ""
        for dim in dim_sizes
    }

    dims_li = "".join(
        f"<li><span{dim_css_map[dim]}>{escape(str(dim))}</span>: {size}</li>"
        for dim, size in dim_sizes.items()
    )

    return f"<ul class='xr-dim-list'>{dims_li}</ul>"


def _icon(icon_name: str) -> str:
    """Return an SVG icon in HTML."""
    # icon_name should be defined in xarray/static/html/icon-svg-inline.html
    return (
        f"<svg class='icon xr-{icon_name}'><use xlink:href='#{icon_name}'></use></svg>"
    )


def summarize_attrs(attrs: dict[Hashable, object]) -> str:
    """Return a summary of the attributes as HTML."""
    from faninsar._core.render import HtmlProperties

    return str(HtmlProperties(attrs, margin="0px 0px 0px 1em"))


def summarize_variable(
    name: Hashable,
    var: Variable,
    is_index: bool = False,
    dtype: np.dtype = None,
) -> str:
    """Return a summary of the variable as HTML."""
    variable = var.variable if hasattr(var, "variable") else var

    cssclass_idx = " class='xr-has-index'" if is_index else ""
    dims_str = f"({', '.join(escape(dim) for dim in var.dims)})"
    name = escape(str(name))
    dtype = dtype or escape(str(var.dtype))

    # "unique" ids required to expand/collapse subsections
    attrs_id = "attrs-" + str(uuid.uuid4())
    data_id = "data-" + str(uuid.uuid4())
    disabled = "" if len(var.attrs) else "disabled"

    preview = escape(inline_variable_array_repr(variable, 35))
    attrs_ul = summarize_attrs(var.attrs)
    data_repr = short_data_repr_html(variable)

    attrs_icon = _icon("icon-file-text2")
    data_icon = _icon("icon-database")

    return (
        f"<div class='xr-var-name'><span{cssclass_idx}>{name}</span></div>"
        f"<div class='xr-var-dims'>{dims_str}</div>"
        f"<div class='xr-var-dtype'>{dtype}</div>"
        f"<div class='xr-var-preview xr-preview'>{preview}</div>"
        f"<input id='{attrs_id}' class='xr-var-attrs-in' "
        f"type='checkbox' {disabled}>"
        f"<label for='{attrs_id}' title='Show/Hide attributes'>"
        f"{attrs_icon}</label>"
        f"<input id='{data_id}' class='xr-var-data-in' type='checkbox'>"
        f"<label for='{data_id}' title='Show/Hide data repr'>"
        f"{data_icon}</label>"
        f"<div class='xr-var-attrs'>{attrs_ul}</div>"
        f"<div class='xr-var-data'>{data_repr}</div>"
    )


def summarize_coords(variables: DatasetCoordinates) -> str:
    """Return a summary of the coordinates as HTML."""
    li_items = []
    for k, v in variables.items():
        li_content = summarize_variable(k, v, is_index=k in variables.xindexes)
        li_items.append(f"<li class='xr-var-item'>{li_content}</li>")

    vars_li = "".join(li_items)

    return f"<ul class='xr-var-list'>{vars_li}</ul>"


def summarize_vars(variables: Mapping[Hashable, Variable]) -> str:
    """Return a summary of the variables as HTML."""
    vars_li = "".join(
        f"<li class='xr-var-item'>{summarize_variable(k, v)}</li>"
        for k, v in variables.items()
    )

    return f"<ul class='xr-var-list'>{vars_li}</ul>"


def short_index_repr_html(index: Index) -> str:
    """Return a short repr of the index as HTML."""
    if hasattr(index, "_repr_html_"):
        return index._repr_html_()

    return f"<pre>{escape(repr(index))}</pre>"


def summarize_index(coord_names: Hashable, index: Index) -> str:
    """Return a summary of the index as HTML."""
    name = "<br>".join([escape(str(n)) for n in coord_names])

    index_id = f"index-{uuid.uuid4()}"
    preview = escape(inline_index_repr(index))
    details = short_index_repr_html(index)

    data_icon = _icon("icon-database")

    return (
        f"<div class='xr-index-name'><div>{name}</div></div>"
        f"<div class='xr-index-preview'>{preview}</div>"
        f"<div></div>"
        f"<input id='{index_id}' class='xr-index-data-in' type='checkbox'/>"
        f"<label for='{index_id}' title='Show/Hide index repr'>{data_icon}</label>"
        f"<div class='xr-index-data'>{details}</div>"
    )


def summarize_indexes(indexes: Indexes) -> str:
    """Return a summary of the indexes as HTML."""
    indexes_li = "".join(
        f"<li class='xr-var-item'>{summarize_index(v, i)}</li>"
        for v, i in indexes.items()
    )
    return f"<ul class='xr-var-list'>{indexes_li}</ul>"


def summarize_indexes_faninsar(indexes: Indexes) -> str:
    """Return a summary of the indexes as HTML for FanInSAR objects."""
    from faninsar._core.render import HtmlIndexes

    return str(HtmlIndexes(indexes, format_array_flat))


def collapsible_section(
    name: Hashable,
    inline_details: str = "",
    details: str = "",
    n_items: int | None = None,
    enabled: bool = True,
    collapsed: bool = False,
) -> str:
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())

    has_items = n_items is not None and n_items
    n_items_span = "" if n_items is None else f" <span>({n_items})</span>"
    enabled = "" if enabled and has_items else "disabled"
    collapsed = "" if collapsed or not has_items else "checked"
    tip = " title='Expand/collapse section'" if enabled else ""

    return (
        f"<input id='{data_id}' class='xr-section-summary-in' "
        f"type='checkbox' {enabled} {collapsed}>"
        f"<label for='{data_id}' class='xr-section-summary' {tip}>"
        f"{name}:{n_items_span}</label>"
        f"<div class='xr-section-inline-details'>{inline_details}</div>"
        f"<div class='xr-section-details'>{details}</div>"
    )


def _mapping_section(
    mapping: T_Xarray,
    name: str,
    details_func: Callable,
    max_items_collapse: int,
    expand_option_name: str,
    enabled: bool = True,
) -> str:
    n_items = len(mapping)
    expanded = _get_boolean_with_default(
        expand_option_name,
        n_items < max_items_collapse,
    )
    collapsed = not expanded

    return collapsible_section(
        name,
        details=details_func(mapping),
        n_items=n_items,
        enabled=enabled,
        collapsed=collapsed,
    )


def dim_section(obj: DataArray | Dataset) -> str:
    dim_list = format_dims(obj.sizes, obj.xindexes.dims)

    return collapsible_section(
        "Dimensions",
        inline_details=dim_list,
        enabled=False,
        collapsed=True,
    )


def array_section(obj: DataArray) -> str:
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())
    collapsed = (
        "checked"
        if _get_boolean_with_default("display_expand_data", default=True)
        else ""
    )
    variable = getattr(obj, "variable", obj)
    preview = escape(inline_variable_array_repr(variable, max_width=70))
    data_repr = short_data_repr_html(obj)
    data_icon = _icon("icon-database")

    return (
        "<div class='xr-array-wrap'>"
        f"<input id='{data_id}' class='xr-array-in' type='checkbox' {collapsed}>"
        f"<label for='{data_id}' title='Show/hide data repr'>{data_icon}</label>"
        f"<div class='xr-array-preview xr-preview'><span>{preview}</span></div>"
        f"<div class='xr-array-data'>{data_repr}</div>"
        "</div>"
    )


def pairs_section(pairs: Pairs) -> str:
    """Format a Pairs object as HTML."""
    # function import is delayed to avoid circular import
    from faninsar._core.render import PairsSVG, add_svg_string

    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())
    preview = f"faninsar.Pairs<pairs={len(pairs)},dates={len(pairs.dates)}>"
    data_repr = pairs.to_dataframe().to_html(max_rows=5, justify="center")
    data_repr = add_svg_string(data_repr, PairsSVG(len(pairs)).to_tag())
    data_icon = _icon("icon-database")

    return (
        "<div class='xr-array-wrap'>"
        f"<input id='{data_id}' class='xr-array-in' type='checkbox' checked>"
        f"<label for='{data_id}' title='Show/hide data repr'>{data_icon}</label>"
        f"<div class='xr-array-preview xr-preview'><span>{preview}</span></div>"
        f"<div class='xr-array-data'>{data_repr}</div>"
        "</div>"
    )


coord_section = partial(
    _mapping_section,
    name="Coordinates",
    details_func=summarize_coords,
    max_items_collapse=25,
    expand_option_name="display_expand_coords",
)


datavar_section = partial(
    _mapping_section,
    name="Data variables",
    details_func=summarize_vars,
    max_items_collapse=15,
    expand_option_name="display_expand_data_vars",
)

index_section = partial(
    _mapping_section,
    name="Indexes",
    details_func=summarize_indexes,
    max_items_collapse=0,
    expand_option_name="display_expand_indexes",
)

index_section_faninsar = partial(
    _mapping_section,
    name="Indexes",
    details_func=summarize_indexes_faninsar,
    max_items_collapse=0,
    expand_option_name="display_expand_indexes",
)

attr_section = partial(
    _mapping_section,
    name="Attributes",
    details_func=summarize_attrs,
    max_items_collapse=10,
    expand_option_name="display_expand_attrs",
)

stats_section = partial(
    _mapping_section,
    name="Statistic",
    details_func=summarize_attrs,
    max_items_collapse=10,
    expand_option_name="display_expand_attrs",
)

property_section = partial(
    _mapping_section,
    name="Properties",
    details_func=summarize_indexes,
    max_items_collapse=0,
    expand_option_name="display_expand_indexes",
)


def _get_indexes_dict(indexes: Indexes) -> dict[tuple[str], int]:
    """Return a dictionary of indexes with their variables."""
    return {
        tuple(index_vars.keys()): idx for idx, index_vars in indexes.group_by_index()
    }


def _obj_repr(obj: T_Xarray, header_components: str, sections: list) -> str:
    """Return HTML repr of an faninsar object.

    If CSS is not injected (untrusted notebook), fallback to the plain text repr.

    """
    header = f"<div class='xr-header'>{''.join(h for h in header_components)}</div>"
    sections = "".join(f"<li class='xr-section-item'>{s}</li>" for s in sections)

    icons_svg, css_style = _load_static_files()
    return (
        "<div>"
        f"{icons_svg}<style>{css_style}</style>"
        f"<pre class='xr-text-repr-fallback'>{escape(repr(obj))}</pre>"
        "<div class='xr-wrap' style='display:none'>"
        f"{header}"
        f"<ul class='xr-sections'>{sections}</ul>"
        "</div>"
        "</div>"
    )


def array_repr(arr: DataArray) -> str:
    dims = OrderedDict((k, v) for k, v in zip(arr.dims, arr.shape))
    indexed_dims = arr.xindexes.dims if hasattr(arr, "xindexes") else {}

    obj_type = f"faninsar.{type(arr).__name__}"
    arr_name = f"'{arr.name}'" if getattr(arr, "name", None) else ""

    header_components = [
        f"<div class='xr-obj-type'>{obj_type}</div>",
        f"<div class='xr-array-name'>{arr_name}</div>",
        format_dims(dims, indexed_dims),
    ]

    sections = [array_section(arr)]

    if hasattr(arr, "coords"):
        sections.append(coord_section(arr.coords))

    if hasattr(arr, "xindexes"):
        indexes = _get_indexes_dict(arr.xindexes)
        sections.append(index_section(indexes))
    if hasattr(arr, "attrs"):
        sections.append(attr_section(arr.attrs))
    if hasattr(arr, "stats"):
        sections.append(stats_section(arr.stats))

    return _obj_repr(arr, header_components, sections)


def dataset_repr(ds: Dataset) -> str:
    obj_type = f"faninsar.{type(ds).__name__}"

    header_components = [f"<div class='xr-obj-type'>{escape(obj_type)}</div>"]

    sections = [
        dim_section(ds),
        coord_section(ds.coords),
        datavar_section(ds.data_vars),
        index_section(_get_indexes_dict(ds.xindexes)),
        attr_section(ds.attrs),
    ]

    return _obj_repr(ds, header_components, sections)


def pairs_repr(pairs: Pairs) -> str:
    # function import is delayed to avoid circular import
    from faninsar._core.render import HtmlDims

    obj_type = "faninsar.Pairs"
    dim = HtmlDims({"pairs": len(pairs), "dates": len(pairs.dates)}, sep="=")
    header_components = [
        f"<div class='xr-obj-type'>{escape(obj_type)}</div>",
        f"<div class='xr-array-name'>{''}</div>",
        str(dim),
    ]

    sections = [
        pairs_section(pairs),
        index_section_faninsar(pairs.xindexes),
    ]

    return _obj_repr(pairs, header_components, sections)


def summarize_datatree_children(children: Mapping[str, DataTree]) -> str:
    n_children = len(children) - 1

    # Get result from datatree_node_repr and wrap it
    def lines_callback(n, c, end):  # noqa: ANN001, ANN202
        return _wrap_datatree_repr(datatree_node_repr(n, c), end=end)

    children_html = "".join(
        (
            lines_callback(n, c, end=False)  # Long lines
            if i < n_children
            else lines_callback(n, c, end=True)
        )  # Short lines
        for i, (n, c) in enumerate(children.items())
    )
    style_str = (
        "<div style='display: inline-grid; "
        "grid-template-columns: 100%; "
        "grid-column: 1 / -1'>"
    )
    return f"{style_str}{children_html}</div>"


children_section = partial(
    _mapping_section,
    name="Groups",
    details_func=summarize_datatree_children,
    max_items_collapse=1,
    expand_option_name="display_expand_groups",
)


def datatree_node_repr(group_title: str, dt: DataTree) -> str:
    header_components = [f"<div class='xr-obj-type'>{escape(group_title)}</div>"]

    ds = dt._to_dataset_view(rebuild_dims=False)

    sections = [
        children_section(dt.children),
        dim_section(ds),
        coord_section(ds.coords),
        datavar_section(ds.data_vars),
        attr_section(ds.attrs),
    ]

    return _obj_repr(ds, header_components, sections)


def _wrap_datatree_repr(r: str, end: bool = False) -> str:
    """Wrap HTML representation with a tee to the left of it.

    Enclosing HTML tag is a <div> with :code:`display: inline-grid` style.

    Turns:
    [    title    ]
    |   details   |
    |_____________|

    into (A):
    |─ [    title    ]
    |  |   details   |
    |  |_____________|

    or (B):
    └─ [    title    ]
       |   details   |
       |_____________|

    Parameters
    ----------
    r: str
        HTML representation to wrap.
    end: bool
        Specify if the line on the left should continue or end.

        Default is True.

    Returns
    -------
    str
        Wrapped HTML representation.

        Tee color is set to the variable :code:`--xr-border-color`.

    """
    # height of line
    end = bool(end)
    height = "100%" if end is False else "1.2em"
    return "".join(
        [
            "<div style='display: inline-grid; grid-template-columns: 0px 20px auto; width: 100%;'>",  # noqa: E501
            "<div style='",
            "grid-column-start: 1;",
            "border-right: 0.2em solid;",
            "border-color: var(--xr-border-color);",
            f"height: {height};",
            "width: 0px;",
            "'>",
            "</div>",
            "<div style='",
            "grid-column-start: 2;",
            "grid-row-start: 1;",
            "height: 1em;",
            "width: 20px;",
            "border-bottom: 0.2em solid;",
            "border-color: var(--xr-border-color);",
            "'>",
            "</div>",
            "<div style='",
            "grid-column-start: 3;",
            "'>",
            r,
            "</div>",
            "</div>",
        ],
    )


def datatree_repr(dt: DataTree) -> str:
    obj_type = f"datatree.{type(dt).__name__}"
    return datatree_node_repr(obj_type, dt)


def repr_inline(
    cls: object | str,
    data_ls: list,
    length: int,
    *args,
    **kwargs,
) -> str:
    """Return the inline representation of the class.

    Parameters
    ----------
    cls : object or str
        Class used to generate the inline representation.
    data_ls : list
        List of data in string format.
    length : int
        Length of the class.
    max_width : int, optional
        Maximum width of the inline representation, by default None.
    *args, **kwargs : optional
        Additional arguments.

    """
    name = cls.__class__.__name__ if not isinstance(cls, str) else cls
    max_width = _parse_max_width(*args, **kwargs)
    string = f"{name}([...], length={length})"
    if len(string) > max_width:
        string = f"{name}(length={length})"
        if len(string) > max_width:
            return ""
        return string

    data_str_used = ""
    all_used = False
    used_data = False
    for i, data in enumerate(data_ls):
        data_str = str(data)
        data_str_used += f"{data_str}, "
        if len(string) + len(data_str_used) > max_width:
            break
        used_data = True
        if i == len(data_ls) - 1:
            all_used = True

    if used_data:
        if all_used:
            string = f"{name}([{data_str_used[:-2]}], length={length})"
        else:
            string = f"{name}([{data_str_used}...], length={length})"

    return string


def _parse_max_width(*args, **kwargs) -> int:
    """Parse the maximum width from the arguments."""
    if args and isinstance(args[0], int):
        return args[0]
    max_width = kwargs.get("max_width")
    if max_width is not None:
        return max_width
    return 75
