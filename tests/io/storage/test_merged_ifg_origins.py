"""Regression: geo merged IFG placement uses base-canvas origin + look offset.

The persisted placement contract (``ifg_sample_placement``) is
``row_origin + look_origin * looks``; the geo merged writer therefore records
the base canvas origin (0 for the global fine grid) in ``row/col_origin`` and
the first full look block index in ``look_origin`` — persisting the block's
native offset in all three fields double-counts it.
"""

from faninsar.stack.session import _merged_ifg_origins


def test_geo_merged_uses_base_canvas_origin_and_look_offset() -> None:
    """Geo merged writes base canvas origin once, with the look block offset."""
    row, col, look = _merged_ifg_origins((6, 1371, 43, 8000), 1, 5, geo=True)
    assert (row, col, look) == (0, 0, (6, 9))
    # Authoritative first native sample, computed once from the contract:
    assert (row + look[0] * 1, col + look[1] * 5) == (6, 45)
    # First FULL look block inside the layout window (partial block dropped):
    assert look[0] * 1 >= 6
    assert look[1] * 5 >= 43
    assert look[1] * 5 - 43 < 5


def test_radar_merged_writer_shape_unchanged() -> None:
    """Radar/burst semantics are out of scope: block-start base is kept."""
    row, col, look = _merged_ifg_origins((6, 1371, 43, 8000), 2, 10, geo=False)
    assert (row, col, look) == (6, 50, (3, 5))
