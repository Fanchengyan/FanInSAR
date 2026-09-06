"""SVG diagrams for pair topology."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dominate.svg import line, rect, svg, text
from dominate.tags import html_tag

if TYPE_CHECKING:
    from dominate.tags import html_tag


class PairsSVG:
    """Render a compact pair topology diagram."""

    def __init__(self, n_pairs: int) -> None:
        """Initialize the diagram for *n_pairs* relationships."""
        self.n_pairs = n_pairs

    def __str__(self) -> str:
        """Return the rendered SVG as a string."""
        return str(self.to_tag())

    def to_tag(self) -> html_tag:
        """Build the SVG dominate tag."""
        # Fixed dimensions
        width = 200
        height = 200

        # Maximum number of lines directly displayed
        max_direct_lines = 10

        max_pairs = 1000  # Maximum number of pairs to display
        max_display_pairs = 30  # Maximum number of pairs to display

        # Calculate the maximum number of lines to display
        pairs_used = min(max_pairs, self.n_pairs)
        if self.n_pairs <= max_direct_lines:
            lines_to_display = self.n_pairs
        else:
            lines_to_display = int(
                max_direct_lines
                + (pairs_used - max_direct_lines)
                * (
                    (max_display_pairs - max_direct_lines)
                    / (max_pairs - max_direct_lines)
                ),
            )
        # avoid zero lines
        lines_to_display = max(lines_to_display, 1)

        # Calculate the number of lines to display
        line_height = (height - 50) / lines_to_display

        # Create SVG tag
        svg_tag = svg(
            width=width,
            height=height,
        )

        # Add rectangles for primary and secondary
        x1 = 10
        x2 = 110
        fill = "#ECB172A0" if self.n_pairs <= 10 else "#8B4903A0"
        stroke = "#bfbfbfA0" if self.n_pairs <= 10 else "#9e9e9eA0"
        rect_width = 80
        rect_height = line_height * lines_to_display
        svg_tag += rect(x=x1, y=25, width=rect_width, height=rect_height, fill=fill)
        svg_tag += rect(x=x2, y=25, width=rect_width, height=rect_height, fill=fill)

        # Add lines
        for i in range(lines_to_display + 1):
            stroke_width = 1 if i % lines_to_display == 0 else 0.75
            y = 25 + i * line_height
            svg_tag += line(
                x1=x1,
                y1=y,
                x2=x1 + rect_width,
                y2=y,
                stroke=stroke,
                stroke_width=stroke_width,
            )  # Primary lines
            svg_tag += line(
                x1=x2,
                y1=y,
                x2=x2 + rect_width,
                y2=y,
                stroke=stroke,
                stroke_width=stroke_width,
            )  # Secondary lines

        # Add text labels and count
        svg_tag += text(
            "primary",
            x=50,
            y=height - 3,
            style="font-size:0.85em; font-weight:bold; text-anchor:middle;",
        )
        svg_tag += text(
            "secondary",
            x=150,
            y=height - 3,
            style="font-size:0.85em; font-weight:bold; text-anchor:middle;",
        )
        svg_tag += text(
            f"{self.n_pairs} Pairs",
            x=100,
            y=15,
            style="font-size:0.9em; font-weight:bold; text-anchor:middle;",
        )

        return svg_tag
