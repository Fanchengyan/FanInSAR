#!/usr/bin/env python3
"""Manual demo of HistColorbar with imshow and contourf.

Run:
    python tests/manual/hcb_demo.py

This script creates a few figures demonstrating:
- imshow + vertical colorbar (right) with extend triangles
- imshow + horizontal colorbar (bottom) with extend triangles
- contourf + vertical colorbar (left) with discrete levels and extend
- log-scaled histogram counts example
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm

# Ensure Figure.hist_colorbar is registered
from faninsar.plots import HistColorbar  # noqa: F401  # side-effect registration


def demo_imshow() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0, scale=1.0, size=(120, 160))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    # Vertical, right, extend=both
    ax = axes[0]
    im = ax.imshow(data, cmap="viridis", norm=Normalize(vmin=-3, vmax=3)
    )
    ax.set_title("imshow | vertical-right | extend=both")
    hcb = fig.hist_colorbar(
        data=data.ravel(),
        mappable=im,
        ax=ax,
        location="right",
        orientation="vertical",
        extend="both",
        label="Value",
        hist_label="Count",
        hist_bins="auto",
    )
    # hcb.ax_hist.grid(True, axis="both", which="major")

    # Horizontal, bottom, extend=max
    ax = axes[1]
    im = ax.imshow(data, cmap="plasma", norm=Normalize(vmin=-2.5, vmax=2.0))
    ax.set_title("imshow | horizontal-bottom | extend=max")
    hcb1 = fig.hist_colorbar(
        data=data.ravel(),
        mappable=im,
        ax=ax,
        location="bottom",
        orientation="horizontal",
        extend="max",
        label="Value",
        hist_label="Count",
        # hist_bins=60,
    )
    # hcb1.ax_hist.grid(True, axis="both", which="major")
    # print(hcb1.ax_hist.get_ylim())

    fig.suptitle("HistColorbar — imshow examples")
    # plt.savefig("hcb_demo.png", dpi=300)]\
    plt.show()


def demo_contourf_and_log() -> None:
    # Synthetic smooth field
    y, x = np.mgrid[-2:2:200j, -3:3:300j]
    r = np.hypot(x, y)
    z = np.sinc(r) * np.cos(2 * x) * np.sin(2 * y)

    levels = np.linspace(-0.6, 0.6, 10)
    cmap = "coolwarm"
    norm = BoundaryNorm(levels, ncolors=256, extend="both")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    # contourf, vertical-left, extend=both (discrete levels)
    ax = axes[0]
    cs = ax.contourf(x, y, z, levels=levels, cmap=cmap, extend="both")
    ax.set_title("contourf | vertical-left | extend=both (BoundaryNorm)")
    hcb = fig.hist_colorbar(
        data=z.ravel(),
        mappable=cs,
        ax=ax,
        location="left",
        orientation="vertical",
        extend="both",
        label="Level",
        hist_label="Count",
        hist_bins="auto",
    )
    # hcb.ax_hist.grid(True, axis="y", which="major")

    # log-count histogram demonstration with skewed data
    rng = np.random.default_rng(123)
    skew = rng.exponential(scale=2.0, size=40_000)
    ax = axes[1]
    im = ax.imshow(rng.normal(size=(50, 50)), cmap="magma")
    ax.set_title("imshow | vertical-right | log-count histogram")
    hcb = fig.hist_colorbar(
        data=skew,
        mappable=im,
        ax=ax,
        location="right",
        orientation="vertical",
        extend="min",
        label="Value",
        hist_label="log Count",
        # hist_bins=100,
        log=True,
    )
    # hcb.ax_hist.grid(True, axis="y", which="major")


    fig.suptitle("HistColorbar — contourf & log-count examples")
    # plt.savefig("hcb_demo_contourf_log.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    demo_imshow()
    demo_contourf_and_log()
    plt.show()
