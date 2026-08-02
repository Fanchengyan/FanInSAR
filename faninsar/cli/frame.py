"""``faninsar frame`` — process a full multi-swath frame in one command."""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


def run_frame_cli(
    *,
    reference: str,
    secondary: str,
    output: str,
    dem: str | None = None,
    reference_orbit: str | None = None,
    secondary_orbit: str | None = None,
    swaths: str = "IW1,IW2,IW3",
    bursts: str | None = None,
    az_looks: int = 2,
    rg_looks: int = 10,
    goldstein: float = 0.5,
    device: str = "cpu",
) -> int:
    """Run the full-frame production pipeline from the command line.

    Returns
    -------
    int
        Exit code (0 on success).

    """
    from faninsar.processing.geometry.dem import GeoidAdjustedDEM, RasterDEM
    from faninsar.processing.geometry.egm96 import EGM96Geoid
    from faninsar.processing.pipeline import run_full_frame

    dem_sampler = None
    if dem is not None:
        dem_sampler = GeoidAdjustedDEM(
            RasterDEM(path=dem, interpolation="biquintic"), EGM96Geoid()
        )
    burst_map: dict[str, list[int]] | None = None
    if bursts is not None:
        burst_map = {}
        current_sw: str | None = None
        for segment in bursts.split(","):
            if ":" in segment:
                sw, _, first = segment.partition(":")
                if not first:
                    message = f"invalid --bursts entry {segment!r}"
                    raise SystemExit(message)
                burst_map[sw.strip()] = [int(first)]
                current_sw = sw.strip()
            elif current_sw is not None:
                burst_map[current_sw].append(int(segment))
            else:
                message = "--bursts must start with a swath entry like IW1:0"
                raise SystemExit(message)

    state = run_full_frame(
        reference,
        secondary,
        output_dir=Path(output),
        dem=dem_sampler,
        swaths=tuple(name.strip() for name in swaths.split(",")),
        multilook=(az_looks, rg_looks),
        goldstein_alpha=goldstein,
        device=device,
        reference_orbit_path=reference_orbit,
        secondary_orbit_path=secondary_orbit,
        burst_indices=burst_map,
    )
    assert state.complex_ifg is not None
    logger.info("merged frame: %s", state.complex_ifg.shape)
    logger.info("timings: %s", state.stage_timings_s)
    return 0
