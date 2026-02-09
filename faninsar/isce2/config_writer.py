"""INI configuration file writer for ISCE2 processing.

This module provides a configuration file writer that generates INI format
config files compatible with ISCE2's SentinelWrapper.py.
"""

from __future__ import annotations

from pathlib import Path

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


class ConfigWriter:
    """INI format configuration file writer (corresponds to original config class).

    Generates INI configuration files compatible with SentinelWrapper.py.

    Parameters
    ----------
    filepath : Path
        Configuration file path (should end with .ini).

    Examples
    --------
    >>> config = ConfigWriter(Path("configs/config_reference.ini"))
    >>> config.write_sentinel1_tops(
    ...     {
    ...         "dirname": "/data/safe",
    ...         "swaths": "1 2 3",
    ...         "orbit_dir": "/data/orbits",
    ...         "outdir": "/work/reference",
    ...         "auxdir": "/data/aux",
    ...         "pol": "vv",
    ...     }
    ... )
    >>> config.write_topo(
    ...     {
    ...         "reference": "/work/reference",
    ...         "dem": "/data/dem.wgs84",
    ...         "geom_referenceDir": "/work/geom_reference",
    ...         "numProcess": 4,
    ...     }
    ... )
    >>> config.finalize()

    """

    def __init__(self, filepath: Path) -> None:
        """Initialize the configuration file writer."""
        if not filepath.suffix:
            filepath = filepath.with_suffix(".ini")

        self.filepath = filepath
        self.f = open(filepath, "w")
        self.function_counter = 0

        # Write [Common] section
        self.f.write("[Common]\n")
        self.f.write("##########################\n")

    def write_sentinel1_tops(self, params: dict) -> None:
        """Write Sentinel1_TOPS configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - dirname: SAFE file path
            - swaths: swath list (e.g., "1 2 3")
            - orbit_dir: orbit directory (or orbit: orbit file)
            - orbit_type: "precise" or "restituted"
            - outdir: output directory
            - auxdir: auxiliary file directory
            - bbox: bounding box (optional)
            - pol: polarization

        """
        self.function_counter += 1
        self.f.write("##########################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("Sentinel1_TOPS : \n")
        self.f.write(f"dirname : {params['dirname']}\n")
        self.f.write(f"swaths : {params['swaths']}\n")

        # Orbit file handling
        if params.get("orbit_type") == "precise":
            self.f.write(f"orbitdir : {params['orbit_dir']}\n")
        else:
            self.f.write(f"orbit : {params.get('orbit_file', '')}\n")

        self.f.write(f"outdir : {params['outdir']}\n")
        self.f.write(f"auxdir : {params.get('auxdir', '')}\n")

        if params.get("bbox"):
            self.f.write(f"bbox : {params['bbox']}\n")

        self.f.write(f"pol : {params.get('pol', 'vv')}\n")
        self.f.write("##########################\n")

    def write_topo(self, params: dict) -> None:
        """Write topo configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - reference: reference directory
            - dem: DEM file path
            - geom_referenceDir: geometry output directory
            - numProcess: number of parallel processes

        """
        self.function_counter += 1
        self.f.write("##########################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("topo : \n")
        self.f.write(f"reference : {params['reference']}\n")
        self.f.write(f"dem : {params['dem']}\n")
        self.f.write(f"geom_referenceDir : {params['geom_referenceDir']}\n")
        self.f.write(f"numProcess : {params.get('numProcess', 1)}\n")
        self.f.write("##########################\n")

    def write_baseline(self, params: dict) -> None:
        """Write computeBaseline configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - reference: reference directory
            - secondary: secondary directory
            - baseline_file: baseline output file path

        """
        self.function_counter += 1
        self.f.write("##########################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("computeBaseline : \n")
        self.f.write(f"reference : {params['reference']}\n")
        self.f.write(f"secondary : {params['secondary']}\n")
        self.f.write(f"baseline_file : {params['baseline_file']}\n")
        self.f.write("##########################\n")

    def write_baseline_grid(self, params: dict) -> None:
        """Write baselineGrid configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - reference: reference directory
            - secondary: secondary directory
            - baseline_file: baseline output file path

        """
        self.function_counter += 1
        self.f.write("##########################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("baselineGrid : \n")
        self.f.write(f"reference : {params['reference']}\n")
        self.f.write(f"secondary : {params['secondary']}\n")
        self.f.write(f"baseline_file : {params['baseline_file']}\n")
        self.f.write("##########################\n")

    def write_geo2rdr(self, params: dict) -> None:
        """Write geo2rdr configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - secondary: secondary directory
            - reference: reference directory
            - geom_reference: geometry reference directory
            - coreg_dir: coregistered output directory
            - overlap: overlap flag (True/False)
            - use_gpu: GPU flag (True/False)
            - misreg_az: azimuth misregistration file (optional)
            - misreg_rng: range misregistration file (optional)

        """
        self.function_counter += 1
        self.f.write("##########################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("geo2rdr :\n")
        self.f.write(f"secondary : {params['secondary']}\n")
        self.f.write(f"reference : {params['reference']}\n")
        self.f.write(f"geom_referenceDir : {params['geom_reference']}\n")
        self.f.write(f"coregSLCdir : {params['coreg_dir']}\n")
        self.f.write(f"overlap : {params.get('overlap', 'False')}\n")
        self.f.write(f"useGPU : {'True' if params.get('use_gpu') else 'False'}\n")

        if params.get("misreg_az"):
            self.f.write(f"azimuth_misreg : {params['misreg_az']}\n")
        if params.get("misreg_rng"):
            self.f.write(f"range_misreg : {params['misreg_rng']}\n")
        self.f.write("##########################\n")

    def write_resamp_withcarrier(self, params: dict) -> None:
        """Write resamp_withCarrier configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - secondary: secondary directory
            - reference: reference directory
            - coreg_dir: coregistered directory
            - overlap: overlap flag
            - misreg_az: azimuth misregistration file (optional)
            - misreg_rng: range misregistration file (optional)

        """
        self.function_counter += 1
        self.f.write("##########################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("resamp_withCarrier : \n")
        self.f.write(f"secondary : {params['secondary']}\n")
        self.f.write(f"reference : {params['reference']}\n")
        self.f.write(f"coregdir : {params['coreg_dir']}\n")
        self.f.write(f"overlap : {params.get('overlap', 'False')}\n")

        if params.get("misreg_az"):
            self.f.write(f"azimuth_misreg : {params['misreg_az']}\n")
        if params.get("misreg_rng"):
            self.f.write(f"range_misreg : {params['misreg_rng']}\n")
        self.f.write("##########################\n")

    def write_generate_igram(self, params: dict) -> None:
        """Write generateIgram configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - reference: reference directory
            - secondary: secondary directory
            - interferogram: interferogram output directory
            - flatten: flatten flag
            - prefix: interferogram prefix
            - overlap: overlap flag
            - misreg_az: azimuth misregistration (optional)
            - misreg_rng: range misregistration (optional)

        """
        self.function_counter += 1
        self.f.write("###################################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("generateIgram : \n")
        self.f.write(f"reference : {params['reference']}\n")
        self.f.write(f"secondary : {params['secondary']}\n")
        self.f.write(f"interferogram : {params['interferogram']}\n")
        self.f.write(f"flatten : {params.get('flatten', 'False')}\n")
        self.f.write(f"interferogram_prefix : {params.get('prefix', 'fine')}\n")
        self.f.write(f"overlap : {params.get('overlap', 'False')}\n")

        if params.get("misreg_az"):
            self.f.write(f"azimuth_misreg : {params['misreg_az']}\n")
        if params.get("misreg_rng"):
            self.f.write(f"range_misreg : {params['misreg_rng']}\n")
        self.f.write("###################################\n")

    def write_overlap_withdem(self, params: dict) -> None:
        """Write overlap_withDEM configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - interferogram: interferogram directory
            - reference_dir: reference directory
            - secondary_dir: secondary directory
            - overlap_dir: overlap output directory

        """
        self.function_counter += 1
        self.f.write("###################################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("overlap_withDEM : \n")
        self.f.write(f"interferogram : {params['interferogram']}\n")
        self.f.write(f"reference_dir : {params['reference_dir']}\n")
        self.f.write(f"secondary_dir : {params['secondary_dir']}\n")
        self.f.write(f"overlap_dir : {params['overlap_dir']}\n")
        self.f.write("###################################\n")

    def write_azimuth_misreg(self, params: dict) -> None:
        """Write estimateAzimuthMisreg configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - overlap_dir: overlap directory
            - out_azimuth: output azimuth misregistration file
            - coh_threshold: coherence threshold
            - plot: plot flag

        """
        self.function_counter += 1
        self.f.write("###################################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("estimateAzimuthMisreg : \n")
        self.f.write(f"overlap_dir : {params['overlap_dir']}\n")
        self.f.write(f"out_azimuth : {params['out_azimuth']}\n")
        self.f.write(f"coh_threshold : {params.get('coh_threshold', '0.85')}\n")
        self.f.write(f"plot : {params.get('plot', 'False')}\n")
        self.f.write("###################################\n")

    def write_range_misreg(self, params: dict) -> None:
        """Write estimateRangeMisreg configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - reference: reference directory
            - secondary: secondary directory
            - out_range: output range misregistration file
            - snr_threshold: SNR threshold

        """
        self.function_counter += 1
        self.f.write("###################################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("estimateRangeMisreg : \n")
        self.f.write(f"reference : {params['reference']}\n")
        self.f.write(f"secondary : {params['secondary']}\n")
        self.f.write(f"out_range : {params['out_range']}\n")
        self.f.write(f"snr_threshold : {params.get('snr_threshold', '10.0')}\n")
        self.f.write("###################################\n")

    def write_merge_bursts(self, params: dict) -> None:
        """Write mergeBursts configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - stack: stack directory (optional)
            - reference: reference directory
            - dirname: directory with burst products
            - name_pattern: pattern for burst products
            - outfile: output merged file path
            - method: merge method ('top', 'bot', 'avg')
            - aligned: aligned flag
            - valid_only: valid only flag
            - use_virtual: use virtual files flag
            - multilook: multilook flag
            - range_looks: range looks
            - azimuth_looks: azimuth looks
            - multilook_tool: multilook tool (optional)
            - no_data_value: no data value (optional)

        """
        self.function_counter += 1
        self.f.write("###################################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("mergeBursts : \n")

        if params.get("stack"):
            self.f.write(f"stack : {params['stack']}\n")

        self.f.write(f"inp_reference : {params['reference']}\n")
        self.f.write(f"dirname : {params['dirname']}\n")
        self.f.write(f"name_pattern : {params.get('name_pattern', 'fine*int')}\n")
        self.f.write(f"outfile : {params['outfile']}\n")
        self.f.write(f"method : {params.get('method', 'top')}\n")
        self.f.write(f"aligned : {params.get('aligned', 'True')}\n")
        self.f.write(f"valid_only : {params.get('valid_only', 'True')}\n")
        self.f.write(f"use_virtual_files : {params.get('use_virtual', 'True')}\n")
        self.f.write(f"multilook : {params.get('multilook', 'True')}\n")
        self.f.write(f"range_looks : {params.get('range_looks', '9')}\n")
        self.f.write(f"azimuth_looks : {params.get('azimuth_looks', '3')}\n")

        if params.get("multilook_tool"):
            self.f.write(f"multilook_tool : {params['multilook_tool']}\n")
        if params.get("no_data_value"):
            self.f.write(f"no_data_value : {params['no_data_value']}\n")
        self.f.write("###################################\n")

    def write_filter_coherence(self, params: dict) -> None:
        """Write FilterAndCoherence configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - input: input interferogram
            - filt: filtered output
            - coh: coherence output
            - strength: filter strength
            - slc1: SLC 1 path
            - slc2: SLC 2 path
            - complex_coh: complex coherence output (optional)
            - range_looks: range looks
            - azimuth_looks: azimuth looks

        """
        self.function_counter += 1
        self.f.write("###################################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("FilterAndCoherence : \n")
        self.f.write(f"input : {params['input']}\n")
        self.f.write(f"filt : {params['filt']}\n")
        self.f.write(f"coh : {params['coh']}\n")
        self.f.write(f"strength : {params.get('strength', '0.5')}\n")
        self.f.write(f"slc1 : {params['slc1']}\n")
        self.f.write(f"slc2 : {params['slc2']}\n")
        self.f.write(f"complex_coh : {params.get('complex_coh', '')}\n")
        self.f.write(f"range_looks : {params.get('range_looks', '9')}\n")
        self.f.write(f"azimuth_looks : {params.get('azimuth_looks', '3')}\n")
        self.f.write("###################################\n")

    def write_unwrap(self, params: dict) -> None:
        """Write unwrap configuration.

        Parameters
        ----------
        params : dict
            Dictionary with keys:
            - ifg: interferogram file
            - unw: unwrapped output file
            - coh: coherence file
            - nomcf: no MCF flag
            - reference: reference directory
            - defomax: max deformation
            - rlks: range looks
            - alks: azimuth looks
            - rmfilter: remove filter flag
            - method: unwrap method ('snaphu', 'icu')

        """
        self.function_counter += 1
        self.f.write("###################################\n")
        self.f.write(f"[Function-{self.function_counter}]\n")
        self.f.write("unwrap : \n")
        self.f.write(f"ifg : {params['ifg']}\n")
        self.f.write(f"unw : {params['unw']}\n")
        self.f.write(f"coh : {params['coh']}\n")
        self.f.write(f"nomcf : {params.get('nomcf', 'False')}\n")
        self.f.write(f"reference : {params['reference']}\n")
        self.f.write(f"defomax : {params.get('defomax', '2')}\n")
        self.f.write(f"rlks : {params.get('rlks', '9')}\n")
        self.f.write(f"alks : {params.get('alks', '3')}\n")
        self.f.write(f"rmfilter : {params.get('rmfilter', 'False')}\n")
        self.f.write(f"method : {params.get('method', 'snaphu')}\n")
        self.f.write("###################################\n")

    def finalize(self) -> None:
        """Close the configuration file."""
        self.f.close()
        logger.info("Config file written: %s", self.filepath)
