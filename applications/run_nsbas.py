import configparser
import importlib
from pathlib import Path

import numpy as np
import torch

import faninsar as fis
from faninsar import NSBAS, datasets, query, setup_logger

logger = setup_logger("test2.log", log_level="info", log_name="FanInSAR")


def parse_default_config(config: configparser.ConfigParser) -> dict:
    """Parse DEFAULT section in config file."""
    params = {}
    params["dateset_name"] = config["DEFAULT"]["dataset_name"]
    params["dataset_file"] = config["DEFAULT"]["dataset_file"]
    params["home_dir"] = Path(config["DEFAULT"]["home_dir"])
    params["dataset_args"] = eval(config["DEFAULT"]["dataset_args"])

    if not config["DEFAULT"]["output_dir"]:
        output_dir = Path().cwd()
    else:
        output_dir = Path(output_dir)
    params["output_dir"] = output_dir

    params["ref_file"] = Path(config["DEFAULT"]["ref_file"])

    roi_str = eval(config["DEFAULT"]["roi"])
    roi_crs = config["DEFAULT"]["roi_crs"]
    roi = None
    if len(roi_str) == 4:
        roi = query.BoundingBox(*roi_str, crs=roi_crs)
    params["roi"] = roi

    # datasets
    if params.dateset_name not in datasets.available:
        module = importlib.import_module(params.dataset_file)
        Dataset = getattr(module, params.dateset_name)
    else:
        Dataset = datasets.get_dataset(params.dateset_name)

    params["Dataset"] = Dataset

    return params


def parse_NSBAS_config(config: configparser.ConfigParser) -> dict:
    """Parse NSBAS section in config file."""
    params = {}
    params["coherence_threshold"] = config["NSBAS"].getfloat("coherence_threshold")
    params["wls"] = config["NSBAS"].getboolean("wls")

    model_name = config["NSBAS"]["model_name"]
    model_file = config["NSBAS"]["model_file"]

    if model_name in NSBAS.available_models:
        model = NSBAS.get_model(model_name)
    else:
        module = importlib.import_module(model_file)
        model = getattr(module, model_name)

    params["model"] = model

    return params


def parse_loops_config(config: configparser.ConfigParser) -> dict:
    """Parse loops section in config file."""
    params = {}
    params["max_acquisition"] = config["loops"].getint("max_acquisition")
    params["interval_days"] = config["loops"].getint("interval_days")
    params["loop_dir"] = Path(config["loops"]["loop_dir"])
    return params


def parse_config(config_file):
    """Parse config file."""
    config = configparser.ConfigParser()
    config.read(config_file)

    cfg_default = parse_default_config(config)
    cfg_nsbas = parse_NSBAS_config(config)
    cfg_loops = parse_loops_config(config)

    cfg = {**cfg_default, **cfg_nsbas, **cfg_loops}

    return cfg


def correct_phase_by_loops(ifg_values, pairs, loops, max_acquisition, interval_days):
    """Correct phase by loops."""
    # filter pairs first to reduce high-cost computation in to_loops
    pairs_idx = pairs.days <= interval_days * max_acquisition
    pairs_used = pairs[pairs_idx]
    phi = ifg_values[pairs_idx]

    # calculate loops using filtered pairs
    loops = pairs_used.to_loops(
        max_acquisition=max_acquisition, interval_days=interval_days
    )

    # keep the loops that made up by the nearest-acquisition pairs
    loops_used = [
        loop
        for loop in loops
        if ((loop.pairs.days == interval_days).sum() == len(loop.pairs) - 1)
    ]
    C = np.asarray(loops_used)

    # remove the pairs not used in loops
    mask_u = (C == 0).sum(axis=0) != C.shape[0]
    C = C[:, mask_u]
    pairs_u = pairs_used[mask_u]
    phi = phi[mask_u]

    U1 = NSBAS.calculate_U(C, pairs_u, phi)
    d_c = phi - 2 * np.pi * U1

    return d_c


def main(config_file):
    """Run NSBAS."""

    cfg = parse_config(config_file)

    ds_unw = cfg["Dataset"](cfg["home_dir"], **cfg["dataset_args"])

    pairs = ds_unw.pairs
    pdc = fis.PhaseDeformationConverter(frequency=5.405)
    ifg_values = pdc.phase2deformation(ifg_values)

    ################# loop correction ###################
    # filter pairs first to reduce high-cost computation in to_loops

    do_loop_correction(cfg, pairs, ifg_values)

    nsbas = NSBAS(ds_unw, geo_query)
    nsbas.run()
    nsbas.save()


def do_loop_correction(cfg, pairs, ifg_values):
    pairs_idx = pairs.days <= cfg.interval_days * cfg.max_acquisition
    pairs_used = pairs[pairs_idx]
    d = ifg_values[pairs_idx]

    # calculate loops using filtered pairs
    loops = pairs_used.to_loops(
        max_acquisition=cfg.max_acquisition, interval_days=cfg.interval_days
    )

    # keep the loops that made up by the nearest-acquisition pairs
    loops_used = [
        loop
        for loop in loops
        if ((loop.pairs.days == cfg.interval_days).sum() == len(loop.pairs) - 1)
    ]
    Cc = np.asarray(loops_used)

    # remove the pairs not used in loops
    mask_u = (Cc == 0).sum(axis=0) != Cc.shape[0]
    Cc1 = Cc[:, mask_u]
    pairs_u = pairs_used[mask_u]
    d_u = d[mask_u]

    U1 = NSBAS.calculate_U(Cc1, pairs_u, d_u)
    d_c = d_u - 2 * np.pi * U1

