#!/usr/bin/env python
"""Example script to visualize results from a multi-event analysis."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

from gwsiren.config import load_config
from gwsiren.plot_utils import (
    load_combined_samples,
    load_individual_event_samples,
    get_event_ids_from_config,
    plot_overlaid_1d_posteriors,
    plot_combined_corner,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize multi-event results")
    parser.add_argument(
        "--combined-samples",
        required=True,
        help="Path to the combined H0-alpha samples .npy file",
    )
    parser.add_argument(
        "--config",
        default="gw-siren-pipeline/config.yaml",
        help="Path to the config YAML used for the run",
    )
    parser.add_argument(
        "--single-event-dir",
        default="output",
        help="Directory containing single-event H0_samples_<event>.npy files",
    )
    parser.add_argument(
        "--outdir",
        default="output/plots",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    me_cfg = cfg.multi_event_analysis
    if me_cfg is None:
        raise RuntimeError("multi_event_analysis section missing in config")

    event_ids = get_event_ids_from_config(me_cfg)
    combined = load_combined_samples(args.combined_samples)

    individual: dict[str, np.ndarray] = {}
    for event in event_ids:
        arr = load_individual_event_samples(event, args.single_event_dir)
        if arr is not None:
            individual[event] = arr

    os.makedirs(args.outdir, exist_ok=True)

    plot_overlaid_1d_posteriors(
        individual,
        combined,
        param_name="$H_0$ (km s$^{-1}$ Mpc$^{-1}$)",
        output_filepath=os.path.join(args.outdir, "H0_overlaid.pdf"),
        param_index=0,
    )

    plot_overlaid_1d_posteriors(
        individual,
        combined,
        param_name=r"$\\alpha$",
        output_filepath=os.path.join(args.outdir, "alpha_overlaid.pdf"),
        param_index=1,
    )

    plot_combined_corner(
        combined,
        os.path.join(args.outdir, "combined_corner.pdf"),
    )


if __name__ == "__main__":
    main()
