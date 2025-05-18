#!/usr/bin/env python
"""End-to-end pipeline for estimating the Hubble constant without visualizations.

This script extracts the core analysis logic from ``viz.py`` but omits any
plotting. It downloads GW event data, processes the galaxy catalogue, performs
sky localisation and candidate host selection, and finally runs the MCMC
analysis to obtain samples of ``H0``. The resulting samples are saved to
``output/H0_samples_<event>.npy`` and summary statistics are printed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from emcee.interruptible_pool import InterruptiblePool
from gwsiren import CONFIG

from gw_data_fetcher import (
    configure_astropy_cache,
    fetch_candidate_data,
    DEFAULT_CACHE_DIR_NAME,
)
from event_data_extractor import extract_gw_event_parameters
from gwsiren.data.catalogs import (
    download_and_load_galaxy_catalog,
    clean_galaxy_catalog,
    apply_specific_galaxy_corrections,
    DEFAULT_GALAXY_CORRECTIONS,
    DEFAULT_RANGE_CHECKS,
)
from sky_analyzer import (
    generate_sky_map_and_credible_region,
    select_galaxies_in_sky_region,
    filter_galaxies_by_redshift,
    estimate_event_specific_z_max,
)
from h0_mcmc_analyzer import (
    get_log_likelihood_h0,
    run_mcmc_h0,
    process_mcmc_samples,
    DEFAULT_SIGMA_V_PEC,
    DEFAULT_C_LIGHT,
    DEFAULT_OMEGA_M,
    DEFAULT_MCMC_N_WALKERS,
    DEFAULT_MCMC_N_DIM,
    DEFAULT_MCMC_INITIAL_H0_MEAN,
    DEFAULT_MCMC_INITIAL_H0_STD,
    DEFAULT_MCMC_N_STEPS,
    DEFAULT_MCMC_BURNIN,
    DEFAULT_MCMC_THIN_BY,
    DEFAULT_H0_PRIOR_MIN,
    DEFAULT_H0_PRIOR_MAX,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"
DEFAULT_EVENT_NAME = "GW170817"
CATALOG_TYPE = "glade+"
NSIDE_SKYMAP = CONFIG.skymap["default_nside"]
CDF_THRESHOLD = CONFIG.skymap["credible_level"]
HOST_Z_MAX_FALLBACK = 0.05


def save_h0_samples_and_print_summary(
    h0_samples: Iterable[float], event_name: str, num_hosts: int | None = None
) -> None:
    """Save ``H0`` samples and print summary statistics.

    Args:
        h0_samples: Array of ``H0`` samples.
        event_name: Name of the gravitational wave event.
        num_hosts: Optional number of candidate host galaxies used.
    """
    samples = np.asarray(h0_samples)
    if samples.size == 0:
        logger.warning("No H0 samples available for saving.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"H0_samples_{event_name}.npy")
    np.save(out_path, samples)
    logger.info("H0 samples saved to %s", out_path)

    q16, q50, q84 = np.percentile(samples, [16, 50, 84])
    logger.info(
        "%s H0 = %.1f +%.1f / -%.1f km s⁻¹ Mpc⁻¹ (68%% C.I.)",
        event_name,
        q50,
        q84 - q50,
        q50 - q16,
    )
    if num_hosts is not None:
        logger.info("Number of candidate host galaxies: %d", num_hosts)


def main() -> None:
    """Run the end-to-end analysis pipeline."""
    parser = argparse.ArgumentParser(description="Run H0 estimation pipeline")
    parser.add_argument(
        "event_name",
        nargs="?",
        default=DEFAULT_EVENT_NAME,
        help=f"GW event name (default: {DEFAULT_EVENT_NAME})",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_dir = configure_astropy_cache(DEFAULT_CACHE_DIR_NAME)
    if not cache_dir:
        logger.critical("Failed to configure astropy cache.")
        sys.exit(1)

    event_name = args.event_name
    logger.info("Starting analysis for %s", event_name)

    success, gw_data_obj = fetch_candidate_data(event_name, cache_dir)
    if not success:
        logger.error("Failed to fetch GW data: %s", gw_data_obj)
        sys.exit(1)

    dL_samples, ra_samples, dec_samples = extract_gw_event_parameters(
        gw_data_obj, event_name
    )
    if dL_samples is None or ra_samples is None or dec_samples is None:
        logger.error("Essential GW parameters are missing. Exiting.")
        sys.exit(1)

    host_z_max = estimate_event_specific_z_max(
        dL_samples,
        percentile_dL=95.0,
        z_margin_factor=1.2,
        min_z_max_val=0.01,
        max_z_max_val=0.3,
    )
    logger.info("Using host_z_max=%.4f", host_z_max)

    prob_map, sky_mask, _ = generate_sky_map_and_credible_region(
        ra_samples, dec_samples, nside=NSIDE_SKYMAP, cdf_threshold=CDF_THRESHOLD
    )

    glade_raw = download_and_load_galaxy_catalog(catalog_type=CATALOG_TYPE)
    if glade_raw.empty:
        logger.error("Galaxy catalogue failed to load.")
        sys.exit(1)

    glade_clean = clean_galaxy_catalog(glade_raw, range_filters=DEFAULT_RANGE_CHECKS)
    if glade_clean.empty:
        logger.error("Galaxy catalogue empty after cleaning.")
        sys.exit(1)

    if sky_mask.any():
        spatial_hosts = select_galaxies_in_sky_region(
            glade_clean, sky_mask, nside=NSIDE_SKYMAP
        )
    else:
        logger.warning("Sky mask empty. Selecting hosts from full catalogue.")
        spatial_hosts = glade_clean

    if spatial_hosts.empty:
        logger.error("No galaxies found in sky region.")
        sys.exit(1)

    redshift_filtered = filter_galaxies_by_redshift(spatial_hosts, host_z_max)
    if redshift_filtered.empty:
        logger.error("No galaxies remaining after redshift cut.")
        sys.exit(1)

    candidate_hosts = apply_specific_galaxy_corrections(
        redshift_filtered, event_name, DEFAULT_GALAXY_CORRECTIONS
    )
    if candidate_hosts.empty:
        logger.error("No candidate host galaxies available after corrections.")
        sys.exit(1)

    logger.info("%d candidate host galaxies identified", len(candidate_hosts))

    log_likelihood = get_log_likelihood_h0(
        dL_samples,
        candidate_hosts["z"].values,
        DEFAULT_SIGMA_V_PEC,
        DEFAULT_C_LIGHT,
        DEFAULT_OMEGA_M,
        DEFAULT_H0_PRIOR_MIN,
        DEFAULT_H0_PRIOR_MAX,
    )

    n_cores = os.cpu_count() or 1
    with InterruptiblePool(n_cores) as pool:
        sampler = run_mcmc_h0(
            log_likelihood,
            event_name,
            n_walkers=DEFAULT_MCMC_N_WALKERS,
            n_dim=DEFAULT_MCMC_N_DIM,
            initial_h0_mean=DEFAULT_MCMC_INITIAL_H0_MEAN,
            initial_h0_std=DEFAULT_MCMC_INITIAL_H0_STD,
            n_steps=DEFAULT_MCMC_N_STEPS,
            pool=pool,
        )

    if sampler is None:
        logger.error("MCMC failed.")
        sys.exit(1)

    flat_samples = process_mcmc_samples(
        sampler,
        event_name,
        burnin=DEFAULT_MCMC_BURNIN,
        thin_by=DEFAULT_MCMC_THIN_BY,
        n_dim=DEFAULT_MCMC_N_DIM,
    )
    if flat_samples is None:
        logger.error("No samples produced by MCMC.")
        sys.exit(1)

    save_h0_samples_and_print_summary(flat_samples, event_name, len(candidate_hosts))
    logger.info("Analysis completed for %s", event_name)


if __name__ == "__main__":
    main()
