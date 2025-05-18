"""Core pipeline logic for H0 estimation."""

from __future__ import annotations

import logging
import os
from typing import Iterable

import numpy as np
import pandas as pd
from emcee.interruptible_pool import InterruptiblePool

from gwsiren import CONFIG
from gwsiren.gw_data_fetcher import configure_astropy_cache, fetch_candidate_data
from gwsiren.event_data_extractor import extract_gw_event_parameters
from gwsiren.data.catalogs import (
    download_and_load_galaxy_catalog,
    clean_galaxy_catalog,
    apply_specific_galaxy_corrections,
    DEFAULT_GALAXY_CORRECTIONS,
    DEFAULT_RANGE_CHECKS,
)
from gwsiren.sky_analyzer import (
    generate_sky_map_and_credible_region,
    select_galaxies_in_sky_region,
    filter_galaxies_by_redshift,
    estimate_event_specific_z_max,
)
from gwsiren.h0_mcmc_analyzer import (
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
    DEFAULT_ALPHA_PRIOR_MIN,
    DEFAULT_ALPHA_PRIOR_MAX,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"
DEFAULT_EVENT_NAME = "GW170817"
CATALOG_TYPE = "glade+"
NSIDE_SKYMAP = CONFIG.skymap["default_nside"]
CDF_THRESHOLD = CONFIG.skymap["credible_level"]
HOST_Z_MAX_FALLBACK = 0.05


def save_h0_samples_and_print_summary(
    h0_samples: Iterable[Iterable[float]], event_name: str, num_hosts: int | None = None
) -> None:
    """Save posterior samples and log summary statistics.

    Args:
        h0_samples: Iterable of MCMC samples where the first column is ``H0`` and
            the second column is ``alpha``.
        event_name: Name of the GW event.
        num_hosts: Optional count of candidate host galaxies used in the
            likelihood.
    """
    samples = np.asarray(h0_samples)
    if samples.size == 0:
        logger.warning("No MCMC samples available for saving.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"H0_samples_{event_name}.npy")
    np.save(out_path, samples)
    logger.info("MCMC samples saved to %s", out_path)

    h0_vals = samples[:, 0]
    alpha_vals = samples[:, 1]

    q16_h0, q50_h0, q84_h0 = np.percentile(h0_vals, [16, 50, 84])
    q16_a, q50_a, q84_a = np.percentile(alpha_vals, [16, 50, 84])

    logger.info(
        "%s H0 = %.1f +%.1f / -%.1f km s⁻¹ Mpc⁻¹ (68%% C.I.)",
        event_name,
        q50_h0,
        q84_h0 - q50_h0,
        q50_h0 - q16_h0,
    )
    logger.info(
        "%s alpha = %.2f +%.2f / -%.2f (68%% C.I.)",
        event_name,
        q50_a,
        q84_a - q50_a,
        q50_a - q16_a,
    )
    if num_hosts is not None:
        logger.info("Number of candidate host galaxies: %d", num_hosts)


def run_full_analysis(
    event_name: str,
    perform_mcmc: bool = True,
    *,
    nside_skymap: int = NSIDE_SKYMAP,
    cdf_threshold: float = CDF_THRESHOLD,
    catalog_type: str = CATALOG_TYPE,
    host_z_max_fallback: float = HOST_Z_MAX_FALLBACK,
) -> dict:
    """Execute the full analysis workflow for a GW event.

    Args:
        event_name: Name of the gravitational wave event.
        perform_mcmc: Whether to run the MCMC stage.
        nside_skymap: HEALPix ``nside`` parameter for sky maps.
        cdf_threshold: Credible level threshold for the sky mask.
        catalog_type: Galaxy catalog identifier.
        host_z_max_fallback: Fallback redshift cut if estimation fails.

    Returns:
        Dictionary containing intermediate and final data products. If an error
        occurs, the ``error`` key will contain a message.
    """

    results = {
        "event_name": event_name,
        "nside_skymap": nside_skymap,
        "cdf_threshold": cdf_threshold,
        "host_z_max": None,
        "prob_map": None,
        "sky_mask": None,
        "sky_map_threshold_val": None,
        "glade_raw_df": None,
        "glade_cleaned_df": None,
        "spatially_selected_hosts_df": None,
        "redshift_filtered_hosts_df": None,
        "candidate_hosts_df": None,
        "sampler": None,
        "flat_h0_samples": None,
        "error": None,
    }

    try:
        cache_dir = configure_astropy_cache(CONFIG.fetcher["cache_dir_name"])
        if not cache_dir:
            raise RuntimeError("Failed to configure astropy cache")

        success, gw_data_obj = fetch_candidate_data(event_name, cache_dir)
        if not success:
            raise RuntimeError(f"Failed to fetch GW data: {gw_data_obj}")

        dL_samples, ra_samples, dec_samples = extract_gw_event_parameters(
            gw_data_obj, event_name
        )
        if dL_samples is None or ra_samples is None or dec_samples is None:
            raise RuntimeError("Essential GW parameters are missing")
        results["dL_samples"] = dL_samples
        results["ra_samples"] = ra_samples
        results["dec_samples"] = dec_samples

        host_z_max = estimate_event_specific_z_max(dL_samples)
        if host_z_max is None:
            host_z_max = host_z_max_fallback
        results["host_z_max"] = host_z_max

        prob_map, sky_mask, threshold = generate_sky_map_and_credible_region(
            ra_samples,
            dec_samples,
            nside=nside_skymap,
            cdf_threshold=cdf_threshold,
        )
        results["prob_map"] = prob_map
        results["sky_mask"] = sky_mask
        results["sky_map_threshold_val"] = threshold

        glade_raw = download_and_load_galaxy_catalog(catalog_type=catalog_type)
        if glade_raw.empty:
            raise RuntimeError("Galaxy catalogue failed to load")
        results["glade_raw_df"] = glade_raw

        glade_clean = clean_galaxy_catalog(
            glade_raw, range_filters=DEFAULT_RANGE_CHECKS
        )
        if glade_clean.empty:
            raise RuntimeError("Galaxy catalogue empty after cleaning")
        results["glade_cleaned_df"] = glade_clean

        if sky_mask.any():
            spatial_hosts = select_galaxies_in_sky_region(
                glade_clean, sky_mask, nside=nside_skymap
            )
        else:
            logger.warning("Sky mask empty. Selecting hosts from full catalogue.")
            spatial_hosts = glade_clean
        if spatial_hosts.empty:
            raise RuntimeError("No galaxies found in sky region")
        results["spatially_selected_hosts_df"] = spatial_hosts

        redshift_filtered = filter_galaxies_by_redshift(spatial_hosts, host_z_max)
        if redshift_filtered.empty:
            raise RuntimeError("No galaxies remaining after redshift cut")
        results["redshift_filtered_hosts_df"] = redshift_filtered

        candidate_hosts = apply_specific_galaxy_corrections(
            redshift_filtered, event_name, DEFAULT_GALAXY_CORRECTIONS
        )
        if candidate_hosts.empty:
            raise RuntimeError("No candidate host galaxies available after corrections")
        results["candidate_hosts_df"] = candidate_hosts

        if perform_mcmc:
            log_likelihood = get_log_likelihood_h0(
                dL_samples,
                candidate_hosts["z"].values,
                candidate_hosts["mass_proxy"].values,
                DEFAULT_SIGMA_V_PEC,
                DEFAULT_C_LIGHT,
                DEFAULT_OMEGA_M,
                DEFAULT_H0_PRIOR_MIN,
                DEFAULT_H0_PRIOR_MAX,
                DEFAULT_ALPHA_PRIOR_MIN,
                DEFAULT_ALPHA_PRIOR_MAX,
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
                    alpha_prior_min=DEFAULT_ALPHA_PRIOR_MIN,
                    alpha_prior_max=DEFAULT_ALPHA_PRIOR_MAX,
                    n_steps=DEFAULT_MCMC_N_STEPS,
                    pool=pool,
                )
            if sampler is None:
                raise RuntimeError("MCMC failed")
            results["sampler"] = sampler

            flat_samples = process_mcmc_samples(
                sampler,
                event_name,
                burnin=DEFAULT_MCMC_BURNIN,
                thin_by=DEFAULT_MCMC_THIN_BY,
                n_dim=DEFAULT_MCMC_N_DIM,
            )
            if flat_samples is None:
                raise RuntimeError("No samples produced by MCMC")
            results["flat_h0_samples"] = flat_samples
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error during analysis for %s: %s", event_name, exc)
        results["error"] = str(exc)

    return results


__all__ = [
    "run_full_analysis",
    "save_h0_samples_and_print_summary",
    "OUTPUT_DIR",
    "DEFAULT_EVENT_NAME",
    "CATALOG_TYPE",
    "NSIDE_SKYMAP",
    "CDF_THRESHOLD",
    "HOST_Z_MAX_FALLBACK",
]
