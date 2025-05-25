#!/usr/bin/env python
"""Run a multi-event dark siren analysis using global MCMC."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
from emcee.interruptible_pool import InterruptiblePool

from gwsiren import CONFIG
from gwsiren.config import (
    MEMCMCConfig,
    MEPriorBoundaries,
)
from gwsiren.multi_event_data_manager import prepare_event_data
from gwsiren.combined_likelihood import CombinedLogLikelihood
from gwsiren.global_mcmc import (
    run_global_mcmc,
    process_global_mcmc_samples,
    save_global_samples,
)

logger = logging.getLogger(__name__)


def _resolve_prior(priors: dict | None, key: str, default_min: float, default_max: float) -> tuple[float, float]:
    if priors and key in priors and isinstance(priors[key], MEPriorBoundaries):
        return priors[key].min, priors[key].max
    return default_min, default_max


def _resolve_value(obj: object | None, attr: str, fallback: float) -> float:
    if obj is not None and getattr(obj, attr) is not None:
        return getattr(obj, attr)
    return fallback


def execute_multi_event_analysis() -> None:
    """Execute the multi-event analysis pipeline using configuration from ``CONFIG``."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    me_settings = CONFIG.multi_event_analysis
    if me_settings is None:
        logger.error("multi_event_analysis section not found in config")
        return

    run_cfg = me_settings.run_settings
    label = run_cfg.run_label if run_cfg and run_cfg.run_label else "multi_event_run"
    base_out = run_cfg.base_output_directory if run_cfg and run_cfg.base_output_directory else "output/multi_event_runs"
    current_run_dir = Path(base_out) / label
    current_run_dir.mkdir(parents=True, exist_ok=True)

    all_packages: List = []
    for event_spec in me_settings.events_to_combine:
        try:
            package = prepare_event_data(asdict(event_spec))
        except Exception as exc:  # pragma: no cover - unexpected errors
            logger.error("Failed to prepare data for %s: %s", event_spec.event_id, exc)
            continue
        all_packages.append(package)

    if not all_packages:
        logger.error("No event data packages prepared. Exiting.")
        return

    prior_cfg = me_settings.priors
    h0_min, h0_max = _resolve_prior(prior_cfg, "H0", CONFIG.mcmc["prior_h0_min"], CONFIG.mcmc["prior_h0_max"])
    a_min, a_max = _resolve_prior(prior_cfg, "alpha", CONFIG.mcmc.get("prior_alpha_min", -1.0), CONFIG.mcmc.get("prior_alpha_max", 1.0))

    cosmo_cfg = me_settings.cosmology
    sigma_v = _resolve_value(cosmo_cfg, "sigma_v_pec", CONFIG.cosmology["sigma_v_pec"])
    c_light = _resolve_value(cosmo_cfg, "c_light", CONFIG.cosmology["c_light"])
    omega_m = _resolve_value(cosmo_cfg, "omega_m_val", CONFIG.cosmology["omega_m"])

    combined_ll = CombinedLogLikelihood(
        all_packages,
        global_h0_min=h0_min,
        global_h0_max=h0_max,
        global_alpha_min=a_min,
        global_alpha_max=a_max,
        sigma_v=sigma_v,
        c_val=c_light,
        omega_m_val=omega_m,
        force_non_vectorized=True,  # Force non-vectorized for better performance with many galaxies
    )

    mcmc_cfg: MEMCMCConfig | None = me_settings.mcmc
    n_walkers = mcmc_cfg.n_walkers if mcmc_cfg and mcmc_cfg.n_walkers is not None else CONFIG.mcmc["walkers"]
    n_steps = mcmc_cfg.n_steps if mcmc_cfg and mcmc_cfg.n_steps is not None else CONFIG.mcmc["steps"]
    burnin = mcmc_cfg.burnin if mcmc_cfg and mcmc_cfg.burnin is not None else CONFIG.mcmc["burnin"]
    thin_by = mcmc_cfg.thin_by if mcmc_cfg and mcmc_cfg.thin_by is not None else CONFIG.mcmc["thin_by"]

    init_cfg = {"H0": {"mean": 70.0, "std": 10.0}, "alpha": {"mean": 0.0, "std": 0.5}}
    if mcmc_cfg and mcmc_cfg.initial_pos_config:
        init_cfg = {
            k: {"mean": v.mean, "std": v.std}
            for k, v in mcmc_cfg.initial_pos_config.items()
        }

    n_dim = 2
    
    sampler = None
    # Always use serial mode to avoid pickling issues with backend module references
    # The likelihood objects contain self.xp (backend module) which cannot be pickled
    # for multiprocessing, regardless of which backend is actually used.
    logger.info(
        "Using serial mode (pool=None) to avoid pickling issues with likelihood objects "
        "that contain backend module references."
    )
    sampler = run_global_mcmc(
        combined_ll,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_dim=n_dim,
        initial_pos_config=init_cfg,
        pool=None,  # Always None to avoid pickling issues
    )

    samples = process_global_mcmc_samples(sampler, burnin=burnin, thin_by=thin_by, n_dim=n_dim)
    if samples is None:
        logger.error("Failed to process MCMC samples")
        return

    out_path = current_run_dir / "global_samples.npy"
    save_global_samples(samples, str(out_path))

    h0_vals = samples[:, 0]
    alpha_vals = samples[:, 1]
    q16_h0, q50_h0, q84_h0 = np.percentile(h0_vals, [16, 50, 84])
    q16_a, q50_a, q84_a = np.percentile(alpha_vals, [16, 50, 84])
    logger.info(
        "Combined %d events", len(all_packages)
    )
    logger.info(
        "H0 = %.1f +%.1f/-%.1f km s^-1 Mpc^-1 (68%% C.I.)",
        q50_h0,
        q84_h0 - q50_h0,
        q50_h0 - q16_h0,
    )
    logger.info(
        "alpha = %.2f +%.2f/-%.2f (68%% C.I.)",
        q50_a,
        q84_a - q50_a,
        q50_a - q16_a,
    )


if __name__ == "__main__":
    execute_multi_event_analysis()
