"""Utilities for running and processing global MCMC analyses."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import emcee

from gwsiren.h0_mcmc_analyzer import (
    DEFAULT_MCMC_N_DIM,
    DEFAULT_MCMC_N_WALKERS,
    DEFAULT_MCMC_N_STEPS,
    DEFAULT_MCMC_BURNIN,
    DEFAULT_MCMC_THIN_BY,
)

logger = logging.getLogger(__name__)


def run_global_mcmc(
    combined_log_likelihood_obj: Callable[[np.ndarray], float],
    n_walkers: int = DEFAULT_MCMC_N_WALKERS,
    n_steps: int = DEFAULT_MCMC_N_STEPS,
    n_dim: int = DEFAULT_MCMC_N_DIM,
    initial_pos_config: Optional[dict[str, dict[str, float]]] = None,
    pool: Optional[Any] = None,
) -> Optional[emcee.EnsembleSampler]:
    """Run MCMC on a combined multi-event likelihood.
    This likelihood should be constructed using a list of EventDataPackage objects.

    Args:
        combined_log_likelihood_obj: Callable returning the log-likelihood for a
            parameter vector ``[H0, alpha]``.
        n_walkers: Number of walkers in the ensemble sampler.
        n_steps: Number of MCMC steps to run.
        n_dim: Dimension of the parameter space. Should be ``2``.
        initial_pos_config: Dictionary with mean/std for walker initialisation.
        pool: Optional multiprocessing pool for parallel likelihood evaluation.

    Returns:
        The ``emcee.EnsembleSampler`` instance after the run, or ``None`` on
        failure.
    """

    if initial_pos_config is None:
        initial_pos_config = {
            "H0": {"mean": 70.0, "std": 10.0},
            "alpha": {"mean": 0.0, "std": 0.5},
        }

    logger.info(
        "Running global MCMC (%d walkers, %d steps)...", n_walkers, n_steps
    )

    h0_mean = initial_pos_config.get("H0", {}).get("mean", 70.0)
    h0_std = initial_pos_config.get("H0", {}).get("std", 10.0)
    alpha_mean = initial_pos_config.get("alpha", {}).get("mean", 0.0)
    alpha_std = initial_pos_config.get("alpha", {}).get("std", 0.5)

    pos_h0 = h0_mean + h0_std * np.random.randn(n_walkers, 1)
    pos_alpha = alpha_mean + alpha_std * np.random.randn(n_walkers, 1)

    # Respect prior boundaries if attributes are available on the likelihood obj
    h0_min = getattr(combined_log_likelihood_obj, "h0_min", None)
    h0_max = getattr(combined_log_likelihood_obj, "h0_max", None)
    if h0_min is not None and h0_max is not None:
        out_of_bounds = (pos_h0 < h0_min) | (pos_h0 > h0_max)
        pos_h0[out_of_bounds] = np.random.uniform(h0_min, h0_max, size=out_of_bounds.sum())

    alpha_min = getattr(combined_log_likelihood_obj, "alpha_min", None)
    alpha_max = getattr(combined_log_likelihood_obj, "alpha_max", None)
    if alpha_min is not None and alpha_max is not None:
        out_of_bounds = (pos_alpha < alpha_min) | (pos_alpha > alpha_max)
        pos_alpha[out_of_bounds] = np.random.uniform(
            alpha_min, alpha_max, size=out_of_bounds.sum()
        )

    walkers0 = np.hstack((pos_h0, pos_alpha))

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        combined_log_likelihood_obj,
        moves=emcee.moves.StretchMove(),
        pool=pool,
    )

    try:
        sampler.run_mcmc(walkers0, n_steps, progress=True)
    except ValueError as exc:
        logger.error("ValueError during global MCMC: %s", exc)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Unexpected error during global MCMC: %s", exc)
        return None

    logger.info("Global MCMC run completed.")
    return sampler


def process_global_mcmc_samples(
    sampler_obj: Optional[emcee.EnsembleSampler],
    burnin: int = DEFAULT_MCMC_BURNIN,
    thin_by: int = DEFAULT_MCMC_THIN_BY,
    n_dim: int = DEFAULT_MCMC_N_DIM,
) -> Optional[np.ndarray]:
    """Process raw MCMC chains by applying burn-in and thinning.

    Args:
        sampler_obj: Sampler returned by :func:`run_global_mcmc`.
        burnin: Number of initial steps to discard.
        thin_by: Thinning factor for the chains.
        n_dim: Expected dimensionality of the samples.

    Returns:
        Array of processed samples with shape ``(N, n_dim)`` or ``None`` if
        processing fails.
    """
    if sampler_obj is None:
        logger.warning("Sampler object is None; cannot process samples.")
        return None

    logger.info("Processing global MCMC samples (burn-in=%d, thin=%d)...", burnin, thin_by)
    try:
        flat = sampler_obj.get_chain(discard=burnin, thin=thin_by, flat=True)
        if flat.size == 0 or flat.shape[1] != n_dim:
            logger.warning("Processed sample array has unexpected shape %s", flat.shape)
            return None
        return flat
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to process global MCMC samples: %s", exc)
        return None


def save_global_samples(samples: np.ndarray, out_path: str) -> None:
    """Save processed MCMC samples to a NumPy ``.npy`` file."""
    if samples is None or samples.size == 0:
        logger.warning("No samples to save.")
        return
    np.save(out_path, samples)
    logger.info("Saved MCMC samples to %s", out_path)


__all__ = [
    "run_global_mcmc",
    "process_global_mcmc_samples",
    "save_global_samples",
]

