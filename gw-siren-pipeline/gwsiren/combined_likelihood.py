"""Utilities for combining event log-likelihoods."""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from gwsiren import CONFIG
from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0
from gwsiren.multi_event_data_manager import EventDataPackage

logger = logging.getLogger(__name__)

DEFAULT_GLOBAL_H0_MIN = CONFIG.mcmc["prior_h0_min"]
DEFAULT_GLOBAL_H0_MAX = CONFIG.mcmc["prior_h0_max"]
DEFAULT_GLOBAL_ALPHA_MIN = CONFIG.mcmc.get("prior_alpha_min", -1.0)
DEFAULT_GLOBAL_ALPHA_MAX = CONFIG.mcmc.get("prior_alpha_max", 1.0)
DEFAULT_SIGMA_V = CONFIG.cosmology["sigma_v_pec"]
DEFAULT_C_LIGHT = CONFIG.cosmology["c_light"]
DEFAULT_OMEGA_M = CONFIG.cosmology["omega_m"]


class CombinedLogLikelihood:
    """Combine log-likelihoods from multiple GW events.

    This callable object evaluates the joint log-likelihood for ``H0`` and
    ``alpha`` by summing individual event log-likelihoods.
    """

    def __init__(
        self,
        event_data_packages: List[EventDataPackage],
        global_h0_min: float = DEFAULT_GLOBAL_H0_MIN,
        global_h0_max: float = DEFAULT_GLOBAL_H0_MAX,
        global_alpha_min: float = DEFAULT_GLOBAL_ALPHA_MIN,
        global_alpha_max: float = DEFAULT_GLOBAL_ALPHA_MAX,
        sigma_v: float = DEFAULT_SIGMA_V,
        c_val: float = DEFAULT_C_LIGHT,
        omega_m_val: float = DEFAULT_OMEGA_M,
    ) -> None:
        """Initialize the combined likelihood.

        Args:
            event_data_packages: Data for each event to include in the
                combination.
            global_h0_min: Lower prior bound for ``H0``.
            global_h0_max: Upper prior bound for ``H0``.
            global_alpha_min: Lower prior bound for ``alpha``.
            global_alpha_max: Upper prior bound for ``alpha``.
            sigma_v: Peculiar velocity dispersion in km/s.
            c_val: Speed of light in km/s.
            omega_m_val: Matter density parameter.
        """
        self.event_data_packages = event_data_packages
        self.global_h0_min = global_h0_min
        self.global_h0_max = global_h0_max
        self.global_alpha_min = global_alpha_min
        self.global_alpha_max = global_alpha_max

        common_kwargs = {
            "sigma_v": sigma_v,
            "c_val": c_val,
            "omega_m_val": omega_m_val,
            "h0_min": self.global_h0_min,
            "h0_max": self.global_h0_max,
            "alpha_min": self.global_alpha_min,
            "alpha_max": self.global_alpha_max,
        }

        self.single_event_likelihoods = []
        for pkg in self.event_data_packages:
            ll = get_log_likelihood_h0(
                dL_gw_samples=pkg.dl_samples,
                host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
                host_galaxies_mass_proxy=pkg.candidate_galaxies_df[
                    "mass_proxy"
                ].values,
                host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
                **common_kwargs,
            )
            self.single_event_likelihoods.append(ll)

    def __call__(self, theta: List[float]) -> float:
        """Evaluate the combined log-likelihood.

        Args:
            theta: Sequence ``[H0, alpha]``.

        Returns:
            Sum of log-likelihoods over all events, or ``-np.inf`` if the
            parameters fall outside the global priors or any individual
            likelihood evaluates to ``-np.inf``.
        """
        H0, alpha = theta

        if not (self.global_h0_min <= H0 <= self.global_h0_max):
            logger.debug("H0 %s outside global prior", H0)
            return -np.inf
        if not (self.global_alpha_min <= alpha <= self.global_alpha_max):
            logger.debug("alpha %s outside global prior", alpha)
            return -np.inf

        total = 0.0
        for ll in self.single_event_likelihoods:
            val = ll(theta)
            if val == -np.inf:
                return -np.inf
            total += val
        return total


__all__ = ["CombinedLogLikelihood"]
