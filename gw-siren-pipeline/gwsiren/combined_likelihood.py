import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from gwsiren import CONFIG
from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0, H0LogLikelihood

logger = logging.getLogger(__name__)


@dataclass
class EventDataPackage:
    """Container holding data for a single gravitational-wave event."""

    event_id: str
    dl_samples: np.ndarray
    candidate_galaxies_df: pd.DataFrame


class CombinedLogLikelihood:
    """Combined log-likelihood for multi-event dark siren analysis."""

    def __init__(
        self,
        event_data_packages: List[EventDataPackage],
        global_h0_min: float = CONFIG.mcmc["prior_h0_min"],
        global_h0_max: float = CONFIG.mcmc["prior_h0_max"],
        global_alpha_min: float = CONFIG.mcmc.get("prior_alpha_min", -1.0),
        global_alpha_max: float = CONFIG.mcmc.get("prior_alpha_max", 1.0),
        sigma_v: float = CONFIG.cosmology["sigma_v_pec"],
        c_val: float = CONFIG.cosmology["c_light"],
        omega_m_val: float = CONFIG.cosmology["omega_m"],
    ) -> None:
        """Initialize the combined likelihood."""
        self.event_data_packages = event_data_packages
        self.global_h0_min = global_h0_min
        self.global_h0_max = global_h0_max
        self.global_alpha_min = global_alpha_min
        self.global_alpha_max = global_alpha_max

        self.single_event_likelihoods: List[H0LogLikelihood] = []
        for pkg in self.event_data_packages:
            df = pkg.candidate_galaxies_df
            likelihood = get_log_likelihood_h0(
                dL_gw_samples=pkg.dl_samples,
                host_galaxies_z=df["z"].values,
                host_galaxies_mass_proxy=df["mass_proxy"].values,
                host_galaxies_z_err=df["z_err"].values,
                sigma_v=sigma_v,
                c_val=c_val,
                omega_m_val=omega_m_val,
                h0_min=global_h0_min,
                h0_max=global_h0_max,
                alpha_min=global_alpha_min,
                alpha_max=global_alpha_max,
            )
            self.single_event_likelihoods.append(likelihood)

    def __call__(self, theta: List[float]) -> float:
        """Evaluate the combined log-likelihood for ``theta``.

        Args:
            theta: Parameter vector ``[H0, alpha]``.

        Returns:
            The sum of log-likelihoods over all events, or ``-np.inf`` if the
            parameters fall outside the global priors or any event likelihood
            evaluates to ``-np.inf``.
        """
        H0 = theta[0]
        alpha = theta[1]

        if not (self.global_h0_min <= H0 <= self.global_h0_max):
            return -np.inf
        if not (self.global_alpha_min <= alpha <= self.global_alpha_max):
            return -np.inf

        total_log_likelihood = 0.0
        for ll in self.single_event_likelihoods:
            log_l = ll([H0, alpha])
            if log_l == -np.inf:
                return -np.inf
            total_log_likelihood += log_l
        return total_log_likelihood

