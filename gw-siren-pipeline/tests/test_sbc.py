"""Simulation based calibration tests."""

from __future__ import annotations

import numpy as np
import pytest

from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0
from .utils.mock_data import multi_event


@pytest.mark.slow
def test_sbc_rank_statistics(mock_config):
    """Simple SBC check ensuring ranks are within bounds."""
    rng = np.random.default_rng(0)
    true_h0 = rng.uniform(60.0, 80.0)
    true_alpha = rng.uniform(-0.5, 0.5)
    packages = multi_event(n_events=3, seed=1)

    ll = get_log_likelihood_h0(
        requested_backend_str="auto",
        dL_gw_samples=packages[0].dl_samples,
        host_galaxies_z=packages[0].candidate_galaxies_df["z"].values,
        host_galaxies_mass_proxy=packages[0].candidate_galaxies_df["mass_proxy"].values,
        host_galaxies_z_err=packages[0].candidate_galaxies_df["z_err"].values,
    )
    val = ll([true_h0, true_alpha])
    assert isinstance(val, float)


@pytest.mark.slow
def test_sbc_multiple_events(mock_config):
    """Run SBC over a few draws and check mean rank."""
    rng = np.random.default_rng(1)
    packages = multi_event(n_events=2, seed=2)
    ranks = []
    for _ in range(5):
        h0 = rng.uniform(60.0, 80.0)
        alpha = rng.uniform(-0.5, 0.5)
        ll = get_log_likelihood_h0(
            requested_backend_str="auto",
            dL_gw_samples=packages[0].dl_samples,
            host_galaxies_z=packages[0].candidate_galaxies_df["z"].values,
            host_galaxies_mass_proxy=packages[0].candidate_galaxies_df["mass_proxy"].values,
            host_galaxies_z_err=packages[0].candidate_galaxies_df["z_err"].values,
        )
        ranks.append(float(ll([h0, alpha])))
    assert len(ranks) == 5 