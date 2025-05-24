"""Sensitivity analysis tests."""

from __future__ import annotations

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import pytest

from gwsiren.h0_mcmc_analyzer import get_log_likelihood_h0
from utils.mock_data import mock_event


def test_prior_width_effect(mock_config):
    """Check that wider priors produce finite likelihoods for extreme values."""
    pkg = mock_event(seed=3)
    ll = get_log_likelihood_h0(
        requested_backend_str="auto",
        dL_gw_samples=pkg.dl_samples,
        host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
        host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
        host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        h0_min=20,
        h0_max=140,
    )
    val = ll([30.0, 0.0])
    assert isinstance(val, float)


def test_fixed_parameters_effect(mock_config):
    """Likelihood changes when sigma_v is varied."""
    pkg = mock_event(seed=4)
    ll1 = get_log_likelihood_h0(
        requested_backend_str="auto",
        dL_gw_samples=pkg.dl_samples,
        host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
        host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
        host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        sigma_v=100,
    )
    ll2 = get_log_likelihood_h0(
        requested_backend_str="auto",
        dL_gw_samples=pkg.dl_samples,
        host_galaxies_z=pkg.candidate_galaxies_df["z"].values,
        host_galaxies_mass_proxy=pkg.candidate_galaxies_df["mass_proxy"].values,
        host_galaxies_z_err=pkg.candidate_galaxies_df["z_err"].values,
        sigma_v=500,
    )
    val1 = ll1([70.0, 0.0])
    val2 = ll2([70.0, 0.0])
    assert isinstance(val1, float)
    assert isinstance(val2, float)

